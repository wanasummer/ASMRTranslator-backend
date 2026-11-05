from modules.storage import initialize_project, read_text, read_bytes, read_json, save_text, save_json, contains, address
from tests.llm_support import subtitle_stage_factory
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch

import httpx

from modules.asr import create_asr, DocumentASRAdapter
from modules.asr.moss_adapter import MossASRAdapter, merge_slice_blocks
from modules.analysis_plugins import AnalysisPluginError
from modules.error import ASRError
from modules import preflight
from modules.pipeline import run_pipeline
from modules.subtitle_processing.timeline import _build_sentence_blocks
from modules.vad.models import SpeechRegion
from modules.vad.contracts import AsrSlice
from modules.vtt_parser import VTTBlock


def payload():
    return {"language": "ja", "timeline_source": "model", "blocks": [
        {"index": 1, "start_ms": 0, "end_ms": 700, "text": "こんにちは。", "speaker": "S01"},
        {"index": 2, "start_ms": 600, "end_ms": 900, "text": "はい。", "speaker": "S02"}]}


class MossAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_document_adapter_retains_native_times_and_speakers(self):
        client = Mock()
        client.request.return_value = payload(), 1000
        adapter = create_asr("moss-transcribe-diarize", client=client)
        self.assertIsInstance(adapter, DocumentASRAdapter)
        self.assertTrue(adapter.uses_project_vad())
        parent = AsrSlice("slice-1", 10000, 11000, 10100, 10900, ("region-1",))
        region = SpeechRegion(10, 11, asr_slice=parent)
        result = await adapter.transcribe_document("input.mp3", [region])
        client.request.assert_called_once_with("/transcribe", "input.mp3", {"language": "ja"},
                                               start_ms=10000, end_ms=11000)
        self.assertEqual([block.speaker for block in result.blocks], ["S01", "S02"])
        self.assertEqual(result.blocks[1].start_ms, 10600)
        self.assertEqual(result.blocks[1].end_ms, 10900)
        self.assertEqual(result.blocks[1].asr_slice, parent)
        self.assertFalse(result.japanese_cleanup_required)
        self.assertEqual((region.start, region.end), (10, 11))

    async def test_crops_restore_times_and_scope_speakers_without_mutating_responses(self):
        client = Mock()
        original = payload()
        client.request.return_value = original, 1000
        result = await MossASRAdapter(client=client).transcribe_document(
            "input.wav", [SpeechRegion(1, 2), SpeechRegion(10, 11)])
        self.assertEqual([b.index for b in result.blocks], [0, 1, 2, 3])
        self.assertEqual([b.start_ms for b in result.blocks], [1000, 1600, 10000, 10600])
        self.assertEqual([b.speaker for b in result.blocks], ["S01", "S02", "S03", "S04"])
        self.assertEqual([call.kwargs for call in client.request.call_args_list], [
            {"start_ms": 1000, "end_ms": 2000}, {"start_ms": 10000, "end_ms": 11000}])
        self.assertEqual(original, payload())
        self.assertEqual(result.raw_response["speaker_scope"], "slice")
        self.assertEqual(result.raw_response["slices"][1]["speaker_map"], {"S01": "S03", "S02": "S04"})

    async def test_missing_or_invalid_vad_never_uploads_whole_recording(self):
        client = Mock()
        for slices in (None, [], [SpeechRegion(-1, 2)], [SpeechRegion(0, float("inf"))],
                       [SpeechRegion(3, 4), SpeechRegion(1, 2)]):
            with self.assertRaises(ASRError):
                await MossASRAdapter(client=client).transcribe_document("input.wav", slices)
        client.request.assert_not_called()

    async def test_crop_duration_mismatch_is_rejected(self):
        client = Mock()
        client.request.return_value = payload(), 2000
        with self.assertRaisesRegex(ASRError, "长度"):
            await MossASRAdapter(client=client).transcribe_document("input.wav", [SpeechRegion(1, 2)])

    async def test_overlapping_crops_dedupe_only_matching_cross_crop_observations(self):
        first = AsrSlice("one", 0, 2000, 0, 2000, ())
        second = AsrSlice("two", 1000, 3000, 1000, 3000, ())
        blocks = [
            VTTBlock(0, 1200, 1800, "はい。", speaker="S01", asr_slice=first),
            VTTBlock(1, 1200, 1800, "はい。", speaker="S02", asr_slice=first),
            VTTBlock(2, 1250, 1850, "はい。", speaker="S03", asr_slice=second),
            VTTBlock(3, 1300, 1900, "こんにちは。", speaker="S04", asr_slice=second),
            VTTBlock(4, 2200, 2800, "はい。", speaker="S03", asr_slice=second),
        ]
        result = merge_slice_blocks(blocks)
        self.assertEqual([b.speaker for b in result], ["S01", "S02", "S04", "S03"])
        self.assertEqual([b.index for b in result], [0, 1, 2, 3])
        self.assertEqual(blocks[3].index, 3)

    async def test_invalid_or_failed_service_is_a_typed_error(self):
        client = Mock()
        client.request.side_effect = AnalysisPluginError("wrong identity")
        with self.assertRaisesRegex(ASRError, "1/1.*wrong identity"):
            await MossASRAdapter(client=client).transcribe_document("input.wav", [SpeechRegion(0, 1)])
        for row in ({"start_ms": True}, {"end_ms": 1001}, {"speaker": "person"}, {"index": 2}, {"text": ""}):
            body = payload()
            body["blocks"][0].update(row)
            with self.assertRaises(ASRError):
                MossASRAdapter.parse_result(body, 1000)

    async def test_health_requires_exact_protocol_and_ready_model(self):
        health = {"service": "moss-transcribe-diarize", "api_version": "moss-transcribe-diarize/1", "ready": True, "status": "ok"}
        config = {"asrEngine": "moss-transcribe-diarize", "asrBaseUrl": "http://127.0.0.1", "asrPort": 8008}
        for override, expected in (({}, True), ({"service": "different"}, False), ({"ready": False}, False)):
            client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={**health, **override})))
            with patch("modules.preflight.async_client", return_value=client):
                result = await preflight.run_preflight_check("asr", config)
            self.assertEqual(result.success, expected, result.message)


class MossPipelineTests(unittest.TestCase):
    def test_vad_crops_reach_llm_and_cache_without_secondary_detection_or_cleanup(self):
        client = Mock()
        client.request.return_value = payload(), 1000
        adapter = MossASRAdapter(client=client)
        preserved = []

        def translate(blocks, config, *, preserve_source_timeline, output_dir):
            preserved.append(preserve_source_timeline)
            # Exercise a translator returning several strings for each native block.
            parts = {block.index: [block.text[:2], block.text[2:]] for block in blocks}
            return _build_sentence_blocks(blocks, parts, preserve_source_timeline=preserve_source_timeline), {"enabled": True}

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"fixture")
            output = Path(directory) / "work"
            with patch("modules.asr.create_asr", return_value=adapter), \
                    patch("modules.pipeline._run_vad_stage", return_value=[SpeechRegion(10, 11)]) as vad, \
                    patch("modules.pipeline._create_japanese_cleaner_with_fallback") as cleaner, \
                    patch("modules.subtitle_processing.create_subtitle_stage", side_effect=subtitle_stage_factory(translate)), \
                    patch("modules.detect.create_detect_stage") as detect, redirect_stdout(StringIO()):
                for _ in range(2):
                    result = run_pipeline(str(source), "", str(output), tts_backend="subtitles-only",
                        asr_backend="moss-transcribe-diarize", asr_kwargs={"endpoint": "http://plugin"},
                        japanese_cleaner="regex-only", detect_enabled=True, llm_kwargs={"provider": "test"})
                    self.assertEqual([block.speaker for block in result["blocks"]], ["S01", "S02"])
                    self.assertEqual(result["blocks"][0].start_ms, 10000)
                    self.assertEqual(result["blocks"][0].asr_slice.id, "moss-slice-000001")
                    self.assertIn("S01：こんにちは。", read_text(result["final_vtt"]))
            vad.assert_called_once()
            detect.assert_not_called()
            cleaner.assert_not_called()
            client.request.assert_called_once()
            self.assertEqual(preserved, [True, True])
            raw = json.loads(read_text(output / "asr_subtitles/raw_response.json"))
            self.assertEqual(raw["slices"][0]["response"], payload())
            self.assertEqual(raw["slices"][0]["start_ms"], 10000)
            self.assertFalse(contains(output / "asr_subtitles/cleaned.vtt"))
