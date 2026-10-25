import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch

import httpx

from modules.asr import create_asr, DocumentASRAdapter
from modules.asr.moss_adapter import MossASRAdapter
from modules.analysis_plugins import AnalysisPluginError
from modules.error import ASRError
from modules import preflight
from modules.pipeline import run_pipeline
from modules.llm_enhancer import _build_sentence_blocks


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
        self.assertFalse(adapter.uses_project_vad())
        result = await adapter.transcribe_document("input.mp3", [(10, 20)])
        client.request.assert_called_once_with("/transcribe", "input.mp3", {"language": "ja"})
        self.assertEqual([block.speaker for block in result.blocks], ["S01", "S02"])
        self.assertEqual(result.blocks[1].start_ms, 600)

    async def test_invalid_or_failed_service_is_a_typed_error(self):
        client = Mock()
        client.request.side_effect = AnalysisPluginError("wrong identity")
        with self.assertRaises(ASRError):
            await MossASRAdapter(client=client).transcribe_document("input.wav")
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
    def test_native_document_reaches_export_and_cache_with_speakers_without_vad(self):
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
                    patch("modules.pipeline._run_vad_stage") as vad, \
                    patch("modules.llm_enhancer.enhance_asr_blocks", side_effect=translate), \
                    patch("modules.detect.create_detect_stage") as detect, redirect_stdout(StringIO()):
                for _ in range(2):
                    result = run_pipeline(str(source), "", str(output), tts_backend="subtitles-only",
                        asr_backend="moss-transcribe-diarize", asr_kwargs={"endpoint": "http://plugin"},
                        japanese_cleaner="regex-only", detect_enabled=True, llm_kwargs={"provider": "test"})
                    self.assertEqual([block.speaker for block in result["blocks"]], ["S01", "S02"])
                    self.assertIn("S01：こんにちは。", Path(result["final_vtt"]).read_text(encoding="utf-8"))
            vad.assert_not_called()
            detect.assert_not_called()
            client.request.assert_called_once()
            self.assertEqual(preserved, [True, True])
            raw = json.loads((output / "asr_subtitles/raw_response.json").read_text(encoding="utf-8"))
            self.assertEqual(raw, payload())
