from modules.storage import initialize_project, read_text, read_bytes, read_json, save_text, save_json, contains, address
"""External analysis adapters preserve transcript identity and timeline lineage."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from modules.analysis_plugins import AnalysisPluginClient, AnalysisPluginError, validate_endpoint, validate_timeline
from modules.detect import create_detect_stage
from modules.detect.contracts import AlignmentRequest, DetectConfig, DetectStageInput, JapaneseToken, SentenceBoundary, SentenceDetection
from modules.detect.qwen_aligner import QwenForceAlignDetectStage, QwenForceAlignSentenceAligner
from modules.detect.sentence_detectors import ConservativeJapaneseTokenizer, MecabJapaneseSentenceDetector
from modules.detect.stage import DefaultDetectStage
from modules.pipeline import _serialize_parsed_vtt_block
from modules.vad.contracts import AsrSlice
from modules.vad.engine import create_vad
from modules.vtt_parser import VTTBlock
from modules import preflight


class AnalysisAdapters(unittest.TestCase):
    def test_http_rejection_retains_actionable_service_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"fixture")
            @contextmanager
            def normalized(*args):
                yield audio, 1000
            client = AnalysisPluginClient("http://plugin", "moss-transcribe-diarize", "moss-transcribe-diarize/1",
                client_factory=lambda **_: httpx.Client(transport=httpx.MockTransport(
                    lambda _: httpx.Response(422, json={"detail": "Audio exceeds 30 minutes; split the recording"}))))
            with patch("modules.analysis_plugins.normalized_audio", normalized):
                with self.assertRaisesRegex(AnalysisPluginError, "split the recording") as error:
                    client.request("/transcribe", "source.wav", {})
            self.assertEqual(error.exception.code, "ANALYSIS_HTTP_422")

    def test_transport_checks_request_identity_and_audio_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "upload.wav"
            audio.write_bytes(b"normalized fixture")
            @contextmanager
            def normalized(*args):
                yield audio, 1000
            invalid = False
            def response(request):
                payload = request.read().decode()
                request_id = re.search('name="request_id"\r\n\r\n([^\r]+)', payload).group(1)
                return httpx.Response(200, json={"service": "asmr-vad", "api_version": "asmr-vad/1",
                    "request_id": "wrong" if invalid else request_id, "duration_ms": 1000, "segments": []})
            client = AnalysisPluginClient("http://plugin", "asmr-vad", "asmr-vad/1",
                client_factory=lambda **_: httpx.Client(transport=httpx.MockTransport(response)))
            with patch("modules.analysis_plugins.normalized_audio", normalized):
                self.assertEqual(client.request("/detect", "source.mp3", {})[1], 1000)
                invalid = True
                with self.assertRaisesRegex(AnalysisPluginError, "identity"):
                    client.request("/detect", "source.mp3", {})

    def test_timelines_reject_bool_overlap_and_out_of_range(self):
        for rows in ([{"start_ms": False, "end_ms": 1}], [{"start_ms": 0, "end_ms": 0}],
                     [{"start_ms": 0, "end_ms": 1001}], [{"start_ms": 0, "end_ms": 700}, {"start_ms": 600, "end_ms": 900}]):
            with self.assertRaises(AnalysisPluginError):
                validate_timeline({"segments": rows}, "segments", 1000)
        self.assertEqual(validate_timeline({"segments": []}, "segments", 1000), [])

    def test_bind_address_uses_loopback_and_invalid_endpoints_are_rejected(self):
        self.assertEqual(validate_endpoint("http://0.0.0.0:8006"), "http://127.0.0.1:8006")
        for endpoint in ("file:///tmp/service", "http://user:pass@host", "http://host:bad", "http://host?key=secret"):
            with self.assertRaises(ValueError):
                validate_endpoint(endpoint)

    def test_vad_selection_does_not_load_native_component(self):
        with patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("native loaded")):
            engine = create_vad({"backend": "asmr-vad", "endpoint": "http://plugin"})
        engine.client = SimpleNamespace(request=lambda *args, **kwargs: ({"segments": [
            {"start_ms": 120, "end_ms": 900}]}, 1000))
        result = engine.detect_sync("source.wav")
        self.assertEqual((result[0].start, result[0].end), (0.12, 0.9))

    def detection(self):
        text = "今日は休みです。明日は仕事です。"
        parent = AsrSlice("slice-1", 900, 5100, 1000, 5000, ("region-1", "region-2"))
        detection = MecabJapaneseSentenceDetector(ConservativeJapaneseTokenizer()).detect(
            SimpleNamespace(source_text=text))
        return parent, detection

    def test_qwen_sentences_reach_next_stage_with_absolute_times_and_punctuation(self):
        parent, detection = self.detection()
        calls = []
        def request(*args, **kwargs):
            calls.append(kwargs)
            return {"words": [{"text": "今日は休みです", "start_ms": 100, "end_ms": 1400},
                               {"text": "明日は仕事です", "start_ms": 1700, "end_ms": 3600}]}, 4000
        config = DetectConfig(aligner="qwen-force-align")
        aligner = QwenForceAlignSentenceAligner(config, SimpleNamespace(audio_path="source.wav"),
                                               client=SimpleNamespace(request=request))
        stage = QwenForceAlignDetectStage(config, "source.wav", aligner=aligner)
        source = VTTBlock(index=0, start_ms=1000, end_ms=5000, text=detection.normalized_text,
                          source_text=detection.normalized_text, asr_slice=parent)
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((source,)), directory)
            audit = json.loads(read_text(Path(directory) / "detect_output/sentence_alignment.json"))
        self.assertEqual([row.text for row in result.blocks], ["今日は休みです。", "明日は仕事です。"])
        self.assertEqual([(row.start_ms, row.end_ms) for row in result.blocks], [(1100, 2400), (2700, 4600)])
        self.assertEqual(
            [(row.aligned_sentence.source_block_id, row.aligned_sentence.source_start_ms, row.aligned_sentence.source_end_ms)
             for row in result.blocks],
            [("slice-1", 1000, 5000), ("slice-1", 1000, 5000)],
        )
        self.assertEqual(source.start_ms, 1000)
        self.assertTrue(all(row.asr_slice.region_ids == parent.region_ids for row in result.blocks))
        self.assertEqual(audit["summary"]["alignedSlices"], 1)
        self.assertEqual(calls[0], {"start_ms": 1000, "end_ms": 5000})
        from modules.subtitle_processing.timeline import _build_sentence_blocks
        translated = _build_sentence_blocks(result.blocks, {
            0: ["今天休息。", "不需要上班。"], 1: ["明天要工作。"],
        }, preserve_source_timeline=True)
        self.assertEqual([(row.start_ms, row.end_ms, row.text) for row in translated], [
            (1100, 2400, "今天休息。不需要上班。"), (2700, 4600, "明天要工作。"),
        ])
        self.assertEqual([row.source_text for row in translated], [row.text for row in result.blocks])
        serialized = [_serialize_parsed_vtt_block(row) for row in translated]
        self.assertEqual(
            [(row["source_block_id"], row["source_start_ms"], row["source_end_ms"]) for row in serialized],
            [("slice-1", 1000, 5000), ("slice-1", 1000, 5000)],
        )

    def test_qwen_standalone_stage_bypasses_mecab_and_vocalization_filter(self):
        with patch("modules.detect.factory.create_sentence_detector", side_effect=AssertionError("MeCab loaded")), \
             patch("modules.detect.factory.create_atom_detector", side_effect=AssertionError("VAD loaded")):
            stage = create_detect_stage(DetectConfig(aligner="qwen-force-align"), "source.wav")
        self.assertIsInstance(stage, QwenForceAlignDetectStage)
        parent = AsrSlice("slice-breath", 0, 8000, 0, 8000, ("region-1",))
        text = "はぁ...はぁ...お姉さんにいっぱい甘えましょうね。気持ちよくなりましょうね。"
        words = [
            {"text": "はぁ", "start_ms": 100, "end_ms": 300},
            {"text": "はぁ", "start_ms": 350, "end_ms": 550},
            {"text": "お姉さんにいっぱい甘えましょうね", "start_ms": 700, "end_ms": 3200},
            {"text": "気持ちよくなりましょうね", "start_ms": 3800, "end_ms": 6800},
        ]
        stage.aligner.client = SimpleNamespace(request=lambda *args, **kwargs: ({"words": words}, 8000))
        source = VTTBlock(index=0, start_ms=0, end_ms=8000, text=text, source_text=text, asr_slice=parent)
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((source,)), directory)
        self.assertEqual(result.summary["bypassedSlices"], 0)
        self.assertEqual(result.summary["alignedSlices"], 1)
        self.assertEqual(
            [b.text for b in result.blocks],
            ["はぁ...はぁ...お姉さんにいっぱい甘えましょうね。", "気持ちよくなりましょうね。"],
        )
        self.assertEqual([(b.start_ms, b.end_ms) for b in result.blocks], [(100, 3200), (3800, 6800)])

    def test_qwen_short_single_word_expands_to_min_duration_without_failing(self):
        parent = AsrSlice("slice-short", 128900, 130460, 128900, 130460, ("region-1",))
        detection = SentenceDetection("うん.", (), (), "qwen-force-align", 1.0)
        aligner = QwenForceAlignSentenceAligner(
            DetectConfig(aligner="qwen-force-align"),
            SimpleNamespace(audio_path="source.wav"),
            client=SimpleNamespace(request=lambda *args, **kwargs: (
                {"words": [{"text": "うん", "start_ms": 480, "end_ms": 560}]}, 1560
            )),
        )
        result = aligner.align(AlignmentRequest(parent, parent.id, detection))
        self.assertEqual(result.outcome, "exact")
        self.assertGreaterEqual(result.blocks[0].end_ms - result.blocks[0].start_ms, 280)
        self.assertGreaterEqual(result.blocks[0].start_ms, 128900)
        self.assertLessEqual(result.blocks[0].end_ms, 130460)

    def test_qwen_splits_unpunctuated_speech_across_acoustic_pause_and_skips_short_cut(self):
        parent = AsrSlice("slice-2", 900, 6100, 1000, 6000, ("region-1",))
        text = "あ今日は休みです明日は仕事です"
        tokens = (
            JapaneseToken("あ", 0, 1),
            JapaneseToken("今日は休みです", 1, 8),
            JapaneseToken("明日は仕事です", 8, 15),
        )
        boundaries = (
            SentenceBoundary(1, 1, 0.85, "interjection", ("terminal_punctuation",)),
            SentenceBoundary(2, 8, 0.42, "statement", ("new_clause_starter",)),
            SentenceBoundary(3, 15, 0.90, "statement", ("sentence_end",)),
        )
        detection = SentenceDetection(text, tokens, boundaries, "mecab-japanese", 0.75)
        words = [
            {"text": "あ", "start_ms": 0, "end_ms": 120},
            {"text": "今日は休みです", "start_ms": 140, "end_ms": 1800},
            {"text": "明日は仕事です", "start_ms": 2550, "end_ms": 4500},
        ]
        aligner = QwenForceAlignSentenceAligner(
            DetectConfig(),
            SimpleNamespace(audio_path="source.wav"),
            client=SimpleNamespace(request=lambda *args, **kwargs: ({"words": words}, 5000)),
        )
        result = aligner.align(AlignmentRequest(parent, parent.id, detection))
        self.assertEqual(result.outcome, "aligned")
        self.assertEqual([b.text for b in result.blocks], ["あ今日は休みです", "明日は仕事です"])
        self.assertEqual([(b.start_ms, b.end_ms) for b in result.blocks], [(1000, 2800), (3550, 5500)])

    def test_qwen_failure_retains_parent_and_reports_fallback(self):
        parent, detection = self.detection()
        for words in ([], [{"text": "違う", "start_ms": 1, "end_ms": 500}],
                      [{"text": "今日は休みです明日は仕事です", "start_ms": 10, "end_ms": 10}]):
            aligner = QwenForceAlignSentenceAligner(DetectConfig(), SimpleNamespace(audio_path="source.wav"),
                client=SimpleNamespace(request=lambda *args, **kwargs: ({"words": words}, 4000)))
            result = aligner.align(AlignmentRequest(parent, parent.id, detection))
            self.assertEqual(result.outcome, "fallback")
            self.assertEqual((result.blocks[0].start_ms, result.blocks[0].end_ms), (1000, 5000))
            self.assertEqual(result.blocks[0].text, detection.normalized_text)
            self.assertTrue(result.attempts)


class Preflight(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_health_requires_correct_identity_and_ready_model(self):
        for service, pid, api in (("alignment", "qwen-force-align", "qwen-force-align/1"),):
            for identity, ready, expected in ((pid, True, True), ("wrong", True, False), (pid, False, False)):
                def factory(**kwargs):
                    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200,
                        json={"service": identity, "api_version": api, "status": "ok", "ready": ready})))
                with patch("modules.preflight.async_client", factory):
                    result = await preflight.run_preflight_check(service, {})
                self.assertEqual(result.success, expected, result.message)


if __name__ == "__main__":
    unittest.main()
