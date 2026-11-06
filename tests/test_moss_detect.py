import tempfile
import unittest
from dataclasses import replace
from unittest.mock import Mock

from modules.analysis_plugins import AnalysisPluginError
from modules.detect.contracts import DetectConfig, DetectStageInput
from modules.detect.contracts import DetectError
from modules.detect.factory import (create_detect_stage, detect_cache_signature,
                                    detect_pipeline_supports, validate_detect_selection)
from modules.detect.moss_config import MossDetectConfig
from modules.detect.moss_stage import MossDetectStage, native_sentences, proportional_spans
from modules.storage import read_json
from modules.vad.contracts import AsrSlice
from modules.vtt_parser import VTTBlock


class MossDetectTests(unittest.TestCase):
    def setUp(self):
        self.parent = AsrSlice("crop", 1000, 8000, 1000, 8000, ("region",))
        self.block = VTTBlock(0, 2000, 6000, "こんにちは。はい。", speaker="S01",
                              asr_slice=self.parent, source_text="こんにちは。はい。")
        self.client = Mock()
        self.client.request.return_value = {"words": [
            {"text": "こんにちは", "start_ms": 100, "end_ms": 1800},
            {"text": "はい", "start_ms": 2100, "end_ms": 3500}]}, 4000
        self.stage = MossDetectStage(MossDetectConfig(), "in.wav", client=self.client)

    def test_split_preserves_text_speakers_outer_bounds_lineage_and_overlaps(self):
        other = replace(self.block, index=1, speaker="S02", start_ms=2500, end_ms=3000, text="はい。")
        with tempfile.TemporaryDirectory() as directory:
            result = self.stage.run(DetectStageInput((self.block, other)), directory)
            raw = read_json(directory + "/detect_output/moss_detection_input.json")
            audit = read_json(directory + "/detect_output/moss_detection.json")
        self.assertEqual(raw["blocks"][0]["text"], self.block.text)
        self.assertEqual(audit["records"][0]["outcome"], "aligned")
        self.assertEqual([b.text for b in result.blocks], ["こんにちは。", "はい。", "はい。"])
        self.assertEqual([(b.start_ms, b.end_ms) for b in result.blocks], [(2000, 3800), (3800, 6000), (2500, 3000)])
        self.assertEqual([b.speaker for b in result.blocks], ["S01", "S01", "S02"])
        self.assertTrue(all(b.asr_slice == self.parent for b in result.blocks))
        self.assertEqual(self.block.text, "こんにちは。はい。")
        self.assertIsNot(result.blocks[-1], other)
        self.assertEqual(result.blocks[0].aligned_sentence.source_start_ms, 2000)
        self.client.request.assert_called_once_with("/align", "in.wav",
            {"text": self.block.text, "language": "Japanese"}, start_ms=2000, end_ms=6000)

    def test_unavailable_mismatch_or_ambiguous_alignment_retains_original(self):
        failures = [AnalysisPluginError("offline"),
            ({"words": [{"text": "違う", "start_ms": 0, "end_ms": 3000}]}, 4000),
            ({"words": [{"text": "こんにちははい", "start_ms": 0, "end_ms": 3000}]}, 4000),
            ({"words": []}, 4000),
            ({"words": []}, 5000)]
        for failure in failures:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.client.request.side_effect = failure if isinstance(failure, Exception) else None
                self.client.request.return_value = failure
                result = self.stage.run(DetectStageInput((self.block,)), directory)
                self.assertEqual(result.blocks, (self.block,))
                self.assertEqual(result.summary["fallbackSlices"], 1)

    def test_single_sentences_tags_and_skips_need_no_alignment(self):
        for text in ("はい。", "[clear throat]", "[event! loud?] はい。", "あ……", "「はい！」"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                block = replace(self.block, text=text)
                self.assertEqual(self.stage.run(DetectStageInput((block,)), directory).blocks, (block,))
        with tempfile.TemporaryDirectory() as directory:
            block = replace(self.block, skip_tts=True)
            self.assertEqual(self.stage.run(DetectStageInput((block,)), directory).blocks, (block,))
        self.client.request.assert_not_called()
        self.assertEqual(native_sentences("「はい！」 次です。"), ("「はい！」 ", "次です。"))

    def test_native_subdivision_without_client_uses_mora_and_assigns_source_block_id(self):
        stage = MossDetectStage(MossDetectConfig(), "in.wav")
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((self.block,)), directory)
        self.assertEqual([b.text for b in result.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(len(result.blocks), 2)
        self.assertTrue(result.blocks[0].aligned_sentence.source_block_id)
        self.assertEqual(result.blocks[0].aligned_sentence.source_block_id,
                         result.blocks[1].aligned_sentence.source_block_id)
        self.assertEqual(result.blocks[0].aligned_sentence.method, "moss-native")
        self.assertEqual(result.summary["alignedSlices"], 1)
        self.assertEqual(result.summary["fallbackSlices"], 0)
        self.assertEqual(result.blocks[0].start_ms, 2000)
        self.assertEqual(result.blocks[1].end_ms, 6000)
        self.assertEqual(result.blocks[0].end_ms, result.blocks[1].start_ms)

    def test_backend_factory_and_cache_own_moss_policy(self):
        config = validate_detect_selection(DetectConfig(aligner="moss-native"), "moss-transcribe-diarize")
        self.assertIsInstance(create_detect_stage(config, "in.wav"), MossDetectStage)
        signature = detect_cache_signature(config)
        self.assertEqual(signature, detect_cache_signature(replace(config, atom_silence_ms=900)))
        self.assertEqual(signature, detect_cache_signature(replace(config, qwen_endpoint="http://other:8007")))
        self.assertNotEqual(signature, detect_cache_signature(replace(config,
            moss=MossDetectConfig(min_sentence_duration_ms=400))))

    def test_moss_detect_is_selected_explicitly_and_only_for_moss(self):
        self.assertTrue(detect_pipeline_supports("moss-native", "moss-transcribe-diarize"))
        for aligner in ("joint-vad", "qwen-force-align"):
            self.assertTrue(detect_pipeline_supports(aligner, "moss-transcribe-diarize"))
            self.assertTrue(detect_pipeline_supports(aligner, "fish-asr"))
        self.assertFalse(detect_pipeline_supports("moss-native", "fish-asr"))
        self.assertIsInstance(validate_detect_selection(DetectConfig(aligner="moss-native"), "moss-transcribe-diarize"), DetectConfig)
        self.assertIsInstance(validate_detect_selection(DetectConfig(aligner="joint-vad"), "moss-transcribe-diarize"), DetectConfig)
        self.assertIsInstance(validate_detect_selection(DetectConfig(aligner="qwen-force-align"), "moss-transcribe-diarize"), DetectConfig)
        with self.assertRaisesRegex(DetectError, "DETECT_BACKEND_INCOMPATIBLE"):
            validate_detect_selection(DetectConfig(aligner="moss-native"), "common-asr")
        with self.assertRaisesRegex(DetectError, "DETECT_IMPLEMENTATION_UNKNOWN"):
            validate_detect_selection(DetectConfig(aligner="unknown"), "fish-asr")
