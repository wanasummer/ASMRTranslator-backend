from modules.storage import initialize_project, read_text, read_bytes, read_json, save_text, save_json, contains, address
"""Qwen policy isolation and common downstream handoff contracts."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from modules.detect import DetectConfig, QwenConfig, create_detect_stage, detect_cache_signature
from modules.detect.contracts import (
    AlignmentRequest, DetectStageInput, SentenceBoundary, SentenceDetection,
)
from modules.detect.qwen_aligner import QwenForceAlignSentenceAligner
from modules.detect.qwen_sentences import QwenSentencePartitioner
from modules.detect.aligners import create_sentence_aligner
from modules.detect.contracts import DetectError
from modules.subtitle_processing.timeline import _build_sentence_blocks
from modules.pipeline import _serialize_parsed_vtt_block
from modules.vad.contracts import AsrSlice
from modules.vtt_parser import VTTBlock


class QwenPipelineIsolationTests(unittest.TestCase):
    def test_fsmn_aligner_registry_does_not_route_qwen_through_its_pipeline(self):
        with self.assertRaises(DetectError):
            create_sentence_aligner("qwen-force-align", DetectConfig(), None)

    def test_fsmn_settings_cannot_change_qwen_policy_or_cache(self):
        baseline = DetectConfig(aligner="qwen-force-align")
        fsmn_changed = replace(
            baseline, sentence_detector="unavailable", atom_detector="unavailable",
            region_partitioner="unavailable", duration_estimator="unavailable",
            atom_silence_ms=1000, boundary_bias=100, acoustic_weight=0,
            language_weight=0, duration_weight=100, max_sentence_count=1,
            min_sentence_duration_ms=5000, max_sentence_duration_ms=60000,
            vocalization_chars="", vocalization_ratio=1,
        )
        with patch("modules.detect.factory.create_sentence_detector", side_effect=AssertionError("MeCab")), \
                patch("modules.detect.factory.create_atom_detector", side_effect=AssertionError("FSMN")), \
                patch("modules.detect.factory.create_duration_estimator", side_effect=AssertionError("mora")), \
                patch("modules.detect.factory.create_region_partitioner", side_effect=AssertionError("joint")):
            original = create_detect_stage(baseline, "audio.wav")
            changed = create_detect_stage(fsmn_changed, "audio.wav")
        self.assertEqual(original.config, changed.config)
        self.assertIsInstance(changed.config, QwenConfig)
        self.assertEqual(detect_cache_signature(baseline), detect_cache_signature(fsmn_changed))
        words = [{"text": "今日は休みです", "start_ms": 0, "end_ms": 1500},
                 {"text": "明日は仕事です", "start_ms": 2200, "end_ms": 4000}]
        self.assertEqual(original.aligner.partitioner.partition(words, "今日は休みです。明日は仕事です。"),
                         changed.aligner.partitioner.partition(words, "今日は休みです。明日は仕事です。"))

    def test_qwen_parameters_and_version_are_part_of_its_cache(self):
        config = DetectConfig(aligner="qwen-force-align")
        signature = detect_cache_signature(config)
        self.assertIn("version", signature)
        self.assertNotEqual(signature, detect_cache_signature(replace(
            config, qwen=QwenConfig(max_sentence_count=2))))
        self.assertNotEqual(signature, detect_cache_signature(replace(
            config, qwen_endpoint="http://another-service:8007")))
        fsmn = DetectConfig()
        self.assertEqual(detect_cache_signature(fsmn), detect_cache_signature(replace(
            fsmn, qwen=QwenConfig(max_sentence_count=2), qwen_endpoint="http://other:8007")))

    def test_legacy_mecab_evidence_cannot_change_qwen_alignment(self):
        parent = AsrSlice("parent", 0, 3000, 0, 3000, ("region",))
        text = "今日は休みです明日は仕事です"
        detection = SentenceDetection(text, (), (), "qwen-force-align", 1)
        words = [{"text": "今日は休みです", "start_ms": 0, "end_ms": 1000},
                 {"text": "明日は仕事です", "start_ms": 1100, "end_ms": 2300}]
        aligner = QwenForceAlignSentenceAligner(QwenConfig(), "audio.wav",
            client=SimpleNamespace(request=lambda *a, **k: ({"words": words}, 3000)))
        clean = aligner.align(AlignmentRequest(parent, parent.id, detection))
        # A legacy caller's MeCab boundary would previously force an extra cut.
        external = replace(detection, method="mecab-japanese", boundaries=(
            SentenceBoundary(0, len("今日は休みです"), 1, "question", ("terminal_punctuation",)),
        ))
        legacy = aligner.align(AlignmentRequest(parent, parent.id, external))
        self.assertEqual(clean, legacy)
        self.assertEqual(clean.outcome, "exact")

    def test_qwen_output_is_the_only_input_to_translation_and_serialization(self):
        parent = AsrSlice("parent", 900, 5100, 1000, 5000, ("region",))
        original = VTTBlock(index=7, start_ms=1000, end_ms=5000,
            text="今日は休みです。明日は仕事です。", asr_slice=parent,
            source_text="今日は休みです。明日は仕事です。", speaker="speaker-1")
        stage = create_detect_stage(DetectConfig(aligner="qwen-force-align"), "audio.wav")
        stage.aligner.client = SimpleNamespace(request=lambda *a, **k: ({"words": [
            {"text": "今日は休みです", "start_ms": 100, "end_ms": 1400},
            {"text": "明日は仕事です", "start_ms": 1700, "end_ms": 3600},
        ]}, 4000))
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((original,)), directory)
            audit = json.loads(read_text(Path(directory) / "detect_output/sentence_alignment.json"))
        self.assertIn("endpoint", audit["config"])
        self.assertNotIn("atom_detector", audit["config"])
        self.assertNotIn("duration_estimator", audit["config"])
        translated = _build_sentence_blocks(result.blocks, {0: ["今天休息。"], 1: ["明天工作。"]},
                                            preserve_source_timeline=True)
        serialized = [_serialize_parsed_vtt_block(block) for block in translated]
        self.assertEqual([(b["start_ms"], b["end_ms"]) for b in serialized], [(1100, 2400), (2700, 4600)])
        self.assertEqual([b["source_block_id"] for b in serialized], ["parent", "parent"])
        self.assertTrue(all(b.speaker == "speaker-1" for b in translated))
        self.assertEqual((original.index, original.start_ms, original.end_ms), (7, 1000, 5000))
        self.assertEqual(original.text, "今日は休みです。明日は仕事です。")

    def test_qwen_partitioner_does_not_mutate_word_timestamps(self):
        words = [{"text": "はい", "start_ms": 100, "end_ms": 800}]
        snapshot = [dict(row) for row in words]
        ends, _ = QwenSentencePartitioner(QwenConfig()).partition(words, "はい。")
        self.assertEqual(ends, [0])
        self.assertEqual(words, snapshot)

    def test_invalid_qwen_configuration_is_rejected(self):
        for kwargs in ({"max_sentence_count": 0}, {"max_sentence_duration_ms": 1},
                       {"acoustic_weight": float("nan")}, {"min_sentence_duration_ms": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QwenConfig(**kwargs)
