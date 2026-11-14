import asyncio
import math
import struct
import tempfile
import unittest
import wave
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, mock_open, patch

from modules.asr.common_asr_adapter import CommonASRAdapter
from modules.asr.qwen_asr_adapter import FunASRAdapter
from modules.asr_engine import SentenceSegment, _group_sentences_into_slices
from modules.detect import (
    AsrBoostConfig,
    AsrBoostDetectStage,
    DetectConfig,
    DetectError,
    DetectStageInput,
    asr_boost_supports,
    create_detect_stage,
    detect_cache_signature,
    detect_pipeline_supports,
    register_asr_boost_hook,
    validate_detect_selection,
)
from modules.detect.asr_boost_stage import align_sentence_spans
from modules.storage import read_json
from modules.vad import optimize_asmr_segments
from modules.vad.contracts import AsrSlice, VadRegion
from modules.vad.models import SpeechRegion
from modules.vtt_parser import VTTBlock


def _write_two_burst_wav(path: str, duration_ms: int = 4000, valley_start_ms: int = 1850, valley_end_ms: int = 2050) -> None:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_ms / 1000)
    frames = bytearray()
    for index in range(total_frames):
        time_ms = index * 1000.0 / sample_rate
        if valley_start_ms <= time_ms <= valley_end_ms:
            sample = 0
        else:
            sample = int(12000 * math.sin(2.0 * math.pi * 440.0 * (index / sample_rate)))
        frames.extend(struct.pack("<h", sample))
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))


class AsrBoostDetectTests(unittest.TestCase):
    def setUp(self):
        self.parent = AsrSlice("crop-0", 800, 4200, 1000, 4000, ("region-0",))
        self.block = VTTBlock(
            0,
            1000,
            4000,
            "こんにちは。はい。",
            speaker="S01",
            asr_slice=self.parent,
            source_text="こんにちは。はい。",
            asr_boost_spans=(
                {"start_ms": 850, "end_ms": 1820, "text": "こんにちは。"},
                {"start_ms": 2080, "end_ms": 4150, "text": "はい。"},
            ),
        )

    def test_factory_and_backend_validation_support_only_timestamp_asrs(self):
        for backend in ("fish-asr", "fun-asr", "common-asr"):
            self.assertTrue(asr_boost_supports(backend), backend)
            self.assertTrue(detect_pipeline_supports("asr-boost", backend), backend)
            validated = validate_detect_selection(DetectConfig(aligner="asr-boost"), backend)
            self.assertEqual(validated.asr_backend, backend)
            self.assertEqual(validated.asr_boost_settings().backend, backend)
            stage = create_detect_stage(validated, "in.wav")
            self.assertIsInstance(stage, AsrBoostDetectStage)

        for unsupported in ("qwen-audio-3.1-asr-flash", "mimo-asr", "moss-transcribe-diarize", "chickenrice"):
            self.assertFalse(asr_boost_supports(unsupported), unsupported)
            self.assertFalse(detect_pipeline_supports("asr-boost", unsupported), unsupported)
            with self.assertRaisesRegex(DetectError, "DETECT_BACKEND_INCOMPATIBLE"):
                validate_detect_selection(DetectConfig(aligner="asr-boost"), unsupported)

    def test_cache_signature_reflects_asr_boost_parameters(self):
        base = validate_detect_selection(DetectConfig(aligner="asr-boost"), "fish-asr")
        sig = detect_cache_signature(base)
        self.assertEqual(sig, detect_cache_signature(replace(base, qwen_endpoint="http://other:9999")))
        self.assertEqual(sig, detect_cache_signature(replace(base, atom_silence_ms=900)))
        self.assertNotEqual(sig, detect_cache_signature(replace(base, asr_boost_padding_ms=400)))
        self.assertNotEqual(
            sig,
            detect_cache_signature(
                replace(base, asr_boost=AsrBoostConfig(padding_ms=250, min_duration_ms=450, backend="fish-asr"))
            ),
        )

    def test_outer_boundaries_locked_to_vad_speech_and_middle_refined_with_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = f"{directory}/in.wav"
            # Valley is around 1900-2000ms (900-1000ms relative to wav start at 0ms).
            _write_two_burst_wav(wav_path, duration_ms=5000, valley_start_ms=1900, valley_end_ms=2000)
            stage = AsrBoostDetectStage(AsrBoostConfig(padding_ms=250, min_duration_ms=280, backend="fish-asr"), wav_path)
            result = stage.run(DetectStageInput((self.block,)), directory)
            raw = read_json(directory + "/detect_output/sentence_detection_input.json")
            audit = read_json(directory + "/detect_output/alignment_audit.json")

        self.assertEqual(raw["blocks"][0]["text"], "こんにちは。はい。")
        self.assertEqual(audit["slices"][0]["outcome"], "aligned")
        self.assertEqual(len(result.blocks), 2)
        first, second = result.blocks
        self.assertEqual([first.text, second.text], ["こんにちは。", "はい。"])
        # First start_ms strictly locked to parent.speech_start_ms (1000), despite span starting at 850.
        self.assertEqual(first.start_ms, 1000)
        # Last end_ms strictly locked to parent.speech_end_ms (4000), despite span ending at 4150.
        self.assertEqual(second.end_ms, 4000)
        # Contiguous internal boundary refined within [1820 - 250, 2080 + 250] and hitting the silent valley (1900..2000).
        self.assertEqual(first.end_ms, second.start_ms)
        self.assertGreaterEqual(first.end_ms, 1890)
        self.assertLessEqual(first.end_ms, 2010)
        self.assertEqual(first.speaker, "S01")
        self.assertEqual(second.speaker, "S01")
        self.assertEqual(first.aligned_sentence.method, "asr-boost:fish-asr")
        self.assertEqual(first.aligned_sentence.source_block_id, second.aligned_sentence.source_block_id)

    def test_word_level_spans_from_plugin_group_into_sentences_and_apply_plugin_config(self):
        plugin_block = replace(
            self.block,
            asr_boost_spans=(
                {"start_ms": 900, "end_ms": 1400, "text": "今日"},
                {"start_ms": 1400, "end_ms": 1820, "text": "は。"},
                {"start_ms": 2080, "end_ms": 3000, "text": "おやす"},
                {"start_ms": 3000, "end_ms": 4100, "text": "み。"},
            ),
            asr_boost_config={"enabled": True, "padding_ms": 180, "min_duration_ms": 300},
            text="今日は。おやすみ。",
            source_text="今日は。おやすみ。",
        )
        stage = AsrBoostDetectStage(AsrBoostConfig(backend="common-asr"), "missing.wav")
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput((plugin_block,)), directory)
            audit = read_json(directory + "/detect_output/alignment_audit.json")

        self.assertEqual([b.text for b in result.blocks], ["今日は。", "おやすみ。"])
        self.assertEqual(result.blocks[0].start_ms, 1000)
        self.assertEqual(result.blocks[1].end_ms, 4000)
        # Without a readable WAV file, midpoint of (1820, 2080) = 1950 is chosen inside ±180ms padding.
        self.assertEqual(result.blocks[0].end_ms, 1950)
        self.assertEqual(result.blocks[1].start_ms, 1950)
        self.assertEqual(audit["slices"][0]["effectiveConfig"]["padding_ms"], 180)
        self.assertEqual(audit["slices"][0]["effectiveConfig"]["min_duration_ms"], 300)

        disabled_block = replace(plugin_block, asr_boost_config={"enabled": False})
        with tempfile.TemporaryDirectory() as directory:
            disabled_result = stage.run(DetectStageInput((disabled_block,)), directory)
            disabled_audit = read_json(directory + "/detect_output/alignment_audit.json")
        self.assertEqual(len(disabled_result.blocks), 1)
        self.assertEqual(disabled_audit["slices"][0]["reason"], "plugin_disabled")

    def test_custom_registered_hook_is_supported_by_factory(self):
        register_asr_boost_hook(
            "custom-plugin-asr",
            lambda block, parent: (
                {"start_ms": parent.speech_start_ms, "end_ms": (parent.speech_start_ms + parent.speech_end_ms) // 2, "text": "こんにちは。"},
                {"start_ms": (parent.speech_start_ms + parent.speech_end_ms) // 2, "end_ms": parent.speech_end_ms, "text": "はい。"},
            ),
            lambda block, cfg: {"enabled": True, "padding_ms": 200, "min_duration_ms": 250},
        )
        self.assertTrue(asr_boost_supports("custom-plugin-asr"))
        validated = validate_detect_selection(DetectConfig(aligner="asr-boost"), "custom-plugin-asr")
        stage = create_detect_stage(validated, "missing.wav")
        with tempfile.TemporaryDirectory() as directory:
            plain_block = replace(self.block, asr_boost_spans=())
            result = stage.run(DetectStageInput((plain_block,)), directory)
        self.assertEqual([b.text for b in result.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(result.blocks[0].start_ms, 1000)
        self.assertEqual(result.blocks[1].end_ms, 4000)

    def test_local_slice_grouping_and_fun_asr_word_timestamps_feed_asr_boost(self):
        with patch("modules.asr.qwen_asr_adapter.wav_duration", return_value=3.0):
            transcript = FunASRAdapter._parse_transcript(
                {
                    "transcripts": [
                        {
                            "text": "こんにちは。はい。",
                            "sentences": [
                                {
                                    "text": "こんにちは。はい。",
                                    "begin_time": 0,
                                    "end_time": 3000,
                                    "words": [
                                        {"text": "こんにちは", "punctuation": "。", "begin_time": 100, "end_time": 1400},
                                        {"text": "はい", "punctuation": "。", "begin_time": 1600, "end_time": 2900},
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "missing.wav",
                "ja",
                True,
            )
        self.assertEqual(len(transcript.segments), 2)
        self.assertEqual(transcript.segments[0].text, "こんにちは。")
        self.assertAlmostEqual(transcript.segments[0].start, 0.1)
        self.assertAlmostEqual(transcript.segments[0].end, 1.4)

        slices = optimize_asmr_segments([SpeechRegion(1.0, 4.0)])
        sentences = [
            SentenceSegment(id=0, start=1.1, end=2.4, text="こんにちは。"),
            SentenceSegment(id=1, start=2.6, end=3.9, text="はい。"),
        ]
        blocks = _group_sentences_into_slices(sentences, slices)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(len(blocks[0].asr_boost_spans), 2)
        stage = AsrBoostDetectStage(AsrBoostConfig(backend="fun-asr"), "missing.wav")
        with tempfile.TemporaryDirectory() as directory:
            result = stage.run(DetectStageInput(tuple(blocks)), directory)
        self.assertEqual([b.text for b in result.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(result.blocks[0].start_ms, 1000)
        self.assertEqual(result.blocks[0].end_ms, 2500)
        self.assertEqual(result.blocks[1].start_ms, 2500)
        self.assertEqual(result.blocks[1].end_ms, 4000)

    def test_common_asr_adapter_preserves_plugin_spans_and_config_across_timeline_restore(self):
        slices = optimize_asmr_segments([SpeechRegion(10.0, 15.0)])
        client = AsyncMock()
        response = Mock(is_success=True)
        response.json.return_value = {
            "contract_version": 1,
            "stage": "asr",
            "language": "ja",
            "source": "parakeet-asr",
            "timeline_source": "external_segments",
            "asr_boost": {"enabled": True, "padding_ms": 220, "min_duration_ms": 300},
            "blocks": [
                {
                    "index": 0,
                    "start_ms": 9600,
                    "end_ms": 15400,
                    "text": "こんにちは。はい。",
                    "spans": [
                        {"start_ms": 9800, "end_ms": 12200, "text": "こんにちは。"},
                        {"start_ms": 12400, "end_ms": 15100, "text": "はい。"},
                    ],
                }
            ],
        }
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch("modules.asr.common_asr_adapter.httpx.AsyncClient", return_value=context), patch(
            "builtins.open", mock_open(read_data=b"audio")
        ):
            doc = asyncio.run(CommonASRAdapter().transcribe_document("audio.wav", slices))

        self.assertEqual(len(doc.blocks), 1)
        self.assertEqual((doc.blocks[0].start_ms, doc.blocks[0].end_ms), (10000, 15000))
        self.assertEqual(doc.blocks[0].asr_boost_config, {"enabled": True, "padding_ms": 220, "min_duration_ms": 300})
        stage = AsrBoostDetectStage(AsrBoostConfig(backend="common-asr"), "missing.wav")
        with tempfile.TemporaryDirectory() as directory:
            detected = stage.run(DetectStageInput(tuple(doc.blocks)), directory)
        self.assertEqual([b.text for b in detected.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(detected.blocks[0].start_ms, 10000)
        self.assertEqual(detected.blocks[0].end_ms, 12300)
        self.assertEqual(detected.blocks[1].start_ms, 12300)
        self.assertEqual(detected.blocks[1].end_ms, 15000)

    def test_asr_boost_carves_intervals_then_joint_vad_and_qwen_aligners_calibrate(self):
        # 1. ASR-boost pre-stage -> Joint-VAD aligner calibration
        joint_cfg = validate_detect_selection(DetectConfig(aligner="joint-vad", asr_boost_padding_ms=250), "common-asr")
        joint_stage = create_detect_stage(joint_cfg, "missing.wav")
        # Fine VAD atoms have a silence gap [1960, 2040] (midpoint 2000ms) inside the ±250ms window [1570, 2330].
        joint_stage.aligner.atom_detector = Mock(
            detect=Mock(
                return_value=(
                    VadRegion("atom-0", 1000, 1960),
                    VadRegion("atom-1", 2040, 4000),
                )
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            joint_res = joint_stage.run(DetectStageInput((self.block,)), directory)
            intervals = read_json(directory + "/detect_output/asr_boost_intervals.json")
            audit = read_json(directory + "/detect_output/sentence_alignment.json")

        self.assertEqual(intervals["slices"][0]["locked_start_ms"], 1000)
        self.assertEqual(intervals["slices"][0]["locked_end_ms"], 4000)
        self.assertEqual(intervals["slices"][0]["boundaries"][0]["search_start_ms"], 1570)
        self.assertEqual(intervals["slices"][0]["boundaries"][0]["search_end_ms"], 2330)
        self.assertEqual([b.text for b in joint_res.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(joint_res.blocks[0].start_ms, 1000)
        self.assertEqual(joint_res.blocks[0].end_ms, 2000)
        self.assertEqual(joint_res.blocks[1].start_ms, 2000)
        self.assertEqual(joint_res.blocks[1].end_ms, 4000)
        self.assertEqual(audit["slices"][0]["result"]["attempts"][-1]["cutAudits"][0]["calibrator"], "vad-atom")

        # 2. ASR-boost pre-stage -> Qwen Force Align calibration (clamping drifted Qwen boundary to ±padding_ms window)
        qwen_cfg = validate_detect_selection(DetectConfig(aligner="qwen-force-align", asr_boost_padding_ms=250), "common-asr")
        qwen_stage = create_detect_stage(qwen_cfg, "missing.wav")
        # Qwen returns word timestamps relative to speech_start_ms (1000ms):
        # Word 0 ends at 1000 + 400 = 1400ms (drifted before 1570ms window start);
        # Word 1 starts at 1000 + 1020 = 2020ms (inside [1570, 2330] window).
        qwen_stage.aligner.client = Mock(
            request=Mock(
                return_value=(
                    {
                        "words": [
                            {"text": "こんにちは", "start_ms": 50, "end_ms": 400},
                            {"text": "はい", "start_ms": 1020, "end_ms": 2900},
                        ]
                    },
                    3000,
                )
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            qwen_res = qwen_stage.run(DetectStageInput((self.block,)), directory)
            qwen_intervals = read_json(directory + "/detect_output/asr_boost_intervals.json")

        self.assertEqual(qwen_intervals["slices"][0]["boundaries"][0]["search_start_ms"], 1570)
        self.assertEqual([b.text for b in qwen_res.blocks], ["こんにちは。", "はい。"])
        self.assertEqual(qwen_res.blocks[0].start_ms, 1000)
        # First sentence end_ms is clamped from drifted 1400ms into the boost search window [1570, 2330] -> 1570ms.
        self.assertEqual(qwen_res.blocks[0].end_ms, 1570)
        self.assertEqual(qwen_res.blocks[1].start_ms, 2020)
        self.assertEqual(qwen_res.blocks[1].end_ms, 4000)

        # 3. Disabling asr_boost_enabled turns off the pre-stage and invalidates detect cache signatures
        disabled_joint = replace(joint_cfg, asr_boost_enabled=False)
        disabled_qwen = replace(qwen_cfg, asr_boost_enabled=False)
        self.assertIsNone(create_detect_stage(disabled_joint, "missing.wav").asr_boost)
        self.assertIsNone(create_detect_stage(disabled_qwen, "missing.wav").asr_boost)
        self.assertNotEqual(detect_cache_signature(joint_cfg), detect_cache_signature(disabled_joint))
        self.assertNotEqual(detect_cache_signature(qwen_cfg), detect_cache_signature(disabled_qwen))


if __name__ == "__main__":
    unittest.main()
