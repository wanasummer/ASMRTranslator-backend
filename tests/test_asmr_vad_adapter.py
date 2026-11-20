"""ASMR selection, independent splitting policies and stage-boundary wiring."""

import asyncio
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from modules.analysis_plugins import AnalysisPluginError
from modules.detect.atom_detectors import ASMRVadAtomDetector, atom_detector_for_backend, create_atom_detector
from modules.detect.contracts import DetectConfig, DetectError
from modules.detect.factory import create_detect_stage, detect_cache_signature
from modules.error import PipelineError
from modules.pipeline import _run_vad_stage, run_pipeline
from modules.vad.asmr_policy import ASMRVadPolicy
from modules.vad.contracts import AsrSlice
from modules.vad.engine import create_vad, resolve_vad_config
from modules.vad.plugin import ASMRPluginVAD
from modules.vad.stage import run_analysis_stage
from modules.vtt_parser import VTTBlock


def local_response(*args, **kwargs):
    return {"segments": [{"start_ms": 100, "end_ms": 900}]}, 1000


class ASMRVadAdapterTests(unittest.TestCase):
    def test_detect_factory_selects_asmr_adapter_for_joint_pipeline(self):
        config = DetectConfig(atom_detector=atom_detector_for_backend("asmr-vad"), vad_threshold=.3)
        with patch("modules.detect.factory.create_sentence_detector"), \
                patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("FSMN loaded")):
            stage = create_detect_stage(config, "source.wav")
        self.assertIsInstance(stage.aligner.atom_detector, ASMRVadAtomDetector)
        self.assertEqual(stage.aligner.atom_detector.engine.config["threshold"], .3)

    def test_atom_factory_supports_default_config_and_wraps_plugin_errors(self):
        detector = create_atom_detector("asmr-vad", "source.wav")
        detector.engine.client = Mock(request=Mock(side_effect=AnalysisPluginError("unavailable")))
        parent = AsrSlice("parent", 1000, 2000, 1000, 2000, ("region",))
        with self.assertRaises(DetectError) as failure:
            detector.detect(parent, 200)
        self.assertEqual(failure.exception.code, "DETECT_ALIGNMENT_FAILED")
        self.assertIsInstance(failure.exception.__cause__, AnalysisPluginError)

    def test_detect_cache_includes_effective_asmr_policy_and_version(self):
        config = DetectConfig(atom_detector="asmr-vad", vad_threshold=.3)
        signature = detect_cache_signature(config)
        self.assertEqual(signature["vad_adapter"]["max_end_silence_time"], 200)
        self.assertEqual(signature["vad_adapter"]["threshold"], .3)
        with patch("modules.vad.plugin.ASMR_VAD_POLICY_VERSION", 999):
            self.assertNotEqual(signature, detect_cache_signature(config))
        self.assertNotIn("vad_adapter", detect_cache_signature(DetectConfig()))

    def test_factory_selects_local_plugin_and_preserves_fsmn_defaults(self):
        with patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("FSMN loaded")):
            engine = create_vad({"backend": "asmr-vad", "threshold": .25})
        self.assertIsInstance(engine, ASMRPluginVAD)
        self.assertEqual(atom_detector_for_backend("asmr-vad"), "asmr-vad")
        self.assertEqual(atom_detector_for_backend("fsmn-vad"), "native-vad")
        self.assertEqual(resolve_vad_config({"backend": "fsmn-vad"}), {"backend": "fsmn-vad"})

    def test_initial_detection_uses_adapter_policy_and_selected_threshold(self):
        client = Mock(request=Mock(side_effect=local_response))
        engine = ASMRPluginVAD({"threshold": .25, "max_end_silence_time": 1500,
                                "max_single_segment_time": 25000}, client=client)
        regions = asyncio.run(engine.detect("source.wav"))
        self.assertEqual([(r.start, r.end) for r in regions], [(.1, .9)])
        self.assertEqual(client.request.call_args.args[2], {
            "threshold": .25, "min_silence_ms": 600, "max_segment_ms": 20000, "min_speech_ms": 100,
        })

    def test_custom_policy_is_independent_and_input_is_not_mutated(self):
        config = {"threshold": .35, "max_end_silence_time": 1500, "max_single_segment_time": 25000}
        snapshot = dict(config)
        client = Mock(request=Mock(side_effect=local_response))
        policy = replace(ASMRVadPolicy(), initial_silence_ms=800, initial_max_segment_ms=15000,
                         min_speech_ms=80, max_merge_gap_sec=.4)
        engine = ASMRPluginVAD(config, client=client, policy=policy)
        engine.detect_sync("source.wav")
        self.assertEqual(config, snapshot)
        self.assertEqual(client.request.call_args.args[2]["min_silence_ms"], 800)
        self.assertEqual(client.request.call_args.args[2]["max_segment_ms"], 15000)
        self.assertEqual(client.request.call_args.args[2]["min_speech_ms"], 80)
        self.assertEqual(engine.config["max_merge_gap_sec"], .4)

    def test_effective_slice_settings_are_idempotent_and_do_not_restore_long_merges(self):
        original = {"backend": "asmr-vad", "max_end_silence_time": 1500,
                    "max_single_segment_time": 25000, "max_sec": 25, "target_max_sec": 25,
                    "max_merge_gap_sec": 2, "split_window_sec": 20, "split_overlap_sec": 4}
        effective = resolve_vad_config(original)
        self.assertEqual(resolve_vad_config(effective), effective)
        self.assertEqual((effective["max_sec"], effective["target_max_sec"], effective["max_merge_gap_sec"]),
                         (20, 20, .6))
        self.assertEqual(original["max_merge_gap_sec"], 2)
        self.assertIn("asmr_policy_version", effective)

    def test_detect_uses_same_local_backend_with_its_own_fine_policy(self):
        client = Mock(request=Mock(side_effect=local_response))
        config = DetectConfig(atom_detector="asmr-vad", vad_threshold=.3, atom_silence_ms=1500)
        with patch("modules.vad.plugin.LocalVADClient", return_value=client), \
                patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("FSMN loaded")):
            detector = ASMRVadAtomDetector("source.wav", config)
            parent = AsrSlice("parent", 1000, 2000, 1000, 2000, ("region",))
            regions = detector.detect(parent, config.atom_silence_ms)
        self.assertEqual(client.request.call_args.args[2], {
            "threshold": .3, "min_silence_ms": 200, "max_segment_ms": 300000, "min_speech_ms": 100,
        })
        self.assertEqual(client.request.call_args.kwargs, {"start_ms": 1000, "end_ms": 2000})
        self.assertEqual((regions[0].speech_start_ms, regions[0].speech_end_ms, regions[0].threshold_ms),
                         (1100, 1900, 200))
        self.assertEqual(regions[0].source, "detect-stage")

    def test_custom_fine_policy_is_owned_by_asmr_adapter(self):
        client = Mock(request=Mock(side_effect=local_response))
        detector = ASMRVadAtomDetector("source.wav", DetectConfig(vad_threshold=.4), client=client,
            policy=replace(ASMRVadPolicy(), detect_silence_ms=300, min_speech_ms=80))
        parent = AsrSlice("parent", 1000, 2000, 1000, 2000, ("region",))
        regions = detector.detect(parent, 999)
        self.assertEqual(client.request.call_args.args[2]["min_silence_ms"], 300)
        self.assertEqual(client.request.call_args.args[2]["min_speech_ms"], 80)
        self.assertEqual(regions[0].threshold_ms, 300)

    def test_both_vad_stage_paths_write_effective_policy_and_keep_lineage(self):
        for analyzed in (False, True):
            with self.subTest(analyzed=analyzed), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                client = Mock(request=Mock(side_effect=local_response))
                stack.enter_context(patch("modules.vad.plugin.LocalVADClient", return_value=client))
                stack.enter_context(patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("FSMN")))
                stack.enter_context(patch("modules.pipeline.get_audio_duration", return_value=1))
                stack.enter_context(patch("modules.vad.stage.get_audio_duration", return_value=1))
                config = {"backend": "asmr-vad", "threshold": .25}
                if analyzed:
                    slices, analysis = run_analysis_stage("source.wav", config, directory)
                    self.assertEqual(analysis.initial_regions[0].threshold_ms, 600)
                    self.assertEqual(slices[0].asr_slice, analysis.slices[0])
                else:
                    slices = _run_vad_stage("source.wav", config, directory)
                self.assertTrue(slices)
                self.assertEqual(client.request.call_args.args[2]["min_silence_ms"], 600)

    def test_main_pipeline_passes_selected_backend_to_both_stages(self):
        stage = Mock()
        stage.run.side_effect = RuntimeError("stop after detection handoff")
        block = VTTBlock(0, 100, 900, "今日は休みです。")
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            factory = stack.enter_context(patch("modules.detect.create_detect_stage", return_value=stage))
            analysis = stack.enter_context(patch("modules.vad.stage.run_analysis_stage",
                return_value=([SimpleNamespace(start=.1, end=.9)], None)))
            stack.enter_context(patch("modules.asr.create_asr", return_value=object()))
            stack.enter_context(patch("modules.asr_engine.transcribe_slices", return_value=[block]))
            stack.enter_context(patch("modules.pipeline._load_cached_asr", return_value=None))
            with redirect_stdout(StringIO()), self.assertRaises(PipelineError):
                run_pipeline("source.wav", "", directory, vad_backend="asmr-vad", vad_threshold=.25,
                             asr_kwargs={"language": "ja"}, detect_enabled=True)
        self.assertEqual(factory.call_args.args[0].atom_detector, "asmr-vad")
        self.assertEqual(factory.call_args.args[0].vad_threshold, .25)
        effective = analysis.call_args.args[1]
        self.assertEqual((effective["backend"], effective["threshold"], effective["max_end_silence_time"]),
                         ("asmr-vad", .25, 600))
        self.assertEqual(stage.run.call_args.args[0].asr_blocks, (block,))

    def test_plugin_failure_is_visible_and_does_not_fall_back_to_fsmn(self):
        client = Mock(request=Mock(side_effect=AnalysisPluginError("worker unavailable")))
        with patch("modules.vad.engine._load_native_vad", side_effect=AssertionError("FSMN")):
            with self.assertRaises(PipelineError):
                asyncio.run(ASMRPluginVAD({}, client=client).detect("source.wav"))

    def test_invalid_threshold_and_split_config_are_rejected(self):
        for config in ({"threshold": float("nan")}, {"threshold": True},
                       {"max_end_silence_time": 0}, {"max_single_segment_time": False},
                       {"split_window_sec": 0}, {"max_merge_gap_sec": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ASMRPluginVAD(config, client=Mock())
