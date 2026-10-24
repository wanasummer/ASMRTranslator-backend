"""Tests for pipeline._run_vad_stage — the no-subtitle VAD→ASR orchestration.

Run from the project root:  python -m unittest tests.test_pipeline_vad
Fully offline — create_vad / optimize_asmr_segments / write_vad_output are
mocked; no funasr, ffmpeg or network is touched.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.asr import ASRError
from modules.pipeline import _run_vad_stage
from modules.vad import SpeechRegion
from modules.vad.stage import run_analysis_stage

# The vad config as assembled by run_pipeline from its vad_* parameters.
_VAD_CFG = {
    "enabled": True,
    "max_end_silence_time": 1500,
    "max_single_segment_time": 25000,
    "min_sec": 0.3,
    "target_min_sec": 5.0,
    "target_max_sec": 25.0,
    "max_sec": 25.0,
    "max_merge_gap_sec": 2.0,
    "split_window_sec": 20.0,
    "split_overlap_sec": 1.0,
}


class _FakeVAD:
    """Scripted VAD engine: an ``available`` flag plus fixed raw regions."""

    def __init__(self, regions=None, available=True):
        self._regions = regions or []
        self.available = available

    async def detect(self, audio_path):
        return self._regions


def _run(regions=None, available=True, optimize_result=None, vad_cfg=None):
    """Call _run_vad_stage with the VAD helpers mocked.

    ``create_vad`` / ``optimize_asmr_segments`` are lazily imported by
    ``_run_vad_stage`` from ``modules.vad``, so they must be patched there.
    """
    raw = regions or []
    slices = optimize_result if optimize_result is not None else raw
    cfg = vad_cfg if vad_cfg is not None else _VAD_CFG
    tmp = tempfile.TemporaryDirectory()
    with mock.patch("modules.vad.create_vad",
                    return_value=_FakeVAD(raw, available=available)), \
         mock.patch("modules.vad.optimize_asmr_segments",
                    return_value=slices) as opt, \
         mock.patch("modules.pipeline.get_audio_duration", return_value=30.0), \
         mock.patch("modules.pipeline.write_vad_output") as wvo, \
         mock.patch("modules.pipeline.print_progress", return_value=None):
        out = _run_vad_stage("audio.wav", cfg, tmp.name)
    tmp.cleanup()
    return out, tmp.name, raw, slices, opt, wvo


class RunVadStageTest(unittest.TestCase):
    def test_unavailable_aborts(self):
        # VAD unavailable/disabled → the no-subtitle path errors out (报错终止).
        with mock.patch("modules.vad.create_vad",
                        return_value=_FakeVAD(available=False)):
            with self.assertRaises(ASRError):
                _run_vad_stage("audio.wav", _VAD_CFG, "out")

    def test_returns_optimized_slices(self):
        raw = [SpeechRegion(0.0, 6.0), SpeechRegion(10.0, 15.0)]
        slices = [SpeechRegion(0.0, 6.0), SpeechRegion(10.0, 15.0)]
        out, _, raw_got, slices_got, opt, wvo = _run(raw, optimize_result=slices)
        self.assertEqual(out, slices)
        self.assertEqual(raw_got, raw)
        self.assertEqual(slices_got, slices)
        # optimize got exactly the detected raw regions.
        self.assertEqual(opt.call_args.args[0], raw)
        self.assertEqual(opt.call_args.kwargs["audio_duration_sec"], 30.0)

    def test_write_vad_output_called_with_artifacts(self):
        raw = [SpeechRegion(0.0, 6.0)]
        slices = [SpeechRegion(0.0, 6.0)]
        _, out_dir, raw_got, slices_got, _, wvo = _run(raw, optimize_result=slices)
        self.assertEqual(wvo.call_count, 1)
        out_arg, raw_arg, slices_arg, cfg = wvo.call_args.args
        self.assertEqual(out_arg, out_dir)
        self.assertEqual(raw_arg, raw_got)
        self.assertEqual(slices_arg, slices_got)
        self.assertIs(cfg["enabled"], True)

    def test_empty_regions_writes_artifacts_and_returns_empty(self):
        # All-silence audio → no slices, but the vad_output artifacts are still
        # written (empty counts) so a failed run can be inspected.
        out, _, raw, slices, _, wvo = _run(regions=[])
        self.assertEqual(out, [])
        self.assertEqual(raw, [])
        self.assertEqual(slices, [])
        self.assertEqual(wvo.call_count, 1)
        _, raw_arg, slices_arg, cfg = wvo.call_args.args
        self.assertEqual(raw_arg, [])
        self.assertEqual(slices_arg, [])


class RunVadAnalysisStageTest(unittest.TestCase):
    def test_audio_duration_clamps_tail_padding_before_asr(self):
        raw = [SpeechRegion(9.0, 9.9)]
        with mock.patch("modules.vad.stage.create_vad",
                        return_value=_FakeVAD(raw)), \
             mock.patch("modules.vad.stage.get_audio_duration", return_value=10.0), \
             mock.patch("modules.vad.stage.write_initial_regions"), \
             mock.patch("modules.vad.stage.write_analysis"):
            slices, analysis = run_analysis_stage("audio.wav", {}, "out")

        self.assertEqual((slices[0].start, slices[0].end), (8.6, 10.0))
        self.assertEqual(analysis.slices[0].extraction_end_ms, 10000)

    def test_vad_speech_past_audio_end_is_clipped_with_its_lineage(self):
        raw = [SpeechRegion(9.0, 10.5)]
        with mock.patch("modules.vad.stage.create_vad",
                        return_value=_FakeVAD(raw)), \
             mock.patch("modules.vad.stage.get_audio_duration", return_value=10.0), \
             mock.patch("modules.vad.stage.write_initial_regions"), \
             mock.patch("modules.vad.stage.write_analysis"):
            slices, analysis = run_analysis_stage("audio.wav", {}, "out")

        contract = analysis.slices[0]
        self.assertEqual((slices[0].start, slices[0].end), (8.6, 10.0))
        self.assertEqual(contract.extraction_end_ms, 10000)
        self.assertEqual(contract.speech_end_ms, 10000)
        self.assertLessEqual(contract.speech_end_ms, contract.extraction_end_ms)
        self.assertEqual(analysis.regions[0].speech_end_ms, 10000)
        self.assertEqual(analysis.regions[0].parent_region_id, "initial-9000-10500")


if __name__ == "__main__":
    unittest.main()
