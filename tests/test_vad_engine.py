"""Tests for modules.vad.engine — create_vad fallback + FSMNVAD wiring + io.

Run from the project root:  python -m unittest tests.test_vad_engine
Fully offline: the native backend (``_load_native_vad``) is mocked, so these
tests stay fast and do not need the DLL, GGUF, funasr or torch.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.vad.engine import (
    FSMNVAD,
    NoOpVAD,
    VADEngine,
    _load_native_vad,
    create_vad,
)
from modules.pipeline import write_vad_output
from modules.vad.models import SpeechRegion
from modules.error import VADUnavailableError


class CreateVadTest(unittest.TestCase):
    def test_disabled_returns_noop(self):
        vad = create_vad({"enabled": False})
        self.assertIsInstance(vad, NoOpVAD)
        self.assertFalse(vad.available)
        self.assertEqual(vad.reason, "vad.enabled=false")

    def test_native_missing_returns_noop(self):
        # native component missing → FSMNVAD raises VADUnavailableError → NoOpVAD.
        with mock.patch("modules.vad.engine._load_native_vad", return_value=None):
            vad = create_vad({"enabled": True})
        self.assertIsInstance(vad, NoOpVAD)
        self.assertFalse(vad.available)
        self.assertIn("native VAD", vad.reason)

    def test_native_load_failure_returns_noop(self):
        # native backend raises while loading → NoOpVAD.
        with mock.patch("modules.vad.engine._load_native_vad",
                        side_effect=RuntimeError("bad dll")):
            vad = create_vad({"enabled": True})
        self.assertIsInstance(vad, NoOpVAD)
        self.assertFalse(vad.available)

    def test_noop_detect_is_empty(self):
        vad = NoOpVAD()
        self.assertFalse(vad.available)
        self.assertEqual(asyncio.run(vad.detect("any.wav")), [])

    def test_engine_subclasses(self):
        self.assertIsInstance(NoOpVAD(), VADEngine)
        self.assertIsInstance(FSMNVAD, type)
        self.assertTrue(hasattr(VADEngine, "detect"))

    def test_fsmnvad_unavailable_error_is_runtime_error(self):
        self.assertTrue(issubclass(VADUnavailableError, RuntimeError))

    def test_fsmnvad_requires_native(self):
        with mock.patch("modules.vad.engine._load_native_vad", return_value=None):
            with self.assertRaises(VADUnavailableError):
                FSMNVAD()


class _FakeNativeVAD:
    """Fake native backend: records run kwargs and returns scripted regions."""

    def __init__(self, regions=None):
        self.regions = regions if regions is not None else [
            SpeechRegion(0.07, 25.08), SpeechRegion(25.08, 28.99),
        ]
        self.run_calls = []

    def run(self, wav, max_seg_ms, max_end_silence_ms):
        self.run_calls.append((len(wav), max_seg_ms, max_end_silence_ms))
        return self.regions


def _make_vad(**engine_kwargs):
    fake = _FakeNativeVAD()
    with mock.patch("modules.vad.engine._load_native_vad", return_value=fake):
        return FSMNVAD(**engine_kwargs), fake


class FSMNVADTest(unittest.TestCase):
    def test_constructs_with_native(self):
        vad, fake = _make_vad(max_single_segment_time=25000,
                              max_end_silence_time=1500)
        self.assertIs(vad._native, fake)
        self.assertIsNone(vad.model)  # no funasr model object
        self.assertEqual(vad.max_single_segment_time, 25000)
        self.assertEqual(vad.max_end_silence_time, 1500)

    def test_detect_forwards_params_and_returns_regions(self):
        vad, fake = _make_vad(max_single_segment_time=25000,
                              max_end_silence_time=1500)
        with mock.patch("modules.vad.native.load_wav_16k_mono",
                        return_value=[0.0] * 16000):
            regions = asyncio.run(vad.detect("a.wav"))
        self.assertEqual(regions, fake.regions)
        self.assertEqual(len(fake.run_calls), 1)
        _, max_seg_ms, max_end_silence_ms = fake.run_calls[0]
        self.assertEqual(max_seg_ms, 25000)
        self.assertEqual(max_end_silence_ms, 1500)

    def test_detect_failure_fails_fast(self):
        vad, _ = _make_vad()
        vad._native = mock.Mock(run=mock.Mock(side_effect=RuntimeError("boom")))
        from modules.error import PipelineError
        with self.assertRaises(PipelineError):
            asyncio.run(vad.detect("a.wav"))


class CreateVadWiringTest(unittest.TestCase):
    def test_create_vad_forwards_params(self):
        fake = _FakeNativeVAD()
        with mock.patch("modules.vad.engine._load_native_vad", return_value=fake):
            vad = create_vad({
                "enabled": True,
                "max_end_silence_time": 1500,
                "max_single_segment_time": 25000,
            })
        self.assertIsInstance(vad, FSMNVAD)
        self.assertEqual(vad.max_end_silence_time, 1500)
        self.assertEqual(vad.max_single_segment_time, 25000)

    def test_create_vad_defaults(self):
        fake = _FakeNativeVAD()
        with mock.patch("modules.vad.engine._load_native_vad", return_value=fake):
            vad = create_vad({"enabled": True})
        self.assertEqual(vad.max_end_silence_time, 1500)
        self.assertEqual(vad.max_single_segment_time, 25000)


class WriteVadOutputTest(unittest.TestCase):
    def test_write_vad_output(self):
        raw = [SpeechRegion(0.3, 1.0)]
        final = [
            SpeechRegion(0.5, 3.2),
            SpeechRegion(5.0, 8.75),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            write_vad_output(tmp, raw, final, {
                "enabled": True,
                "max_single_segment_time": 25000,
                "max_end_silence_time": 1500,
            })
            with open(os.path.join(tmp, "vad_output", "final_slices.json"),
                      encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(
                data["slices"],
                [
                    {"start": 0.5, "end": 3.2, "duration": 2.7},
                    {"start": 5.0, "end": 8.75, "duration": 3.75},
                ],
            )
            with open(os.path.join(tmp, "vad_output", "config.json"),
                      encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["vad_config"]["max_end_silence_time"], 1500)


if __name__ == "__main__":
    unittest.main()
