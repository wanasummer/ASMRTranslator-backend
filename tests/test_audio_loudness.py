"""Tests for modules.audio_loudness (per-segment loudness alignment).

Run from the project root:  python -m unittest discover -s tests
Zero new dependencies — synthetic WAVs are written with the stdlib `wave`
module; pydub + the bundled ffmpeg_bin handle reading/exporting.
"""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.audio_loudness import (
    _cache_fresh,
    _original_core,
    align_tts_to_original,
    compute_gain_db,
    loudness_db,
)
from pydub import AudioSegment

RATE = 24000
FREQ = 1000.0  # 24 samples/cycle at 24 kHz → exact RMS in 20 ms windows


def _sine_samples(amp, dur_ms):
    n = int(RATE * dur_ms / 1000)
    return [int(amp * 32767 * math.sin(2 * math.pi * FREQ * i / RATE)) for i in range(n)]


def _silence_samples(dur_ms):
    return [0] * int(RATE * dur_ms / 1000)


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def _read_seg(path):
    return AudioSegment.from_file(str(path))


class LoudnessDbTest(unittest.TestCase):
    def test_pure_silence_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "silence.wav")
            _write_wav(p, _silence_samples(300))
            self.assertIsNone(loudness_db(_read_seg(p)))

    def test_half_amp_sine_is_about_minus_9_db(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sine.wav")
            _write_wav(p, _sine_samples(0.5, 400))
            db = loudness_db(_read_seg(p))
            self.assertIsNotNone(db)
            self.assertAlmostEqual(db, -9.03, delta=0.5)

    def test_trailing_silence_ignored(self):
        # 200 ms sine + 500 ms silence → same loudness as the pure sine
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sine_tail.wav")
            _write_wav(p, _sine_samples(0.5, 200) + _silence_samples(500))
            db = loudness_db(_read_seg(p))
            self.assertAlmostEqual(db, -9.03, delta=0.5)

    def test_whisper_level_kept(self):
        # A very quiet ASMR whisper is NOT pure silence — must return a number
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "whisper.wav")
            _write_wav(p, _sine_samples(0.02, 300))
            db = loudness_db(_read_seg(p))
            self.assertIsNotNone(db)
            self.assertGreater(db, -50.0)


class ComputeGainDbTest(unittest.TestCase):
    def test_basic_match(self):
        self.assertAlmostEqual(compute_gain_db(-20, -10, 1.0), -10.0, places=2)

    def test_match_component_clamped(self):
        # 40 dB of normalization needed, but the clamp caps it at 20
        self.assertEqual(compute_gain_db(-20, -60, 1.0), 20.0)

    def test_user_volume_applied_in_full(self):
        # -10 + 20*log10(0.5) = -10 - 6.02 — not swallowed by the clamp
        self.assertAlmostEqual(compute_gain_db(-20, -10, 0.5), -16.02, places=2)

    def test_limiter_binds(self):
        # min(10, -0.5 - (-3)) → 2.5 dB so boosted peak stays under -0.5 dBFS
        self.assertEqual(compute_gain_db(-20, -30, 1.0, tts_peak_db=-3.0), 2.5)

    def test_zero_volume_is_silence(self):
        self.assertEqual(compute_gain_db(-20, -10, 0.0), float("-inf"))

    def test_unmeasurable_returns_zero(self):
        self.assertEqual(compute_gain_db(None, -10, 1.0), 0.0)
        self.assertEqual(compute_gain_db(-10, None, 1.0), 0.0)


class OriginalCoreTest(unittest.TestCase):
    def test_slices_unpadded_region(self):
        # 50 ms pad + 300 ms speech + 50 ms pad → core is just the speech
        samples = _silence_samples(50) + _sine_samples(0.1, 300) + _silence_samples(50)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "orig.wav")
            _write_wav(p, samples)
            orig = _read_seg(p)
            seg = {"start_ms": 0, "original_start_ms": 50, "original_end_ms": 350}
            core = _original_core(orig, seg)
            self.assertEqual(len(core), 300)
            self.assertAlmostEqual(loudness_db(core), -23.01, delta=0.5)

    def test_falls_back_when_timing_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "orig.wav")
            _write_wav(p, _sine_samples(0.1, 300))
            self.assertEqual(len(_original_core(_read_seg(p), {})), 300)


class AlignEndToEndTest(unittest.TestCase):
    def _make_seg(self, d, idx, orig_amp, tts_amp, tts_dur_ms):
        orig_samples = _silence_samples(50) + _sine_samples(orig_amp, 300) + _silence_samples(50)
        orig_path = os.path.join(d, f"orig_{idx}.wav")
        tts_path = os.path.join(d, f"tts_{idx}.wav")
        _write_wav(orig_path, orig_samples)
        _write_wav(tts_path, _sine_samples(tts_amp, tts_dur_ms))
        return {
            "index": idx,
            "segment_file": orig_path,
            "tts_file": tts_path,
            "start_ms": 0,
            "original_start_ms": 50,
            "original_end_ms": 350,
        }

    def test_matches_tts_loudness_to_original(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 0, orig_amp=0.1, tts_amp=0.5, tts_dur_ms=400)
            orig_path = seg["segment_file"]
            align_tts_to_original([seg], tts_volume=1.0)

            out = seg["tts_file"]
            self.assertIsNotNone(out)
            self.assertTrue(out.endswith("tts_loud_0000_v100.wav"))
            self.assertTrue(os.path.exists(out))

            aligned_db = loudness_db(_read_seg(out))
            orig_db = loudness_db(_original_core(_read_seg(orig_path), seg))
            self.assertAlmostEqual(aligned_db, orig_db, delta=0.7)

    def test_zero_volume_writes_silence(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 1, orig_amp=0.1, tts_amp=0.5, tts_dur_ms=400)
            align_tts_to_original([seg], tts_volume=0)
            out = seg["tts_file"]
            self.assertTrue(out.endswith("tts_loud_0001_v000.wav"))
            self.assertTrue(os.path.exists(out))
            self.assertIsNone(loudness_db(_read_seg(out)))

    def test_volume_baked_into_cache_key(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 2, orig_amp=0.1, tts_amp=0.5, tts_dur_ms=400)
            align_tts_to_original([seg], tts_volume=1.0)
            v100 = seg["tts_file"]
            align_tts_to_original([seg], tts_volume=0.5)
            v050 = seg["tts_file"]

            self.assertNotEqual(v100, v050)
            self.assertTrue(v050.endswith("tts_loud_0002_v050.wav"))
            # orig -23.01 dB + 20*log10(0.5) ≈ -29 dB
            self.assertAlmostEqual(loudness_db(_read_seg(v050)), -29.03, delta=0.7)

    def test_idempotent_cached_rerun(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 3, orig_amp=0.1, tts_amp=0.5, tts_dur_ms=400)
            align_tts_to_original([seg], tts_volume=1.0)
            out = seg["tts_file"]
            mtime = os.path.getmtime(out)
            align_tts_to_original([seg], tts_volume=1.0)
            self.assertEqual(seg["tts_file"], out)
            self.assertEqual(os.path.getmtime(out), mtime)  # cache hit → no re-export

    def test_cache_invalidated_when_original_changes(self):
        # Same TTS file, same cache path, but the ORIGINAL audio changes → the
        # aligned output must be recomputed (stale-cache regression).
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 5, orig_amp=0.1, tts_amp=0.5, tts_dur_ms=400)
            align_tts_to_original([seg], tts_volume=1.0)
            out = seg["tts_file"]
            quiet_db = loudness_db(_read_seg(out))
            self.assertAlmostEqual(quiet_db, -23.01, delta=0.7)

            # Overwrite the original at the same path with a loud signal,
            # leaving the TTS file and cache path untouched.
            loud_samples = _silence_samples(50) + _sine_samples(0.9, 300) + _silence_samples(50)
            _write_wav(seg["segment_file"], loud_samples)

            align_tts_to_original([seg], tts_volume=1.0)
            self.assertEqual(seg["tts_file"], out)  # same cache key, but must rebuild
            loud_db = loudness_db(_read_seg(out))
            self.assertAlmostEqual(loud_db, -3.93, delta=0.7)  # matches the loud original

    def test_unmeasurable_original_left_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            orig_path = os.path.join(d, "orig_silence.wav")
            tts_path = os.path.join(d, "tts.wav")
            _write_wav(orig_path, _silence_samples(400))
            _write_wav(tts_path, _sine_samples(0.5, 400))
            seg = {"index": 4, "segment_file": orig_path, "tts_file": tts_path,
                   "start_ms": 0, "original_start_ms": 0, "original_end_ms": 400}
            align_tts_to_original([seg], tts_volume=1.0)
            self.assertEqual(seg["tts_file"], tts_path)  # all-silent original → no-op

    def test_missing_files_fail_fast(self):
        with tempfile.TemporaryDirectory() as d:
            seg = {"index": 9,
                   "segment_file": os.path.join(d, "nope.wav"),
                   "tts_file": os.path.join(d, "nope2.wav")}
            with self.assertRaises(FileNotFoundError):
                align_tts_to_original([seg], tts_volume=1.0)

    def test_skip_tts_requires_only_the_original_segment(self):
        with tempfile.TemporaryDirectory() as d:
            orig_path = os.path.join(d, "orig_skip.wav")
            _write_wav(orig_path, _sine_samples(0.1, 400))
            seg = {"index": 3, "segment_file": orig_path, "skip_tts": True}

            result = align_tts_to_original([seg], tts_volume=1.0)

            self.assertIs(result[0], seg)
            self.assertNotIn("tts_file", seg)


class CacheFreshTest(unittest.TestCase):
    def test_loudness_cache_fresh_logic(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "tts.wav")
            out = os.path.join(d, "tts_loud_0000_v100.wav")
            _write_wav(src, _sine_samples(0.5, 400))
            _write_wav(out, _sine_samples(0.1, 400))

            # missing cache
            self.assertFalse(_cache_fresh(os.path.join(d, "nope.wav"), [src]))

            # too small to trust
            small = os.path.join(d, "small.wav")
            with open(small, "wb") as f:
                f.write(b"x" * 50)
            os.utime(small, (100, 200))
            os.utime(src, (100, 100))
            self.assertFalse(_cache_fresh(small, [src]))

            # older than its source → stale
            os.utime(out, (100, 50))
            os.utime(src, (100, 100))
            self.assertFalse(_cache_fresh(out, [src]))

            # newer than its source → fresh
            os.utime(out, (100, 200))
            self.assertTrue(_cache_fresh(out, [src]))


if __name__ == "__main__":
    unittest.main()
