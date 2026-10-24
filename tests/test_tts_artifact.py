"""Tests for modules.tts_artifact (pure-numpy TTS artifact detector).

Run from the project root:  python -m unittest tests.test_tts_artifact
Offline — synthetic WAVs are written with the stdlib `wave` module and decoded
through pydub (same as the pipeline), so ffmpeg must be reachable (the bundled
ffmpeg_bin is used by the other pipeline tests too).
"""

import os
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.tts_artifact import detect_tts_artifact, _yin_frames

SR = 24000


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-32768, min(32767, s)))) for s in samples))


def _sine(amp, dur_ms, freq):
    n = int(SR * dur_ms / 1000)
    t = np.arange(n) / SR
    return amp * 32767 * np.sin(2 * np.pi * freq * t)


def _missing_fundamental(amp, dur_ms):
    """Harmonics 400/600/800 Hz with NO 200 Hz fundamental (dark-case signal)."""
    n = int(SR * dur_ms / 1000)
    t = np.arange(n) / SR
    return amp * 32767 * (np.sin(2 * np.pi * 400 * t)
                          + 0.7 * np.sin(2 * np.pi * 600 * t)
                          + 0.5 * np.sin(2 * np.pi * 800 * t))


class DetectTest(unittest.TestCase):
    def _detect(self, samples):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.wav")
            _write_wav(p, samples)
            return detect_tts_artifact(p)

    def test_pure_sine_is_clean(self):
        r = self._detect(_sine(0.5, 1500, 300.0))
        self.assertFalse(r["flagged"])
        self.assertEqual(r["flags"], set())

    def test_missing_fundamental_is_dark(self):
        # energy at 400/600/800 but not 200 → YIN tracks 200 Hz where E(f0)≈0
        r = self._detect(_missing_fundamental(0.5, 1500))
        self.assertTrue(r["flagged"])
        self.assertIn("dark", r["flags"])
        region = r["regions"][0]
        self.assertGreater(region["weak"], 50)

    def test_white_noise_is_mechanical(self):
        rng = np.random.RandomState(0)
        r = self._detect(rng.randn(int(SR * 1.5)) * 0.3 * 32767)
        self.assertTrue(r["flagged"])
        self.assertIn("mech", r["flags"])
        self.assertGreater(r["regions"][0]["flat"], 0.05)

    def test_silence_is_clean(self):
        r = self._detect(np.zeros(int(SR * 1.5)))
        self.assertFalse(r["flagged"])

    def test_short_file_is_clean(self):
        # shorter than one 50 ms analysis window → no speech regions
        r = self._detect(_sine(0.5, 30, 300.0))
        self.assertFalse(r["flagged"])

    def test_missing_file_reports_error_not_flagged(self):
        r = detect_tts_artifact(os.path.join("nonexistent", "x.mp3"))
        self.assertFalse(r["flagged"])
        self.assertTrue(r.get("error"))


class YinFramesTest(unittest.TestCase):
    def test_matches_librosa_on_voice_like_signal(self):
        librosa = __import__("librosa")  # present in dev env; only for cross-check
        y = _sine(0.5, 800, 300.0)
        mine = _yin_frames(y / 32767.0, SR, hop_length=1200)
        lib = librosa.yin(y / 32767.0, fmin=60, fmax=500, sr=SR,
                          frame_length=2048, hop_length=1200)
        self.assertEqual(mine.shape, lib.shape)
        self.assertLess(float(np.abs(mine - lib).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
