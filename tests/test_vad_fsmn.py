"""Real FSMN-VAD (funasr) inference tests — skipped when funasr is missing.

Run from the project root:  python -m unittest tests.test_vad_fsmn
Loads the actual funasr AutoModel once (class-level) and runs inference on
generated WAVs: pure silence must produce zero speech regions, and the engine
must be reported as live via ``create_vad``.
"""

import asyncio
import os
import sys
import tempfile
import unittest
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.vad.engine import FSMNVAD, create_vad

_SR = 16000


def _fsmn_available() -> bool:
    try:
        import funasr  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def _write_wav(path, samples_int16, sr=_SR):
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.asarray(samples_int16, dtype=np.int16).tobytes())


@unittest.skipUnless(_fsmn_available(), "funasr (fsmn-vad) not available")
class FSMNRealInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = cls._tmp.name
        cls.vad = create_vad({"enabled": True})
        assert isinstance(cls.vad, FSMNVAD)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_create_vad_returns_live_fsmn(self):
        self.assertIsInstance(self.vad, FSMNVAD)
        self.assertTrue(self.vad.available)

    def test_silence_produces_no_regions(self):
        path = os.path.join(self.dir, "silence.wav")
        _write_wav(path, np.zeros(_SR * 3, dtype=np.int16))
        regions = asyncio.run(self.vad.detect(path))
        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
