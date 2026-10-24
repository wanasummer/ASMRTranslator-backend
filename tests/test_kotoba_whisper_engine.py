"""Model-free tests for the Kotoba-Whisper Transformers adapter."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "plugins" / "common-asr" / "Kotoba-Whisper"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless((PLUGIN_DIR / "engine.py").is_file(),
                     "Deployable Kotoba engine is tested in the separate plugins repository")
class KotobaWhisperEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        service = types.ModuleType("test_kotoba_engine_service")
        service.ApiError = type("ApiError", (Exception,), {})
        service.EngineResult = object
        service.TimelineSegment = object
        service.TranscriptBlock = object
        service.slice_wav = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"service": service}):
            cls.engine_module = load_module("test_kotoba_engine", PLUGIN_DIR / "engine.py")

    def test_run_passes_pcm_samples_instead_of_a_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            wav_path = Path(temporary) / "audio.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 1600)

            received = []

            def fake_pipe(audio, **kwargs):
                received.append((audio, kwargs))
                return {"text": "ok"}

            engine = self.engine_module.KotobaWhisperEngine.__new__(
                self.engine_module.KotobaWhisperEngine,
            )
            engine.pipe = fake_pipe
            engine.chunk_length_s = 0
            self.assertEqual(engine._run(wav_path), {"text": "ok"})

            audio, kwargs = received[0]
            self.assertEqual(audio["sampling_rate"], 16000)
            self.assertEqual(audio["raw"].shape, (1600,))
            self.assertNotIsInstance(audio, str)
            self.assertTrue(kwargs["return_timestamps"])

    def test_model_source_uses_a_complete_local_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            for name in self.engine_module.KotobaWhisperEngine._MODEL_FILES:
                (model_dir / name).write_bytes(b"test")
            with patch.dict(os.environ, {"ASR_MODEL_DIR": temporary}, clear=True):
                resolved = self.engine_module.KotobaWhisperEngine._local_model_source()
        self.assertEqual(resolved, str(model_dir.resolve()))

    def test_model_source_rejects_an_incomplete_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"ASR_MODEL_DIR": temporary}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "重新安装插件"):
                    self.engine_module.KotobaWhisperEngine._local_model_source()


if __name__ == "__main__":
    unittest.main()
