"""Model-free contract tests for the deployable common ASR services."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATHS = {
    "parakeet": ROOT / "plugins" / "common-asr" / "Parakeet" / "service.py",
    "kotoba": ROOT / "plugins" / "common-asr" / "Kotoba-Whisper" / "service.py",
}


def load_service(name: str, path: Path):
    module_name = f"test_common_asr_{name}_service"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(all(path.is_file() for path in SERVICE_PATHS.values()),
                     "Deployable ASR services are tested in the separate plugins repository")
class CommonASRContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.services = {
            name: load_service(name, path)
            for name, path in SERVICE_PATHS.items()
        }

    def test_parse_external_segments(self):
        raw = '[{"index":0,"start_ms":100,"end_ms":500},{"index":1,"start_ms":600,"end_ms":900}]'
        for name, service in self.services.items():
            with self.subTest(service=name):
                parsed = service.parse_segments(raw, duration_ms=1000)
                self.assertEqual(
                    [(item.index, item.start_ms, item.end_ms) for item in parsed],
                    [(0, 100, 500), (1, 600, 900)],
                )

    def test_rejects_non_contiguous_index_and_out_of_range_time(self):
        invalid_values = [
            '[{"index":1,"start_ms":0,"end_ms":500}]',
            '[{"index":0,"start_ms":0,"end_ms":1001}]',
        ]
        for name, service in self.services.items():
            for raw in invalid_values:
                with self.subTest(service=name, raw=raw):
                    with self.assertRaises(service.ApiError) as raised:
                        service.parse_segments(raw, duration_ms=1000)
                    self.assertEqual(raised.exception.code, "INVALID_SEGMENTS")

    def test_slice_wav_preserves_requested_boundary_duration(self):
        for name, service in self.services.items():
            with self.subTest(service=name), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source.wav"
                target = Path(temporary) / "slice.wav"
                with wave.open(str(source), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(16000)
                    output.writeframes(b"\x00\x00" * 32000)
                service.slice_wav(source, target, service.TimelineSegment(0, 250, 1250))
                with wave.open(str(target), "rb") as result:
                    self.assertEqual(result.getframerate(), 16000)
                    self.assertEqual(result.getnframes(), 16000)

    def test_downstream_block_contract(self):
        for name, service in self.services.items():
            with self.subTest(service=name):
                block = service.TranscriptBlock(0, 1200, 4600, "  text  ").to_dict()
                self.assertEqual(block, {
                    "index": 0,
                    "start_ms": 1200,
                    "end_ms": 4600,
                    "text": "text",
                    "skip_tts": False,
                })


if __name__ == "__main__":
    unittest.main()
