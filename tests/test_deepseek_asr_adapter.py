"""Unit tests for DeepSeekASRAdapter and factory registration."""

import asyncio
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.asr import (
    ASRError,
    DeepSeekASRAdapter,
    create_asr,
    get_available_asr_backends,
)


def _write_test_wav(path: Path, seconds: float = 0.5, sample_rate: int = 16000) -> None:
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def _response(status: int, payload: object) -> httpx.Response:
    content = json.dumps(payload).encode("utf-8") if isinstance(payload, (dict, list)) else str(payload).encode("utf-8")
    return httpx.Response(
        status,
        content=content,
        request=httpx.Request("POST", DeepSeekASRAdapter._DEFAULT_ENDPOINT),
    )


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, files=None):
        self.calls.append({"url": url, "headers": headers, "files": files})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DeepSeekASRAdapterTests(unittest.TestCase):
    def test_factory_registers_deepseek_asr(self):
        self.assertIn("deepseek-asr", get_available_asr_backends())
        adapter = create_asr("deepseek-asr", api_key="ds-key")
        self.assertIsInstance(adapter, DeepSeekASRAdapter)
        self.assertEqual(adapter.name, "deepseek-asr")
        self.assertTrue(adapter.is_configured())
        self.assertFalse(DeepSeekASRAdapter(api_key="").is_configured())

    def test_transcribe_success_posts_file_and_emits_single_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            _write_test_wav(wav_path)
            client = _FakeAsyncClient([
                _response(200, {"request_id": "req-1", "text": " こんにちは ", "duration_ms": 1250}),
            ])
            adapter = DeepSeekASRAdapter(api_key="ds-secret", language="ja")
            result = asyncio.run(adapter.transcribe(str(wav_path), client=client))

        self.assertEqual(result.text, "こんにちは")
        self.assertAlmostEqual(result.duration, 1.25)
        self.assertEqual(result.language_code, "ja")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "こんにちは")
        self.assertEqual(result.segments[0].start, 0.0)
        self.assertAlmostEqual(result.segments[0].end, 1.25)
        self.assertEqual(client.calls[0]["headers"]["Authorization"], "Bearer ds-secret")
        self.assertIn("file", client.calls[0]["files"])

    def test_transcribe_without_timestamps_returns_empty_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            _write_test_wav(wav_path, seconds=0.5)
            client = _FakeAsyncClient([
                _response(200, {"request_id": "req-2", "text": "テスト"}),
            ])
            adapter = DeepSeekASRAdapter(api_key="ds-secret")
            result = asyncio.run(adapter.transcribe(str(wav_path), timestamps=False, client=client))

        self.assertEqual(result.text, "テスト")
        self.assertAlmostEqual(result.duration, 0.5)
        self.assertEqual(result.segments, [])

    def test_transcribe_retries_on_429_and_5xx(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            _write_test_wav(wav_path)
            client = _FakeAsyncClient([
                _response(429, {"error": "rate limit"}),
                _response(503, {"error": "unavailable"}),
                _response(200, {"text": "成功", "duration_ms": 800}),
            ])
            adapter = DeepSeekASRAdapter(api_key="ds-secret", max_retries=2)
            with patch("modules.asr.deepseek_asr_adapter.asyncio.sleep", return_value=None):
                result = asyncio.run(adapter.transcribe(str(wav_path), client=client))

        self.assertEqual(result.text, "成功")
        self.assertEqual(len(client.calls), 3)

    def test_transcribe_fails_fast_on_401_and_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "clip.wav"
            _write_test_wav(wav_path)
            with self.assertRaises(ASRError):
                asyncio.run(DeepSeekASRAdapter(api_key="").transcribe(str(wav_path)))

            client = _FakeAsyncClient([_response(401, {"error": "unauthorized"})])
            adapter = DeepSeekASRAdapter(api_key="bad-key", max_retries=2)
            with self.assertRaises(ASRError):
                asyncio.run(adapter.transcribe(str(wav_path), client=client))
            self.assertEqual(len(client.calls), 1)
