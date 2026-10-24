"""Tests for modules.asr.fish_asr_adapter.

Run from the project root:  python -m unittest tests.test_fish_asr_adapter
Fully offline — httpx is stubbed; no network is touched.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import modules.asr.fish_asr_adapter as fish_mod
from modules.asr import ASRError, FishAudioASRAdapter
from modules.asr.models import ASRResult


class _FakeResponse:
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class _FakeACM:
    """Async context manager standing in for httpx.AsyncClient."""

    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, data=None, files=None):
        self.owner.calls.append(
            {"url": url, "headers": headers, "data": data, "files": files}
        )
        if self.owner.responses:
            return self.owner.responses.pop(0)
        return _FakeResponse(500, {}, "no response queued")


class FakeHttpx:
    """Stand-in for httpx inside modules.asr.fish_asr_adapter (isolated via patch)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.client_creates = 0

    def AsyncClient(self, **kwargs):
        self.client_creates += 1
        return _FakeACM(self)


def _write_wav(dirpath, name="slice_000.wav"):
    path = os.path.join(dirpath, name)
    with open(path, "wb") as f:
        f.write(b"\x00" * 100)
    return path


def _sample_result(segments=None):
    return {
        "text": "こんにちは。",
        "duration": 2.5,
        "language_code": "ja",
        "language": "ja",
        "segments": segments or [{"text": "こんにちは。", "start": 0.0, "end": 2.5}],
    }


class FishAdapterTest(unittest.TestCase):
    def test_is_configured(self):
        self.assertFalse(FishAudioASRAdapter().is_configured())
        self.assertTrue(FishAudioASRAdapter(api_key="k").is_configured())

    def test_timeout_property(self):
        self.assertEqual(FishAudioASRAdapter(timeout=90).timeout, 90)
        self.assertEqual(FishAudioASRAdapter().timeout, 120.0)

    def test_name(self):
        self.assertEqual(FishAudioASRAdapter(api_key="k").name, "fish-asr")

    def test_transcribe_request_format(self):
        adapter = FishAudioASRAdapter(api_key="k-test", language="ja")
        fake = FakeHttpx([_FakeResponse(200, _sample_result())])
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                result = asyncio.run(adapter.transcribe(wav))

        self.assertIsInstance(result, ASRResult)
        self.assertEqual(result.text, "こんにちは。")
        call = fake.calls[0]
        self.assertEqual(call["url"], "https://api.fish.audio/v1/asr")
        self.assertEqual(call["headers"]["Authorization"], "Bearer k-test")
        self.assertEqual(call["data"]["language"], "ja")
        # ignore_timestamps=false is what returns time slices (true → empty segments).
        self.assertEqual(call["data"]["ignore_timestamps"], "false")
        # multipart audio file field
        self.assertIn("audio", call["files"])
        name, bytes_data, mime = call["files"]["audio"]
        self.assertEqual(name, "slice_000.wav")
        self.assertEqual(mime, "audio/wav")

    def test_transcribe_with_shared_client(self):
        # Passing a shared client must reuse it (no fresh AsyncClient per call).
        adapter = FishAudioASRAdapter(api_key="k-test")
        fake = FakeHttpx([_FakeResponse(200, _sample_result())])
        client = _FakeACM(fake)
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            result = asyncio.run(adapter.transcribe(wav, client=client))
        self.assertEqual(result.text, "こんにちは。")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.client_creates, 0)

    def test_transcribe_without_key_raises(self):
        adapter = FishAudioASRAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with self.assertRaises(ASRError):
                asyncio.run(adapter.transcribe(wav))

    def test_timestamps_false_drops_segments(self):
        adapter = FishAudioASRAdapter(api_key="k")
        fake = FakeHttpx([_FakeResponse(200, _sample_result())])
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                result = asyncio.run(adapter.transcribe(wav, timestamps=False))
        self.assertEqual(result.text, "こんにちは。")
        self.assertEqual(result.segments, [])

    def test_parse_response_segments(self):
        adapter = FishAudioASRAdapter(api_key="k")
        resp = _sample_result([
            {"text": "こんにちは", "start": 0.1, "end": 1.2},
            {"text": "元気ですか", "start": 1.5, "end": 2.5},
        ])
        result = adapter._parse_response(resp)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].start, 0.1)
        self.assertEqual(result.segments[1].end, 2.5)
        self.assertEqual(result.duration, 2.5)
        self.assertEqual(result.language_code, "ja")

    def test_parse_response_error_field(self):
        adapter = FishAudioASRAdapter(api_key="k")
        with self.assertRaises(ASRError):
            adapter._parse_response({"error": {"message": "boom"}})

    def test_parse_response_non_dict(self):
        adapter = FishAudioASRAdapter(api_key="k")
        for bad in ([], "oops", None):
            with self.assertRaises(ASRError):
                adapter._parse_response(bad)

    def test_parse_response_missing_fields(self):
        adapter = FishAudioASRAdapter(api_key="k")
        for bad in ({"text": "x"}, {"duration": 1.0}):
            with self.assertRaises(ASRError):
                adapter._parse_response(bad)

    def test_parse_response_skips_bad_segments(self):
        adapter = FishAudioASRAdapter(api_key="k")
        resp = {"text": "hi", "duration": 3.0, "segments": [
            {"text": "a", "start": 0.0, "end": 1.0},
            "garbage",
            {"text": "b", "start": 2.0, "end": 3.0},
        ]}
        result = adapter._parse_response(resp)
        self.assertEqual(len(result.segments), 2)

    def test_mime_detection(self):
        self.assertEqual(FishAudioASRAdapter._mime_for("a.wav"), "audio/wav")
        self.assertEqual(FishAudioASRAdapter._mime_for("a.mp3"), "audio/mpeg")
        self.assertEqual(FishAudioASRAdapter._mime_for("a.bin"), "audio/wav")

    def test_transport_error_retried(self):
        adapter = FishAudioASRAdapter(api_key="k", max_retries=1)
        fake = FakeHttpx([
            _FakeResponse(500, {}, "boom"),
            _FakeResponse(200, _sample_result()),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                result = asyncio.run(adapter.transcribe(wav))
        self.assertEqual(result.text, "こんにちは。")
        self.assertEqual(len(fake.calls), 2)

    def test_4xx_not_retried(self):
        # 401 is deterministic (bad key) — never retried.
        adapter = FishAudioASRAdapter(api_key="k", max_retries=2)
        fake = FakeHttpx([_FakeResponse(401, {}, "bad key")])
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                with self.assertRaises(ASRError):
                    asyncio.run(adapter.transcribe(wav))
        self.assertEqual(len(fake.calls), 1)

    def test_explicit_no_speech_error_is_an_empty_slice(self):
        adapter = FishAudioASRAdapter(api_key="k", max_retries=0)
        fake = FakeHttpx([_FakeResponse(
            400,
            {"code": "NO_SPEECH_DETECTED", "message": "no speech detected"},
            "no speech detected",
        )])
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "silent.wav")
            import wave
            with wave.open(wav, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\0\0" * 8000)
            with mock.patch.object(fish_mod, "httpx", fake):
                result = asyncio.run(adapter.transcribe(wav))

        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])
        self.assertAlmostEqual(result.duration, 0.5)

    def test_429_retried(self):
        # 429 (concurrent-request limit) is transient — backed off and retried.
        adapter = FishAudioASRAdapter(api_key="k", max_retries=1)
        fake = FakeHttpx([
            _FakeResponse(429, {}, "concurrency limit"),
            _FakeResponse(200, _sample_result()),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                result = asyncio.run(adapter.transcribe(wav))
        self.assertEqual(result.text, "こんにちは。")
        self.assertEqual(len(fake.calls), 2)

    def test_429_exhausted_raises(self):
        adapter = FishAudioASRAdapter(api_key="k", max_retries=1)
        fake = FakeHttpx([_FakeResponse(429, {}, "still limited")] * 2)
        with tempfile.TemporaryDirectory() as tmp:
            wav = _write_wav(tmp)
            with mock.patch.object(fish_mod, "httpx", fake):
                with self.assertRaises(ASRError):
                    asyncio.run(adapter.transcribe(wav))
        self.assertEqual(len(fake.calls), 2)


if __name__ == "__main__":
    unittest.main()
