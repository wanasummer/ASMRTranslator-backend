import asyncio
import os
import tempfile
import unittest
import wave
from unittest import mock

import httpx

from modules.tts.factory import create_tts, get_available_backends
from modules.tts.indextts_adapter import IndexTTSAdapter, _upload_cache
from modules.error import TTSError
from modules.tts.models import TTSSegmentContext
from modules.tts_engine import generate_tts


class _Response:
    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text

    def json(self):
        return self._payload


class _Client:
    calls = []

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/upload"):
            return _Response(payload=["remote/ref.wav"])
        if "/call/" in url:
            return _Response(status_code=404)
        return _Response(payload={"data": [{"url": "/generated.wav"}]})

    def get(self, url, **kwargs):
        self.calls.append(("get", url, {}))
        return _Response(content=b"ID3" + b"x" * 300)


class IndexTTSAdapterTest(unittest.TestCase):
    def test_all_generation_controls_have_official_payload_positions(self):
        adapter = IndexTTSAdapter(emotion_weight=0.4, options={
            "emotion_mode": "vector", "emotion_vector": [0.1] * 8,
            "emotion_random": True, "language": "JA", "duration_factor": 1.2,
            "do_sample": True, "top_p": 0.9, "top_k": 12, "temperature": 0.7,
            "length_penalty": -0.5, "num_beams": 4, "repetition_penalty": 8,
            "max_mel_tokens": 2000, "max_text_tokens_per_segment": 100,
        })
        data = adapter._request_data("你好", {"path": "voice"}, None, "+0%")
        self.assertEqual(data[:6], ["使用情感向量控制", {"path": "voice"}, "你好", "JA", None, 0.4])
        self.assertEqual(data[6:14], [0.1] * 8)
        self.assertEqual(data[14:], ["", True, 100, 1.2, True, 0.9, 12, 0.7, -0.5, 4, 8, 2000])

    def test_inactive_controls_are_neutralized(self):
        adapter = IndexTTSAdapter(emotion_weight=0.2, options={
            "emotion_mode": "same", "emotion_text": "怒", "emotion_random": True,
            "emotion_vector": [1] * 8, "do_sample": False, "num_beams": 1,
            "length_penalty": 2,
        })
        data = adapter._request_data("文本", {}, None, "+0%")
        self.assertEqual(data[5:16], [1.0] + [0.0] * 8 + ["", False])
        self.assertEqual(data[18:24], [False, 1.0, 0, 1.0, 0.0, 1])

    def test_text_mode_enables_experimental_choice_in_same_session(self):
        adapter = IndexTTSAdapter(options={"emotion_mode": "text"})
        client = mock.Mock()
        client.post.side_effect = [
            _Response(404), _Response(payload={"data": [{"choices": ["same", "ref", "vector", "text"]}]}),
        ]
        adapter._prepare_text_mode(client)
        payload = client.post.call_args.kwargs["json"]
        self.assertEqual(payload["data"], [True, "与音色参考音频相同"])
        self.assertEqual(payload["session_hash"], adapter._payload([])["session_hash"])

    def test_text_mode_reports_missing_qwen_emotion(self):
        adapter = IndexTTSAdapter(options={"emotion_mode": "text"})
        client = mock.Mock()
        client.post.side_effect = [_Response(404), _Response(payload={"data": [{"choices": [0, 1, 2]}]})]
        with self.assertRaisesRegex(TTSError, "QwenEmotion"):
            adapter._prepare_text_mode(client)

    def test_missing_segment_reference_is_explicit_error(self):
        adapter = IndexTTSAdapter(options={})
        with self.assertRaisesRegex(TTSError, "缺少原声"):
            adapter.for_segment(TTSSegmentContext(index=0))

    def test_segment_references_are_isolated_and_content_sensitive(self):
        def write_wave(path, value):
            with wave.open(path, "wb") as target:
                target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                target.writeframes(bytes([value, 0]) * 240)

        with tempfile.TemporaryDirectory() as directory:
            voice = os.path.join(directory, "voice.wav")
            write_wave(voice, 1)
            segments = []
            for index in range(2):
                path = os.path.join(directory, f"speech_{index}.wav")
                write_wave(path, index + 2)
                segments.append({"index": index, "text": "你好", "speaker": None, "speech_file": path})
            references = []
            def synthesize(adapter, text, output, rate):
                references.append(adapter._emotion_reference)
                with open(output, "wb") as target:
                    target.write(b"ID3" + bytes([len(references)]) * 300)
                return output
            settings = {"reference_audio": voice, "options": {"emotion_mode": "segment", "voice_source": "reference"}}
            with mock.patch.object(IndexTTSAdapter, "_synthesize_blocking", autospec=True, side_effect=synthesize):
                output = os.path.join(directory, "output")
                generate_tts(segments, output, backend="indextts-2.5", backend_kwargs=settings)
                self.assertCountEqual(references, [s["speech_file"] for s in segments])
                generate_tts(segments, output, backend="indextts-2.5", backend_kwargs=settings)
                self.assertEqual(len(references), 2)
                write_wave(segments[0]["speech_file"], 9)
                generate_tts(segments, output, backend="indextts-2.5", backend_kwargs=settings)
                self.assertEqual(len(references), 3)
                self.assertEqual(references[-1], segments[0]["speech_file"])
                settings["options"]["max_mel_tokens"] = 2000
                generate_tts(segments, output, backend="indextts-2.5", backend_kwargs=settings)
                self.assertEqual(len(references), 5)

    def test_automatic_voice_and_emotion_use_each_source_without_manual_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "speech.wav")
            with wave.open(path, "wb") as target:
                target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                target.writeframes(b"\x01\x00" * 240)
            original = IndexTTSAdapter(reference_audio="unused-manual.wav", options={})
            bound = original.for_segment(TTSSegmentContext(0, path))
            self.assertEqual(bound._reference_audio, path)
            self.assertIsNone(bound._emotion_reference)
            self.assertEqual(original._reference_audio, "")
            self.assertFalse(original._segment_bound)
            data = bound._request_data("你好", {"path": path}, None, "+0%")
            self.assertEqual(data[0], "与音色参考音频相同")
            self.assertEqual(data[5], 1.0)
            output = os.path.join(directory, "out.mp3")
            with mock.patch("modules.tts.indextts_adapter.httpx.Client", _Client):
                asyncio.run(bound.synthesize("你好", "", output))
            self.assertTrue(os.path.isfile(output))
            self.assertEqual(sum(url.endswith("/upload") for _, url, _ in _Client.calls), 1)

    def test_automatic_voice_can_combine_with_independent_emotion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "speech.wav")
            with wave.open(path, "wb") as target:
                target.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                target.writeframes(b"\x01\x00" * 240)
            adapter = IndexTTSAdapter(options={"emotion_mode": "reference", "emotion_reference": "emotion.wav"})
            bound = adapter.for_segment(TTSSegmentContext(0, path))
            self.assertEqual(bound._reference_audio, path)
            self.assertEqual(bound._emotion_reference, "emotion.wav")

    def test_automatic_mode_requires_source_context_before_synthesis(self):
        with self.assertRaisesRegex(TTSError, "分段原声上下文"):
            IndexTTSAdapter(options={})._synthesize_blocking("你好", "out.mp3", "+0%")

    def setUp(self):
        _upload_cache.clear()
        _Client.calls = []

    def test_registered(self):
        self.assertIn("indextts-2.5", get_available_backends())
        adapter = create_tts("indextts-2.5")
        self.assertIsInstance(adapter, IndexTTSAdapter)
        self.assertEqual(adapter._base_url, "http://127.0.0.1:7860")

    def test_invalid_timeout_is_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                IndexTTSAdapter(timeout=value)

    def test_queued_generation_preserves_request_and_downloads_output(self):
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path.endswith("/upload"):
                return httpx.Response(200, json=["remote/ref.wav"])
            if request.method == "POST":
                return httpx.Response(200, json={"event_id": "job-1"})
            if request.url.path.endswith("/job-1"):
                return httpx.Response(200, text=(
                    'event: heartbeat\ndata: null\n\n'
                    'event: generating\ndata: [null]\n\n'
                    'event: complete\ndata: [{"url": "/out.mp3"}]\n\n'
                ))
            return httpx.Response(200, content=b"ID3" + b"x" * 300)

        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as source:
                source.write(b"x" * 300)
            out = os.path.join(tmp, "out.mp3")
            adapter = IndexTTSAdapter(reference_audio=ref)
            client = httpx.Client(transport=httpx.MockTransport(handle))
            with mock.patch("modules.tts.indextts_adapter.sync_client", return_value=client):
                self.assertEqual(asyncio.run(adapter.synthesize("你好", "reference", out)), out)
            self.assertEqual([r.url.path for r in requests], [
                "/gradio_api/upload", "/gradio_api/call/gen_single",
                "/gradio_api/call/gen_single/job-1", "/out.mp3",
            ])
            self.assertIn("你好", requests[1].content.decode("utf-8"))
            with open(out, "rb") as source:
                self.assertTrue(source.read().startswith(b"ID3"))

    def test_queue_failure_and_missing_completion_are_explicit(self):
        for events, expected in (
            ('event: error\ndata: "GPU failed"\n\n', "GPU failed"),
            ('event: heartbeat\ndata: null\n\n', "未收到完成事件"),
            ('event: complete\ndata: invalid\n\n', "无法识别"),
        ):
            with self.subTest(events=events):
                with httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text=events)
                )) as client:
                    with self.assertRaisesRegex(TTSError, expected):
                        IndexTTSAdapter()._wait_generation(client, "http://host/call/gen", "job")

    def test_heartbeats_do_not_extend_job_deadline(self):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text='event: heartbeat\ndata: null\n\n'
        ))) as client:
            with mock.patch("modules.tts.indextts_adapter.time.monotonic", side_effect=[0, 11]):
                with self.assertRaises(httpx.ReadTimeout):
                    IndexTTSAdapter(timeout=10)._wait_generation(client, "http://host/call/gen", "job")

    def test_generation_timeout_is_explained_without_resubmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as source:
                source.write(b"x" * 300)
            adapter = IndexTTSAdapter(reference_audio=ref, timeout=10, max_retries=3)
            with mock.patch("modules.tts.indextts_adapter.httpx.Client", _Client):
                with mock.patch.object(adapter, "_generate", side_effect=httpx.ReadTimeout("timed out")) as generate:
                    with self.assertRaisesRegex(TTSError, "提交/等待合成超时.*10 秒.*ReadTimeout"):
                        asyncio.run(adapter.synthesize("你好", "reference", os.path.join(tmp, "out.mp3")))
            self.assertEqual(generate.call_count, 1)

    def test_request_contract_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as f:
                f.write(b"RIFF" + b"x" * 300)
            out = os.path.join(tmp, "out.mp3")
            adapter = IndexTTSAdapter(reference_audio=ref, max_retries=1)
            with mock.patch("modules.tts.indextts_adapter.httpx.Client", _Client):
                result = asyncio.run(adapter.synthesize("你好", "reference", out, "+10%"))
            self.assertEqual(result, out)
            with open(out, "rb") as f:
                self.assertTrue(f.read().startswith(b"ID3"))
            synthesis = next(c for c in _Client.calls if c[1].endswith("/run/gen_single"))
            data = synthesis[2]["json"]["data"]
            self.assertEqual(data[0], "与音色参考音频相同")
            self.assertEqual(data[2], "你好")
            self.assertEqual(data[3], "ZH")
            self.assertAlmostEqual(data[17], 1 / 1.1)

    def test_gradio_route_is_discovered_without_user_configuration(self):
        adapter = IndexTTSAdapter(base_url="http://192.168.1.20:9000")
        client = mock.Mock()
        client.post.side_effect = [
            _Response(status_code=404),
            _Response(payload={"data": [{"url": "/generated.wav"}]}),
        ]
        client.get.return_value = _Response(payload={
            "api_prefix": "/gradio_api",
            "components": [{"id": 9, "type": "audio"}],
            "dependencies": [{
                "api_name": "custom_generate",
                "outputs": [9],
            }],
        })

        response = adapter._post_generation(client, ["payload"])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            client.post.call_args_list[1].args[0],
            "http://192.168.1.20:9000/gradio_api/run/custom_generate",
        )
        self.assertEqual(adapter._generation_url(), client.post.call_args_list[1].args[0])

    def test_queue_route_is_discovered_and_reused(self):
        requests = []

        def handle(request):
            requests.append(request.url.path)
            if request.url.path == "/config":
                return httpx.Response(200, json={
                    "api_prefix": "/custom",
                    "components": [{"id": 9, "type": "audio"}],
                    "dependencies": [{"api_name": "generate", "outputs": [9]}],
                })
            if request.url.path == "/gradio_api/call/gen_single":
                return httpx.Response(404)
            if request.method == "POST":
                return httpx.Response(200, json={"event_id": "job"})
            return httpx.Response(200, text='event: complete\ndata: [{"url": "/out.wav"}]\n\n')

        adapter = IndexTTSAdapter(base_url="http://host")
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            for _ in range(2):
                self.assertEqual(adapter._generate(client, ["text"]), {"data": [{"url": "/out.wav"}]})
        self.assertEqual(requests.count("/config"), 1)
        self.assertEqual(requests.count("/custom/call/generate"), 2)

    def test_remote_service_rewrites_loopback_asset_url(self):
        adapter = IndexTTSAdapter(base_url="http://192.168.1.20:9000")
        self.assertEqual(
            adapter._asset_url("http://127.0.0.1:9000/gradio_api/file=out.wav"),
            "http://192.168.1.20:9000/gradio_api/file=out.wav",
        )

    def test_audio_download_retries_truncated_gradio_response(self):
        adapter = IndexTTSAdapter()
        response = _Response(content=b"ID3" + b"x" * 300)
        client = mock.Mock()
        client.get.side_effect = [
            httpx.RemoteProtocolError("stale Content-Length"),
            response,
        ]
        self.assertIs(adapter._download_audio(client, "http://host/out.wav"), response)
        self.assertEqual(client.get.call_count, 2)

    def test_upload_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.wav")
            with open(ref, "wb") as f:
                f.write(b"x" * 300)
            adapter = IndexTTSAdapter(reference_audio=ref, max_retries=1)
            with mock.patch("modules.tts.indextts_adapter.httpx.Client", _Client):
                asyncio.run(adapter.synthesize("一", "reference", os.path.join(tmp, "1.mp3")))
                asyncio.run(adapter.synthesize("二", "reference", os.path.join(tmp, "2.mp3")))
            uploads = [c for c in _Client.calls if c[1].endswith("/upload")]
            self.assertEqual(len(uploads), 1)
            queue_attempts = [c for c in _Client.calls if "/call/" in c[1]]
            self.assertEqual(len(queue_attempts), 1)


if __name__ == "__main__":
    unittest.main()
