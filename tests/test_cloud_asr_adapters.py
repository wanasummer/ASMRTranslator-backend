import json
import tempfile
import unittest
import wave
from pathlib import Path

import httpx

from modules.asr import FunASRAdapter, MimoASRAdapter, QwenAudio31ASRFlashAdapter
from modules.error import ASRError
from modules.asr.no_speech import is_no_speech


def write_wav(path: Path, seconds: float = 1.0) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))


class CloudASRAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_shared_no_speech_matcher_handles_nested_errors_without_scanning_transcript_text(self):
        self.assertTrue(is_no_speech({
            "output": {"results": [{"message": "ASR_RESPONSE_HAVE_NO_WORDS"}]},
        }))
        self.assertFalse(is_no_speech({
            "output": {"text": "He literally said no speech detected."},
        }))

    async def test_qwen_audio_31_sends_japanese_base64_and_builds_slice_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "slice.wav"
            write_wav(audio_path)

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content)
                self.assertEqual(payload["model"], "qwen-audio-3.1-asr-flash")
                self.assertEqual(payload["parameters"], {"format": "wav", "sample_rate": 16000})
                audio = payload["input"]["messages"][0]["content"][0]
                self.assertEqual(audio["type"], "input_audio")
                self.assertTrue(audio["input_audio"]["data"].startswith("data:audio/"))
                self.assertEqual(request.headers["X-DashScope-SSE"], "disable")
                self.assertEqual(request.url.host, "workspace.cn-beijing.maas.aliyuncs.com")
                self.assertTrue(request.url.path.endswith("/services/aigc/multimodal-generation/generation"))
                return httpx.Response(200, json={"output": {"text": "こんばんは"}})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await QwenAudio31ASRFlashAdapter(api_key="key", workspace_id="workspace").transcribe(str(audio_path), client=client)
            self.assertEqual(result.text, "こんばんは")
            self.assertEqual(result.segments[0].end, 1.0)

    async def test_qwen_audio_31_treats_no_words_as_an_empty_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "non_speech.wav"
            write_wav(audio_path, 0.5)

            async def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(400, json={
                    "request_id": "request-id",
                    "code": "CLIENT_ERROR",
                    "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                })

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await QwenAudio31ASRFlashAdapter(
                    api_key="key",
                    workspace_id="workspace",
                ).transcribe(str(audio_path), client=client)

            self.assertEqual(result.text, "")
            self.assertEqual(result.segments, [])
            self.assertAlmostEqual(result.duration, 0.5)
            self.assertEqual(result.language_code, "ja")

    def test_qwen_audio_31_parses_nested_sentence_response(self):
        text = QwenAudio31ASRFlashAdapter._response_text({
            "output": {"output": {"sentence": {"text": "おやすみ"}}},
        })
        self.assertEqual(text, "おやすみ")

    async def test_qwen_probe_retries_a_transient_connect_timeout(self):
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectTimeout("temporary connection timeout", request=request)
            return httpx.Response(400, json={"code": "CLIENT_ERROR", "message": "ASR_RESPONSE_HAVE_NO_WORDS"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await QwenAudio31ASRFlashAdapter(api_key="key", workspace_id="workspace").check_connection(client)
        self.assertEqual(attempts, 2)

    async def test_qwen_probe_explains_a_persistent_connect_timeout(self):
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            raise httpx.ConnectTimeout("", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(ASRError, "连接北京业务空间接口超时"):
                await QwenAudio31ASRFlashAdapter(api_key="key", workspace_id="workspace").check_connection(client)
        self.assertEqual(attempts, 2)

    async def test_mimo_uses_documented_auto_language_for_japanese_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "slice.wav"
            write_wav(audio_path, 0.5)

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content)
                self.assertEqual(request.headers["api-key"], "key")
                self.assertEqual(payload["model"], "mimo-v2.5-asr")
                self.assertEqual(payload["asr_options"]["language"], "auto")
                return httpx.Response(200, json={"choices": [{"message": {"content": "おやすみ"}}]})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await MimoASRAdapter(api_key="key").transcribe(str(audio_path), language="ja", client=client)
            self.assertEqual(result.language_code, "ja")
            self.assertAlmostEqual(result.segments[0].end, 0.5)

    async def test_mimo_treats_explicit_no_speech_error_as_empty_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "silent.wav"
            write_wav(audio_path, 0.5)

            async def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(400, json={"code": "NO_SPEECH_DETECTED"})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await MimoASRAdapter(api_key="key").transcribe(str(audio_path), client=client)

            self.assertEqual(result.text, "")
            self.assertEqual(result.segments, [])

    async def test_fun_asr_uploads_submits_polls_and_parses_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "slice.wav"
            write_wav(audio_path)
            requests = []

            async def handler(request: httpx.Request) -> httpx.Response:
                requests.append((request.method, str(request.url)))
                if request.url.path.endswith("/uploads"):
                    self.assertEqual(request.url.host, "dashscope.aliyuncs.com")
                    return httpx.Response(200, json={"data": {
                        "upload_host": "https://upload.example.com",
                        "upload_dir": "temporary/job",
                        "oss_access_key_id": "access",
                        "policy": "policy",
                        "signature": "signature",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }})
                if request.url.host == "upload.example.com":
                    return httpx.Response(200)
                if request.url.path.endswith("/services/audio/asr/transcription"):
                    self.assertEqual(request.url.host, "dashscope.aliyuncs.com")
                    self.assertEqual(request.headers["X-DashScope-OssResourceResolve"], "enable")
                    payload = json.loads(request.content)
                    self.assertEqual(payload["model"], "fun-asr")
                    self.assertEqual(payload["parameters"]["language_hints"], ["ja"])
                    self.assertTrue(payload["input"]["file_urls"][0].startswith("oss://"))
                    return httpx.Response(200, json={"output": {"task_id": "task-1"}})
                if request.url.path.endswith("/tasks/task-1"):
                    return httpx.Response(200, json={"output": {
                        "task_status": "SUCCEEDED",
                        "results": [{"subtask_status": "SUCCEEDED", "transcription_url": "https://result.example.com/result.json"}],
                    }})
                if request.url.host == "result.example.com":
                    return httpx.Response(200, json={"transcripts": [{
                        "text": "こんにちは",
                        "sentences": [{"text": "こんにちは", "begin_time": 100, "end_time": 900}],
                    }]})
                return httpx.Response(404)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await FunASRAdapter(api_key="key", poll_interval=0.01).transcribe(str(audio_path), client=client)
            self.assertEqual(result.text, "こんにちは")
            self.assertEqual((result.segments[0].start, result.segments[0].end), (0.1, 0.9))
            self.assertEqual(len(requests), 5)

    async def test_fun_asr_failed_no_words_task_is_an_empty_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "silent.wav"
            write_wav(audio_path, 0.5)

            async def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/uploads"):
                    return httpx.Response(200, json={"data": {
                        "upload_host": "https://upload.example.com",
                        "upload_dir": "temporary/job",
                        "oss_access_key_id": "access",
                        "policy": "policy",
                        "signature": "signature",
                    }})
                if request.url.host == "upload.example.com":
                    return httpx.Response(200)
                if request.url.path.endswith("/services/audio/asr/transcription"):
                    return httpx.Response(200, json={"output": {"task_id": "task-1"}})
                if request.url.path.endswith("/tasks/task-1"):
                    return httpx.Response(200, json={"output": {
                        "task_status": "FAILED",
                        "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                    }})
                return httpx.Response(404)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await FunASRAdapter(api_key="key", poll_interval=0.01).transcribe(
                    str(audio_path), client=client,
                )

            self.assertEqual(result.text, "")
            self.assertEqual(result.segments, [])
            self.assertAlmostEqual(result.duration, 0.5)

    def test_domestic_adapters_require_workspace_id(self):
        self.assertFalse(QwenAudio31ASRFlashAdapter(api_key="key").is_configured())
        self.assertTrue(FunASRAdapter(api_key="key").is_configured())
        self.assertTrue(QwenAudio31ASRFlashAdapter(api_key="key", workspace_id="workspace").is_configured())

    def test_qwen_probe_accepts_a_valid_silent_audio_response(self):
        response = httpx.Response(400, json={
            "code": "CLIENT_ERROR",
            "message": "ASR_RESPONSE_HAVE_NO_WORDS",
        })
        QwenAudio31ASRFlashAdapter._raise_for_probe(response)

    def test_domestic_adapters_report_unpurchased_models(self):
        response = httpx.Response(403, json={
            "code": "AccessDenied.Unpurchased",
            "message": "Access to model denied.",
        })
        with self.assertRaisesRegex(ASRError, "模型尚未开通"):
            QwenAudio31ASRFlashAdapter._raise_for_status(response)
        with self.assertRaisesRegex(ASRError, "模型尚未开通"):
            FunASRAdapter._raise_for_status(response, "检测工作空间")


if __name__ == "__main__":
    unittest.main()
