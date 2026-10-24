import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
from unittest.mock import AsyncMock, patch

import httpx

from modules import preflight
from modules.error import LLMError


class PreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_common_tts_task_preflight_defers_segment_roles_but_checks_fixed_files(self):
        manifest = {
            "schema_version": 1, "protocol": "common-tts/1", "id": "source-tts", "name": "Source TTS", "version": "1",
            "capabilities": {"required_assets": ["source_speech", "source_segment"]},
            "parameters": {"type": "object", "additionalProperties": False, "properties": {
                "reference": {"type": "string", "x-control": "audio-file", "x-role": "voice_reference"},
            }},
        }
        requests = []
        def health(request):
            requests.append(request.url.path)
            return httpx.Response(200, json={"contract_version": 1, "ready": True,
                "plugin_id": "source-tts", "plugin_version": "1"})

        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.wav"
            with wave.open(str(reference), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(b"\0\0" * 80)
            config = {"ttsBackend": "common-tts", "commonTtsBaseUrl": "http://plugin",
                "commonTtsManifest": manifest, "commonTtsParameters": {}, "probeOnly": False}
            with patch("modules.tts.common_tts_adapter.async_client",
                       side_effect=lambda **_: httpx.AsyncClient(transport=httpx.MockTransport(health))):
                # Neither a text-only configuration nor a fixed reference implies
                # that the slicing stage has already produced the original speech.
                for parameters in ({}, {"reference": str(reference)}):
                    result = await preflight.run_preflight_check("tts", {**config, "commonTtsParameters": parameters})
                    self.assertTrue(result.success, result.message)
            with patch("modules.tts.common_tts_adapter.async_client") as network:
                result = await preflight.run_preflight_check("tts", {**config,
                    "commonTtsManifest": {**manifest, "execution": {"max_upload_bytes": 1}},
                    "commonTtsParameters": {"reference": str(reference)}})
            self.assertFalse(result.success)
            self.assertIn("max_upload_bytes", result.message)
            network.assert_not_called()
            reference.unlink()
            with patch("modules.tts.common_tts_adapter.async_client") as network:
                result = await preflight.run_preflight_check("tts", {**config,
                    "commonTtsParameters": {"reference": str(reference)}})
            self.assertFalse(result.success)
            network.assert_not_called()
        self.assertEqual(requests, ["/health", "/health"])

    async def test_index_automatic_mode_does_not_require_uploaded_voice(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        with patch("modules.preflight.async_client", return_value=client):
            result = await preflight.run_preflight_check("tts", {
                "ttsBackend": "indextts-2.5", "indexTtsBaseUrl": "http://127.0.0.1",
                "indexTtsPort": 7860, "indexTtsReferenceAudio": "", "indexTtsOptions": {},
            })
        self.assertTrue(result.success, result.message)

    async def test_index_fixed_voice_requires_reference(self):
        result = await preflight.run_preflight_check("tts", {
            "ttsBackend": "indextts-2.5", "indexTtsOptions": {"voice_source": "reference"},
        })
        self.assertFalse(result.success)
        self.assertIn("固定音色参考", result.message)

    async def test_index_connection_probe_does_not_require_generation_references(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        with patch("modules.preflight.async_client", return_value=client):
            result = await preflight.run_preflight_check("tts", {
                "ttsBackend": "indextts-2.5", "indexTtsBaseUrl": "http://127.0.0.1",
                "indexTtsPort": 7860, "indexTtsOptions": {"voice_source": "reference", "emotion_mode": "reference"},
                "probeOnly": True,
            })
        self.assertTrue(result.success, result.message)

    async def test_common_tts_connection_probe_does_not_require_workspace_references(self):
        manifest = {
            "schema_version": 1, "protocol": "common-tts/1", "id": "audio8-tts", "name": "Audio8 TTS", "version": "0.4.0",
            "parameters": {
                "type": "object", "additionalProperties": False, "required": ["reference_audio", "reference_text"],
                "properties": {
                    "reference_audio": {"type": "string", "minLength": 1, "default": "", "x-control": "audio-file"},
                    "reference_text": {"type": "string", "minLength": 1, "default": ""},
                },
            },
        }
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
            "contract_version": 1, "ready": True, "plugin_id": "audio8-tts", "plugin_version": "0.4.0",
        })))
        with patch("modules.tts.common_tts_adapter.async_client", return_value=client):
            probe = await preflight.run_preflight_check("tts", {
                "ttsBackend": "common-tts", "commonTtsBaseUrl": "http://127.0.0.1:8004",
                "commonTtsManifest": manifest, "commonTtsParameters": {"reference_audio": "", "reference_text": ""},
                "probeOnly": True,
            })
            self.assertTrue(probe.success, probe.message)
            full = await preflight.run_preflight_check("tts", {
                "ttsBackend": "common-tts", "commonTtsBaseUrl": "http://127.0.0.1:8004",
                "commonTtsManifest": manifest, "commonTtsParameters": {"reference_audio": "", "reference_text": ""},
                "probeOnly": False,
            })
            self.assertFalse(full.success)
            self.assertIn("reference_audio", full.message)

    async def test_subtitles_only_never_checks_tts_services(self):
        with patch("modules.preflight.edge_tts.list_voices", new_callable=AsyncMock) as voices, \
                patch("modules.preflight.httpx.AsyncClient") as client:
            result = await preflight.run_preflight_check("tts", {"ttsBackend": "subtitles-only"})
        self.assertTrue(result.success)
        voices.assert_not_awaited()
        client.assert_not_called()

    async def test_dispatches_a_supported_check(self):
        probe = AsyncMock()
        with patch.dict(preflight._CHECKS, {"llm": probe}, clear=True):
            result = await preflight.run_preflight_check("llm", {"llmProvider": "openai"})

        self.assertTrue(result.success)
        self.assertEqual(result.message, "配置正确，服务连接正常")
        probe.assert_awaited_once()

    async def test_returns_service_failure_as_data(self):
        probe = AsyncMock(side_effect=preflight.PreflightError("配置错误"))
        with patch.dict(preflight._CHECKS, {"tts": probe}, clear=True):
            result = await preflight.run_preflight_check("tts", {})

        self.assertFalse(result.success)
        self.assertEqual(result.message, "配置错误")

    async def test_llm_probe_uses_production_client_contract(self):
        config = {
            "llmProvider": "anthropic",
            "llmBaseUrl": "https://api.deepseek.com/anthropic",
            "llmApiKey": "secret",
            "llmModel": "deepseek-chat",
            "ttsBackend": "fish-audio",
            "fishModel": "s2.1-pro-free",
        }
        with patch("modules.preflight.LLMClient") as client_type:
            client_type.return_value.check_connection = AsyncMock(return_value="好")
            await preflight._check_llm(config)

        client_type.return_value.check_connection.assert_awaited_once()
        llm_config = client_type.call_args.args[0]
        self.assertEqual(llm_config.provider, "anthropic")
        self.assertEqual(llm_config.base_url, "https://api.deepseek.com/anthropic")
        self.assertEqual(llm_config.model, "deepseek-chat")

    async def test_llm_probe_reports_busy_or_rate_limited_service(self):
        config = {"llmProvider": "openai", "llmApiKey": "secret", "llmModel": "gemini-3.5-flash"}
        for status, expected in ((503, "暂时繁忙"), (429, "频率或额度上限")):
            with self.subTest(status=status):
                error = LLMError(f"openai LLM returned {status}")
                error.context["status"] = status
                with patch("modules.preflight.LLMClient") as client_type:
                    client_type.return_value.check_connection = AsyncMock(side_effect=error)
                    with self.assertRaisesRegex(preflight.PreflightError, expected):
                        await preflight._check_llm(config)

    async def test_common_asr_health_is_checked_with_httpx(self):
        real_async_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "http://127.0.0.1:8000/health")
            return httpx.Response(200, json={
                "status": "ok",
                "ready": True,
                "capabilities": {"timestamps": True, "external_segments": True},
            })

        def client_factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        config = {
            "asrEngine": "common-asr",
            "asrBaseUrl": "http://127.0.0.1",
            "asrPort": 8000,
            "asrCustomApiKey": "",
        }
        with patch("modules.preflight.httpx.AsyncClient", side_effect=client_factory):
            await preflight._check_asr(config)

    async def test_edge_tts_rejects_an_unknown_selected_voice(self):
        config = {"ttsBackend": "edge-tts", "edgeVoice": "missing"}
        with patch("modules.preflight.edge_tts.list_voices", new=AsyncMock(return_value=[
            {"ShortName": "zh-CN-XiaoxiaoNeural"},
        ])):
            with self.assertRaisesRegex(preflight.PreflightError, "不存在所选音色"):
                await preflight._check_tts(config)

    async def test_qwen_and_mimo_asr_probe_their_domestic_endpoints(self):
        real_async_client = httpx.AsyncClient
        seen = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), request.headers.get("api-key"), request.headers.get("authorization")))
            if "multimodal-generation" in request.url.path:
                return httpx.Response(400, json={"code": "CLIENT_ERROR", "message": "ASR_RESPONSE_HAVE_NO_WORDS"})
            return httpx.Response(200, json={"data": []})

        def client_factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        with patch("modules.preflight.httpx.AsyncClient", side_effect=client_factory):
            await preflight._check_asr({
                "asrEngine": "qwen-audio-3.1-asr-flash",
                "qwenAsrApiKey": "qwen-key",
                "qwenAsrWorkspaceId": "workspace",
            })
            await preflight._check_asr({"asrEngine": "mimo-asr", "mimoAsrApiKey": "mimo-key"})
        self.assertEqual(seen[0][2], "Bearer qwen-key")
        self.assertIn("workspace.cn-beijing.maas.aliyuncs.com", seen[0][0])
        self.assertEqual(seen[1][1], "mimo-key")

    async def test_qwen_asr_requires_workspace_id(self):
        with self.assertRaisesRegex(preflight.PreflightError, "Workspace ID"):
            await preflight._check_asr({
                "asrEngine": "qwen-audio-3.1-asr-flash",
                "qwenAsrApiKey": "qwen-key",
            })

    async def test_fun_asr_does_not_require_workspace_id(self):
        real_async_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/uploads"):
                return httpx.Response(200, json={"data": {"policy": "ok"}})
            self.assertEqual(request.url.host, "dashscope.aliyuncs.com")
            return httpx.Response(400, json={"code": "InvalidParameter", "message": "file_urls size must be [1, 100]"})

        def client_factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        with patch("modules.preflight.httpx.AsyncClient", side_effect=client_factory):
            await preflight._check_asr({
                "asrEngine": "fun-asr",
                "qwenAsrApiKey": "qwen-key",
                "qwenAsrWorkspaceId": "",
            })

    async def test_fish_tts_probe_only_does_not_require_a_voice(self):
        real_async_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/model")
            return httpx.Response(200, json={"items": []})

        def client_factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        with patch("modules.preflight.httpx.AsyncClient", side_effect=client_factory):
            await preflight._check_tts({
                "ttsBackend": "fish-audio",
                "fishApiKey": "key",
                "probeOnly": True,
            })

    async def test_minimax_tts_probe_only_does_not_require_a_voice(self):
        real_async_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/get_voice")
            return httpx.Response(200, json={"base_resp": {"status_code": 0}})

        def client_factory(**kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        with patch("modules.preflight.httpx.AsyncClient", side_effect=client_factory):
            await preflight._check_tts({
                "ttsBackend": "minimax-tts",
                "minimaxApiKey": "key",
                "probeOnly": True,
            })

    def test_payload_requires_a_config_object(self):
        self.assertEqual(
            preflight.run_preflight_payload({"service": "llm", "config": None}),
            {"success": False, "message": "预检配置格式无效"},
        )


class PreflightProtocolTests(unittest.TestCase):
    def test_diagnostics_do_not_corrupt_stdout_json(self):
        import main as app_main

        stdout = io.StringIO()
        stderr = io.StringIO()

        def fake_check(_payload):
            print('[llm-http-error] {"status":503}')
            return {"success": False, "message": "大模型服务暂时繁忙（HTTP 503）"}

        with patch.object(sys, "argv", ["main.py", "--preflight"]), \
             patch.object(sys, "stdin", io.StringIO('{"service":"llm","config":{}}')), \
             patch("modules.preflight.run_preflight_payload", side_effect=fake_check), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exit_result:
                app_main.main()

        self.assertEqual(exit_result.exception.code, 0)
        self.assertEqual(json.loads(stdout.getvalue()),
                         {"success": False, "message": "大模型服务暂时繁忙（HTTP 503）"})
        self.assertIn("[llm-http-error]", stderr.getvalue())
        self.assertNotIn("[llm-http-error]", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
