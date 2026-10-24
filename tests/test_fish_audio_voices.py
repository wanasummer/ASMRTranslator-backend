import asyncio
import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from modules.error import TTSError
from modules.tts.fish_audio_adapter import FishAudioTTSAdapter


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", content=b"audio"):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = content

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.close_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.responses.pop(0)

    async def delete(self, url, **kwargs):
        self.calls.append(("delete", url, kwargs))
        return self.responses.pop(0)

    async def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.responses.pop(0)

    async def aclose(self):
        self.close_count += 1


class FishAudioVoiceManagementTests(unittest.TestCase):
    def test_sync_synthesis_closes_client_before_event_loop_ends(self):
        class LoopBoundClient:
            def __init__(self, **_kwargs):
                self.loop = None
                self.closed = False

            async def post(self, *_args, **_kwargs):
                loop = asyncio.get_running_loop()
                if self.loop is not None and self.loop is not loop:
                    raise RuntimeError("Event loop is closed")
                self.loop = loop
                return FakeResponse(content=b"audio" * 60)

            async def aclose(self):
                self.closed = True
                self.assert_same_loop = self.loop is asyncio.get_running_loop()

        clients = []

        def create_client(**kwargs):
            client = LoopBoundClient(**kwargs)
            clients.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory, patch(
            "modules.tts.fish_audio_adapter.httpx.AsyncClient", side_effect=create_client
        ):
            adapter = FishAudioTTSAdapter(api_key="secret")
            for index in range(2):
                path = str(Path(directory) / f"tts_{index}.mp3")
                self.assertEqual(adapter.synthesize_sync("你好", "default", path), path)
                self.assertEqual(Path(path).read_bytes(), b"audio" * 60)

        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.closed and client.assert_same_loop for client in clients))

    def test_synthesis_reuses_one_owned_connection_pool(self):
        client = FakeClient([FakeResponse(content=b"one"), FakeResponse(content=b"two")])

        async def exercise():
            adapter = FishAudioTTSAdapter(api_key="secret")
            self.assertEqual(await adapter._call_api("一", None), b"one")
            self.assertEqual(await adapter._call_api("二", None), b"two")
            await adapter.aclose()

        with patch("modules.tts.fish_audio_adapter.httpx.AsyncClient", return_value=client) as factory:
            asyncio.run(exercise())

        factory.assert_called_once_with(timeout=180.0, trust_env=False, transport=ANY)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.close_count, 1)

    def test_list_owned_voices_normalizes_names_and_paginates(self):
        client = FakeClient([
            FakeResponse(payload={
                "items": [{"_id": "voice-1", "title": "柔和旁白", "visibility": "private"}],
                "has_more": True,
            }),
            FakeResponse(payload={
                "items": [{"reference_id": "voice-2", "name": "耳语", "tags": ["asmr"]}],
                "has_more": False,
            }),
        ])
        with patch("modules.tts.fish_audio_adapter.httpx.AsyncClient", return_value=client):
            voices = asyncio.run(FishAudioTTSAdapter(api_key="secret").get_voices())

        self.assertEqual([voice["reference_id"] for voice in voices], ["voice-1", "voice-2"])
        self.assertEqual([voice["name"] for voice in voices], ["柔和旁白", "耳语"])
        self.assertEqual(client.calls[0][2]["params"]["self"], "true")
        self.assertEqual(client.calls[1][2]["params"]["page_number"], 2)
        self.assertEqual(client.calls[0][2]["headers"]["Authorization"], "Bearer secret")

    def test_delete_owned_voice(self):
        client = FakeClient([FakeResponse(status_code=204)])
        with patch("modules.tts.fish_audio_adapter.httpx.AsyncClient", return_value=client):
            asyncio.run(FishAudioTTSAdapter(api_key="secret").delete_voice("voice-1"))

        self.assertEqual(client.calls[0][0], "delete")
        self.assertTrue(client.calls[0][1].endswith("/model/voice-1"))

    def test_voice_management_requires_api_key(self):
        with patch.dict(os.environ, {"FISH_API_KEY": ""}), self.assertRaises(TTSError):
            asyncio.run(FishAudioTTSAdapter().get_voices())

    def test_design_voice_uses_json_and_writes_candidate_wav(self):
        audio = b"RIFF" + b"voice" * 40
        client = FakeClient([FakeResponse(payload={"candidates": [{
            "id": "candidate-1",
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "sample_rate": 44100,
            "duration_ms": 3200,
        }]})])
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "candidate.wav")
            with patch("modules.tts.fish_audio_adapter.httpx.AsyncClient", return_value=client):
                result = asyncio.run(FishAudioTTSAdapter(api_key="secret").design_voice(
                    "温柔的年轻女性声音", "晚上好", output, language="zh"
                ))
            self.assertEqual(Path(output).read_bytes(), audio)

        call = client.calls[0]
        self.assertTrue(call[1].endswith("/v1/voice-design"))
        self.assertNotIn("files", call[2])
        self.assertEqual(call[2]["headers"]["model"], "voice-design-1")
        self.assertEqual(call[2]["json"]["instruction"], "温柔的年轻女性声音")
        self.assertEqual(result["candidate_id"], "candidate-1")

    def test_create_voice_uses_private_multipart_model(self):
        client = FakeClient([FakeResponse(status_code=201, payload={
            "_id": "model-1", "title": "我的音色", "state": "trained"
        })])
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.wav"
            sample.write_bytes(b"RIFF" + b"audio" * 40)
            with patch("modules.tts.fish_audio_adapter.httpx.AsyncClient", return_value=client):
                result = asyncio.run(FishAudioTTSAdapter(api_key="secret").create_voice(
                    "我的音色", [str(sample)], description="测试", texts=["晚上好"]
                ))

        call = client.calls[0]
        self.assertEqual(call[1], FishAudioTTSAdapter._MODELS_URL)
        self.assertNotIn("Content-Type", call[2]["headers"])
        self.assertEqual(call[2]["data"]["visibility"], "private")
        self.assertEqual(call[2]["data"]["texts"], ["晚上好"])
        self.assertEqual(call[2]["files"][0][0], "voices")
        self.assertEqual(result["reference_id"], "model-1")


if __name__ == "__main__":
    unittest.main()
