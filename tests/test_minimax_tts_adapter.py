"""Offline tests for modules.tts.minimax_tts_adapter (network fully mocked).

Run from the project root:  python -m unittest tests.test_minimax_tts_adapter
Covers the T2A v2 request/response contract: hex audio decoding, payload shape
(voice_setting / audio_setting / non-streaming), retry on transient error codes,
error surfaces, rate→speed mapping and factory registration.
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

from modules.error import TTSError
from modules.tts.factory import create_tts, get_available_backends
from modules.tts.minimax_tts_adapter import MiniMaxTTSAdapter

_HEX = "aabbccdd" * 100  # 400 bytes of valid hex audio


class _FakeResponse:
    """Scripted httpx response."""

    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text or str(json_body)

    def json(self):
        return self._json


def _ok_body(audio_hex=_HEX, status_code=0, status_msg="success"):
    return {
        "data": {"audio": audio_hex, "status": 2},
        "extra_info": {"audio_length": 4, "audio_format": "mp3"},
        "trace_id": "t-1",
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def _design_body(voice_id="ttv-voice-20260701-abcdef", status_code=0,
                 status_msg="success", with_audio=True):
    return {
        "voice_id": voice_id,
        "trial_audio": _HEX if with_audio else None,
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def _upload_body(file_id=12345, status_code=0, status_msg="success"):
    return {
        "file": {"file_id": file_id, "bytes": 1000, "filename": "sample.wav",
                 "purpose": "voice_clone"},
        "base_resp": {"status_code": status_code, "status_msg": status_msg},
    }


def _clone_body(voice_id="CloneVoice123456789", status_code=0,
                status_msg="success", demo_url="https://cdn.minimaxi.com/demo.mp3"):
    body = {"base_resp": {"status_code": status_code, "status_msg": status_msg}}
    if status_code == 0:
        body["demo_audio"] = demo_url
    return body


def _voices_body(status_code=0):
    return {
        "system_voice": [{"voice_id": "Chinese (Mandarin)_Warm_Girl"}],
        "voice_cloning": [{"voice_id": "MyClone01",
                           "created_time": "2026-08-01"}],
        "voice_generation": [{"voice_id": "MyDesign01",
                              "created_time": "2026-08-02"}],
        "base_resp": {"status_code": status_code, "status_msg": "success"},
    }


class _FakeTransport:
    """Patches httpx.AsyncClient.post; records calls, returns scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, headers=None, json=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "json": json,
                           "data": kwargs.get("data"), "files": kwargs.get("files")})
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        return _FakeResponse(200, _ok_body())


def _patch_transport(responses):
    """Context manager patching httpx.AsyncClient to a fake transport."""
    fake = _FakeTransport(responses)

    async def _post(self, url, **kwargs):
        return await fake.post(url, **kwargs)

    def _aenter(self):
        return self

    async def _aexit(self, *a):
        return None

    patcher = mock.patch.object(
        __import__("httpx").AsyncClient,
        "post",
        new=_post,
    )
    # httpx.AsyncClient() context manager is real; only post() is replaced.
    patcher.start()
    return fake


class MiniMaxSynthesizeTest(unittest.TestCase):
    def _adapter(self, **kw):
        kw.setdefault("api_key", "test-key")
        kw.setdefault("min_interval_sec", 0)  # unit tests: no global throttle
        return MiniMaxTTSAdapter(**kw)

    def test_success_decodes_hex_and_writes_file(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _ok_body())])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.mp3")
            result = asyncio.run(adapter.synthesize("你好", "v1", out))
            with open(out, "rb") as f:
                self.assertEqual(f.read(), bytes.fromhex(_HEX))
        self.assertEqual(result, out)
        # Request contract
        call = fake.calls[0]
        self.assertTrue(call["url"].endswith("/v1/t2a_v2"))
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        body = call["json"]
        self.assertEqual(body["model"], "speech-02-hd")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["output_format"], "hex")
        self.assertEqual(body["voice_setting"]["voice_id"], "v1")
        self.assertEqual(body["audio_setting"]["format"], "mp3")
        self.assertNotIn("language_boost", body)

    def test_language_boost_only_sent_when_explicit(self):
        adapter = self._adapter(language_boost="Chinese")
        fake = _patch_transport([_FakeResponse(200, _ok_body())])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(adapter.synthesize(
                "你好", "v1", os.path.join(tmp, "out.mp3")))
        self.assertEqual(fake.calls[0]["json"]["language_boost"], "Chinese")

    def test_rate_maps_to_speed(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _ok_body())])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(adapter.synthesize("hi", "v1",
                                           os.path.join(tmp, "a.mp3"), rate="+10%"))
        self.assertEqual(fake.calls[0]["json"]["voice_setting"]["speed"], 1.1)

    def test_rate_clamped_to_minimax_bounds(self):
        self.assertEqual(MiniMaxTTSAdapter._rate_to_speed("+300%", 1.0), 2.0)
        self.assertEqual(MiniMaxTTSAdapter._rate_to_speed("-90%", 1.0), 0.5)
        self.assertEqual(MiniMaxTTSAdapter._rate_to_speed("garbage", 1.0), 1.0)

    def test_api_error_code_raises(self):
        adapter = self._adapter(max_retries=1)
        body = _ok_body(status_code=2013, status_msg="bad params")
        _patch_transport([_FakeResponse(200, body)])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "2013"):
            asyncio.run(adapter.synthesize("hi", "v1", "out.mp3"))

    def test_permanent_error_does_not_retry(self):
        # A non-transient error (2013) must surface immediately — no retries,
        # no backoff waits. The fake transport raises on any extra call.
        adapter = self._adapter()  # max_retries=3 default
        body = _ok_body(status_code=2013, status_msg="bad params")
        fake = _patch_transport([_FakeResponse(200, body),
                                 AssertionError("must not be called again")])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "2013"):
            asyncio.run(adapter.synthesize("hi", "v1", "out.mp3"))
        self.assertEqual(len(fake.calls), 1)

    def test_transient_error_retries_then_succeeds(self):
        adapter = self._adapter()
        transient = _FakeResponse(200, _ok_body(status_code=1039, status_msg="tpm"))
        ok = _FakeResponse(200, _ok_body())
        fake = _patch_transport([transient, ok])
        self.addCleanup(mock.patch.stopall)
        with mock.patch("asyncio.sleep", new=asyncio.sleep):  # real short sleep ok
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "o.mp3")
                asyncio.run(adapter.synthesize("hi", "v1", out, rate="+0%"))
        self.assertEqual(len(fake.calls), 2)

    def test_null_data_raises(self):
        adapter = self._adapter(max_retries=1)
        body = {"data": None, "trace_id": "t-2",
                "base_resp": {"status_code": 0, "status_msg": "success"}}
        _patch_transport([_FakeResponse(200, body)])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "no audio"):
            asyncio.run(adapter.synthesize("hi", "v1", "out.mp3"))

    def test_http_error_raises(self):
        adapter = self._adapter(max_retries=1)
        _patch_transport([_FakeResponse(401, text="unauthorized")])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "401"):
            asyncio.run(adapter.synthesize("hi", "v1", "out.mp3"))

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            adapter = MiniMaxTTSAdapter()
        with self.assertRaisesRegex(TTSError, "MINIMAX_API_KEY"):
            asyncio.run(adapter.synthesize("hi", "v1", "out.mp3"))


class MiniMaxRegistrationTest(unittest.TestCase):
    def test_registered_in_factory(self):
        self.assertIn("minimax-tts", get_available_backends())
        adapter = create_tts("minimax-tts", voice_map={"default": "v0"})
        self.assertIsInstance(adapter, MiniMaxTTSAdapter)
        self.assertEqual(adapter.name, "minimax-tts")
        self.assertEqual(adapter.get_voice_for_speaker(None), "v0")

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            MiniMaxTTSAdapter(model="speech-99")

    def test_speech_02_models_accepted(self):
        for model in ("speech-02-hd", "speech-02-turbo"):
            self.assertEqual(MiniMaxTTSAdapter(model=model)._model, model)

    def test_available_voices_nonempty(self):
        voices = MiniMaxTTSAdapter().get_available_voices()
        self.assertTrue(voices)
        ids = {v.id for v in voices}
        self.assertIn("Chinese (Mandarin)_Warm_Girl", ids)
        # 目录只保留中文普通话女声：英文/日文音色应不存在
        self.assertNotIn("Japanese_Whisper_Belle", ids)
        self.assertNotIn("male-qn-qingse", ids)
        # 全部为女声（无男性音色 ID）
        male_prefixes = ("male-", "clever_boy", "cute_boy", "junlang_", "chunzhen_",
                         "lengdan_", "badao_shaoye", "cartoon_pig")
        self.assertFalse(
            any(v.id.startswith(male_prefixes) for v in voices),
            "音色目录不应包含男性音色",
        )


class MiniMaxThrottleTest(unittest.TestCase):
    """Global request throttle keeps the aggregate rate under the RPM cap."""

    def _adapter(self, **kw):
        kw.setdefault("api_key", "test-key")
        kw.setdefault("min_interval_sec", 0)  # unit tests: no global throttle
        return MiniMaxTTSAdapter(**kw)

    def _reset_throttle(self):
        from modules.tts import minimax_tts_adapter as mod
        mod._last_request_ts = 0.0

    def test_emotion_never_sent(self):
        # 项目策略：任何模型都不发送 emotion 字段。
        with tempfile.TemporaryDirectory() as tmp:
            for model in ("speech-2.8-hd", "speech-2.8-turbo",
                          "speech-2.6-hd", "speech-2.6-turbo",
                          "speech-02-hd", "speech-02-turbo"):
                adapter = self._adapter(model=model, emotion="calm")
                fake = _patch_transport([_FakeResponse(200, _ok_body())])
                self.addCleanup(mock.patch.stopall)
                asyncio.run(adapter.synthesize(
                    "hi", "v1", os.path.join(tmp, "out.mp3")))
                vs = fake.calls[0]["json"]["voice_setting"]
                self.assertNotIn("emotion", vs, f"{model} must not send emotion")
                mock.patch.stopall()

    def test_design_voice_success(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _design_body())])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "trial.mp3")
            vid = asyncio.run(adapter.design_voice(
                "温柔少女", "夜深了，你还醒着吗？", out))
            # 试听音频 hex 解码落盘
            with open(out, "rb") as f:
                self.assertEqual(f.read(), bytes.fromhex(_HEX))
        self.assertEqual(vid, "ttv-voice-20260701-abcdef")
        call = fake.calls[0]
        self.assertTrue(call["url"].endswith("/v1/voice_design"))
        self.assertEqual(call["json"]["prompt"], "温柔少女")
        self.assertEqual(call["json"]["preview_text"], "夜深了，你还醒着吗？")
        self.assertIs(call["json"]["aigc_watermark"], False)
        self.assertNotIn("voice_id", call["json"])  # 未传时不含 voice_id

    def test_design_voice_forwards_custom_voice_id(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _design_body(
            voice_id="MyDesignVoice01"))])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            vid = asyncio.run(adapter.design_voice(
                "p", "t", os.path.join(tmp, "trial.mp3"),
                voice_id="MyDesignVoice01"))
        self.assertEqual(vid, "MyDesignVoice01")
        self.assertEqual(fake.calls[0]["json"]["voice_id"], "MyDesignVoice01")

    def test_design_voice_error_code_raises(self):
        adapter = self._adapter(max_retries=1)
        _patch_transport([_FakeResponse(200, _design_body(status_code=2013,
                                                          status_msg="bad"))])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "2013"):
            asyncio.run(adapter.design_voice("p", "t", "o.mp3"))

    def test_design_voice_transient_retries_then_succeeds(self):
        adapter = self._adapter()
        transient = _FakeResponse(200, _design_body(status_code=1039, status_msg="tpm"))
        ok = _FakeResponse(200, _design_body())
        fake = _patch_transport([transient, ok])
        self.addCleanup(mock.patch.stopall)
        with tempfile.TemporaryDirectory() as tmp:
            vid = asyncio.run(adapter.design_voice(
                "p", "t", os.path.join(tmp, "trial.mp3")))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(vid, "ttv-voice-20260701-abcdef")

    def test_design_voice_missing_audio_raises(self):
        adapter = self._adapter(max_retries=1)
        _patch_transport([_FakeResponse(200, _design_body(with_audio=False))])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "trial_audio"):
            asyncio.run(adapter.design_voice("p", "t", "o.mp3"))

    def _write_sample(self, d, name="sample.wav"):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 512)  # 内容不解析，时长走 mock
        return p

    def test_upload_voice_file_success(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _upload_body())])
        self.addCleanup(mock.patch.stopall)
        with mock.patch("modules.tts.minimax_tts_adapter.get_audio_duration",
                        return_value=30.0):
            with tempfile.TemporaryDirectory() as tmp:
                p = self._write_sample(tmp)
                fid = asyncio.run(adapter.upload_voice_file(p))
        self.assertEqual(fid, 12345)
        call = fake.calls[0]
        self.assertTrue(call["url"].endswith("/v1/files/upload"))
        self.assertEqual(call["data"]["purpose"], "voice_clone")
        self.assertIn("sample.wav", str(call["files"]["file"]))

    def test_upload_voice_file_rejects_short_audio(self):
        adapter = self._adapter(max_retries=1)
        with mock.patch("modules.tts.minimax_tts_adapter.get_audio_duration",
                        return_value=5.0):
            with tempfile.TemporaryDirectory() as tmp:
                p = self._write_sample(tmp)
                with self.assertRaisesRegex(TTSError, "10s"):
                    asyncio.run(adapter.upload_voice_file(p))

    def test_upload_voice_file_rejects_bad_extension(self):
        adapter = self._adapter(max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "sample.txt")
            with open(p, "wb") as f:
                f.write(b"x")
            with self.assertRaisesRegex(TTSError, "mp3/m4a/wav"):
                asyncio.run(adapter.upload_voice_file(p))

    def test_clone_voice_success(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _clone_body())])
        self.addCleanup(mock.patch.stopall)
        vid = asyncio.run(adapter.clone_voice(12345, voice_id="CloneVoice123456789"))
        self.assertEqual(vid, "CloneVoice123456789")
        call = fake.calls[0]
        self.assertTrue(call["url"].endswith("/v1/voice_clone"))
        self.assertEqual(call["json"]["file_id"], 12345)
        self.assertEqual(call["json"]["voice_id"], "CloneVoice123456789")
        self.assertNotIn("language_boost", call["json"])

    def test_clone_voice_forwards_official_optional_fields(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _clone_body())])
        self.addCleanup(mock.patch.stopall)
        asyncio.run(adapter.clone_voice(
            12345,
            voice_id="CloneVoice123456789",
            prompt_audio_id=67890,
            prompt_text="这是增强示例。",
            text_validation="这是主参考音频。",
            accuracy=0.85,
            need_noise_reduction=True,
            need_volume_normalization=True,
        ))
        body = fake.calls[0]["json"]
        self.assertEqual(body["clone_prompt"], {
            "prompt_audio": 67890,
            "prompt_text": "这是增强示例。",
        })
        self.assertEqual(body["text_validation"], "这是主参考音频。")
        self.assertEqual(body["accuracy"], 0.85)
        self.assertIs(body["need_noise_reduction"], True)
        self.assertIs(body["need_volume_normalization"], True)

    def test_clone_prompt_requires_audio_and_text_together(self):
        adapter = self._adapter()
        with self.assertRaisesRegex(TTSError, "同时提供"):
            asyncio.run(adapter.clone_voice(
                12345,
                voice_id="CloneVoice123456789",
                prompt_audio_id=67890,
            ))

    def test_clone_accuracy_rejects_out_of_range(self):
        adapter = self._adapter()
        with self.assertRaisesRegex(TTSError, "0–1"):
            asyncio.run(adapter.clone_voice(
                12345,
                voice_id="CloneVoice123456789",
                text_validation="主参考文本",
                accuracy=1.1,
            ))

    def test_clone_voice_auto_generates_id(self):
        adapter = self._adapter()
        _patch_transport([_FakeResponse(200, _clone_body())])
        self.addCleanup(mock.patch.stopall)
        vid = asyncio.run(adapter.clone_voice(12345, voice_id=None))
        self.assertTrue(vid.startswith("CloneVoice"))

    def test_clone_voice_no_permission_2038(self):
        adapter = self._adapter(max_retries=1)
        _patch_transport([_FakeResponse(
            200, _clone_body(status_code=2038, status_msg="no permission"))])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "认证"):
            asyncio.run(adapter.clone_voice(12345))

    def test_clone_voice_text_requires_model(self):
        adapter = self._adapter(max_retries=1)
        with self.assertRaisesRegex(TTSError, "model"):
            asyncio.run(adapter.clone_voice(12345, text="你好"))

    def test_clone_voice_transient_retries(self):
        adapter = self._adapter()
        transient = _FakeResponse(200, _clone_body(status_code=1002, status_msg="rpm"))
        ok = _FakeResponse(200, _clone_body())
        fake = _patch_transport([transient, ok])
        self.addCleanup(mock.patch.stopall)
        vid = asyncio.run(adapter.clone_voice(12345, voice_id="CloneVoice123456789"))
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(vid, "CloneVoice123456789")

    def test_clone_voice_downloads_demo(self):
        adapter = self._adapter()
        _patch_transport([_FakeResponse(200, _clone_body())])
        self.addCleanup(mock.patch.stopall)
        downloaded = {}
        async def _fake_download(url, output_path):
            downloaded["url"] = url
            downloaded["path"] = output_path
            with open(output_path, "wb") as f:
                f.write(b"ID3" + b"\x00" * 300)
        with mock.patch.object(adapter, "_download_demo_audio", new=_fake_download):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "demo.mp3")
                vid = asyncio.run(adapter.clone_voice(
                    12345, voice_id="CloneVoice123456789", output_path=out))
        self.assertEqual(vid, "CloneVoice123456789")
        self.assertEqual(downloaded["url"], "https://cdn.minimaxi.com/demo.mp3")
        self.assertEqual(downloaded["path"], out)

    def test_get_voices_returns_only_user_voices(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(200, _voices_body())])
        self.addCleanup(mock.patch.stopall)
        voices = asyncio.run(adapter.get_voices())
        # 丢弃 system_voice，只保留用户克隆/设计的音色
        ids = [v["voice_id"] for v in voices]
        self.assertEqual(ids, ["MyClone01", "MyDesign01"])
        self.assertNotIn("Chinese (Mandarin)_Warm_Girl", ids)
        by_id = {v["voice_id"]: v for v in voices}
        self.assertEqual(by_id["MyClone01"]["voice_type"], "voice_cloning")
        self.assertEqual(by_id["MyDesign01"]["voice_type"], "voice_generation")
        self.assertTrue(fake.calls[0]["url"].endswith("/v1/get_voice"))
        self.assertEqual(fake.calls[0]["json"]["voice_type"], "all")

    def test_get_voices_drops_known_system_ids(self):
        adapter = self._adapter()
        body = _voices_body()
        body["voice_cloning"] = [
            {"voice_id": "MyClone01"},
            {"voice_id": "Chinese (Mandarin)_Warm_Girl"},  # 系统音色混入
        ]
        _patch_transport([_FakeResponse(200, body)])
        self.addCleanup(mock.patch.stopall)
        voices = asyncio.run(adapter.get_voices())
        ids = [v["voice_id"] for v in voices]
        self.assertEqual(ids, ["MyClone01", "MyDesign01"])

    def test_delete_voice_success(self):
        adapter = self._adapter()
        fake = _patch_transport([_FakeResponse(
            200, {"voice_id": "MyClone01", "created_time": "2026-08-01",
                  "base_resp": {"status_code": 0, "status_msg": "success"}})])
        self.addCleanup(mock.patch.stopall)
        asyncio.run(adapter.delete_voice("MyClone01", "voice_cloning"))
        call = fake.calls[0]
        self.assertTrue(call["url"].endswith("/v1/delete_voice"))
        self.assertEqual(call["json"]["voice_id"], "MyClone01")
        self.assertEqual(call["json"]["voice_type"], "voice_cloning")

    def test_delete_voice_rejects_bad_type(self):
        adapter = self._adapter(max_retries=1)
        with self.assertRaisesRegex(TTSError, "voice_type"):
            asyncio.run(adapter.delete_voice("x", "voice_system"))

    def test_design_voice_malformed_hex_raises(self):
        adapter = self._adapter(max_retries=1)
        body = _design_body()
        body["trial_audio"] = "not-hex!!"
        _patch_transport([_FakeResponse(200, body)])
        self.addCleanup(mock.patch.stopall)
        with self.assertRaisesRegex(TTSError, "malformed hex"):
            asyncio.run(adapter.design_voice("p", "t", "o.mp3"))

    def test_requests_are_spaced_by_min_interval(self):
        from modules.tts import minimax_tts_adapter as mod
        self._reset_throttle()
        adapter = MiniMaxTTSAdapter(api_key="k", min_interval_sec=0.15)
        fake = _patch_transport([_FakeResponse(200, _ok_body()),
                                 _FakeResponse(200, _ok_body())])
        self.addCleanup(mock.patch.stopall)
        import time as _t
        t0 = _t.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(adapter.synthesize("a", "v1", os.path.join(tmp, "a.mp3")))
            asyncio.run(adapter.synthesize("b", "v1", os.path.join(tmp, "b.mp3")))
        elapsed = _t.monotonic() - t0
        self.assertEqual(len(fake.calls), 2)
        self.assertGreaterEqual(elapsed, 0.15)

    def test_throttle_is_global_across_instances(self):
        # Two separate adapters share the module-level gate (cross-thread safe).
        from modules.tts import minimax_tts_adapter as mod
        self._reset_throttle()
        adapter_a = MiniMaxTTSAdapter(api_key="k", min_interval_sec=0.15)
        adapter_b = MiniMaxTTSAdapter(api_key="k", min_interval_sec=0.15)
        fake = _patch_transport([_FakeResponse(200, _ok_body()),
                                 _FakeResponse(200, _ok_body())])
        self.addCleanup(mock.patch.stopall)
        import time as _t
        t0 = _t.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(adapter_a.synthesize("a", "v1", os.path.join(tmp, "a.mp3")))
            asyncio.run(adapter_b.synthesize("b", "v1", os.path.join(tmp, "b.mp3")))
        self.assertEqual(len(fake.calls), 2)
        self.assertGreaterEqual(_t.monotonic() - t0, 0.15)


if __name__ == "__main__":
    unittest.main()
