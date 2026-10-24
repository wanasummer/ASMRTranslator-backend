"""Offline tests for modules.tts.edge_tts_adapter (network fully mocked).

Run from the project root:  python -m unittest tests.test_edge_tts_adapter
Covers the activated features: live voice catalog + zh-first sort, fallback to the
curated zh list when offline, stream-based synthesis that writes audio AND a
WordBoundary sidecar, and pitch/boundary forwarding to edge_tts.Communicate.
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest import mock

import aiohttp
import edge_tts.communicate as edge_communicate
from modules.error import TTSError
from modules.network import NetworkPolicy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.tts.edge_tts_adapter import EdgeTTSAdapter

SAMPLE_VOICES = [
    {"ShortName": "zh-CN-XiaoxiaoNeural", "Locale": "zh-CN", "Gender": "Female",
     "FriendlyName": "Microsoft Xiaoxiao Online (Natural) - Chinese (Mainland)"},
    {"ShortName": "zh-CN-XiaoyiNeural", "Locale": "zh-CN", "Gender": "Female",
     "FriendlyName": "Microsoft Xiaoyi Online (Natural) - Chinese (Mainland)"},
    {"ShortName": "zh-CN-YunxiNeural", "Locale": "zh-CN", "Gender": "Male",
     "FriendlyName": "Microsoft Yunxi Online (Natural) - Chinese (Mainland)"},
    {"ShortName": "zh-CN-XiaozhenNeural", "Locale": "zh-CN", "Gender": "Female",
     "FriendlyName": "Microsoft Xiaozhen Online (Natural) - Chinese (Mainland)"},
    {"ShortName": "en-US-JennyNeural", "Locale": "en-US", "Gender": "Female",
     "FriendlyName": "Microsoft Jenny Online (Natural) - English (United States)"},
]


def _fake_list_voices(result):
    async def _f(*a, **k):
        return result
    return _f


def _mk_fake_communicate(chunks, calls):
    """Patch factory: a Communicate whose stream() yields `chunks`, recording args."""
    class _Fake:
        def __init__(self, text, voice, **kw):
            calls.append(dict(text=text, voice=voice, **kw))

        async def stream(self):
            for c in chunks:
                yield c
    return _Fake


class VoiceCatalogTest(unittest.TestCase):
    def test_live_catalog_maps_fields_and_keeps_only_zh(self):
        adapter = EdgeTTSAdapter()
        with mock.patch("edge_tts.list_voices", new=_fake_list_voices(SAMPLE_VOICES)):
            voices = adapter.get_available_voices()
        by_id = {v.id: v for v in voices}
        self.assertEqual(by_id["zh-CN-XiaoxiaoNeural"].gender, "female")
        self.assertEqual(by_id["zh-CN-XiaoxiaoNeural"].language, "zh-CN")
        # 中文名优先（晓晓），英文名仅兜底
        self.assertEqual(by_id["zh-CN-XiaoxiaoNeural"].name, "晓晓 (Xiaoxiao)")
        # 白名单过滤：只保留 ASMR 适配的中文普通话女声
        self.assertNotIn("en-US-JennyNeural", by_id)       # 非中文
        self.assertNotIn("zh-CN-YunxiNeural", by_id)       # 男声
        self.assertNotIn("zh-CN-XiaozhenNeural", by_id)    # 成熟女声（不在白名单）
        self.assertEqual(set(by_id), {"zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"})
        # zh-CN voices sorted by id (stable, picker-friendly)
        self.assertEqual([v.id for v in voices],
                         ["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural"])

    def test_live_catalog_cached(self):
        adapter = EdgeTTSAdapter()
        with mock.patch("edge_tts.list_voices",
                        side_effect=_fake_list_voices(SAMPLE_VOICES)) as m:
            adapter.get_available_voices()
            adapter.get_available_voices()
            self.assertEqual(m.call_count, 1)

    def test_fallback_when_network_fails(self):
        adapter = EdgeTTSAdapter()
        with mock.patch("edge_tts.list_voices", side_effect=OSError("no net")):
            voices = adapter.get_available_voices()
        self.assertTrue(voices)
        self.assertTrue(all(v.language == "zh-CN" for v in voices))


class SynthesizeTest(unittest.TestCase):
    def test_edge_access_denied_is_not_an_api_key_error(self):
        error = aiohttp.WSServerHandshakeError(
            SimpleNamespace(real_url="wss://api.msedgeservices.com/test"), (),
            status=403, message="Invalid response status",
        )
        adapter = EdgeTTSAdapter(max_retries=1, default_fallback=False)
        with mock.patch.object(adapter, "_save_routed", new=mock.AsyncMock(side_effect=error)):
            with self.assertRaises(TTSError) as caught:
                asyncio.run(adapter.synthesize("测试", "zh-CN-XiaoxiaoNeural", "unused.mp3"))
        self.assertEqual(caught.exception.descriptor.code, "TTS_ACCESS_DENIED")
        self.assertNotIn("API Key", caught.exception.descriptor.user_message)
        self.assertEqual(caught.exception.context["status"], 403)

    def _stream_chunks(self):
        return [
            {"type": "WordBoundary", "offset": 0, "duration": 45000000, "text": "你好"},
            {"type": "audio", "data": b"\xff\xf3" * 100},
            {"type": "WordBoundary", "offset": 45000000, "duration": 45000000, "text": "世界"},
        ]

    def test_stream_writes_audio_and_boundary_sidecar(self):
        calls = []
        fake_cls = _mk_fake_communicate(self._stream_chunks(), calls)
        adapter = EdgeTTSAdapter()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "tts.mp3")
            with mock.patch("modules.tts.edge_tts_adapter._MicrosoftEdgeCommunicate", new=fake_cls):
                asyncio.run(adapter.synthesize("你好世界", "zh-CN-XiaoxiaoNeural", out))

            self.assertGreater(os.path.getsize(out), 100)
            side = out + ".boundaries.json"
            self.assertTrue(os.path.exists(side))
            with open(side, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["unit"], "ms")
            self.assertEqual([w["text"] for w in data["words"]], ["你好", "世界"])
            # 45000000 ticks (100 ns) -> 4500 ms
            self.assertEqual(data["words"][1]["offset_ms"], 4500.0)
            self.assertEqual(data["words"][1]["duration_ms"], 4500.0)

        # WordBoundary + pitch are forwarded to Communicate
        self.assertEqual(calls[0]["voice"], "zh-CN-XiaoxiaoNeural")
        self.assertEqual(calls[0]["pitch"], "+0Hz")
        self.assertEqual(calls[0]["boundary"], "WordBoundary")

    def test_pitch_forwarded(self):
        calls = []
        fake_cls = _mk_fake_communicate(
            [{"type": "audio", "data": b"\xff" * 200}], calls)
        adapter = EdgeTTSAdapter()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.mp3")
            with mock.patch("modules.tts.edge_tts_adapter._MicrosoftEdgeCommunicate", new=fake_cls):
                asyncio.run(adapter.synthesize("hi", "zh-CN-XiaoxiaoNeural", out, pitch="+20Hz"))
        self.assertEqual(calls[0]["pitch"], "+20Hz")

    def test_no_audio_raises_real_error(self):
        # empty/whitespace text: the stream ends with no audio, so the adapter must
        # raise a real NoAudioReceived-backed error, not "failed after retries: None"
        calls = []
        fake_cls = _mk_fake_communicate([], calls)  # streams nothing

        async def _no_sleep(*a, **k):
            pass

        adapter = EdgeTTSAdapter()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.mp3")
            with mock.patch("modules.tts.edge_tts_adapter._MicrosoftEdgeCommunicate", new=fake_cls):
                with mock.patch("asyncio.sleep", new=_no_sleep):
                    with self.assertRaises(Exception) as ctx:
                        asyncio.run(adapter.synthesize("", "zh-CN-XiaoxiaoNeural", out))
        self.assertNotIn("None", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_no_boundaries_no_sidecar(self):
        calls = []
        fake_cls = _mk_fake_communicate(
            [{"type": "audio", "data": b"\xff" * 200}], calls)
        adapter = EdgeTTSAdapter()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.mp3")
            with mock.patch("modules.tts.edge_tts_adapter._MicrosoftEdgeCommunicate", new=fake_cls):
                asyncio.run(adapter.synthesize("hi", "zh-CN-XiaoxiaoNeural", out))
            self.assertFalse(os.path.exists(out + ".boundaries.json"))

    def test_fallback_to_default_voice_on_error(self):
        calls = []
        chunks = [{"type": "audio", "data": b"\xff" * 200}]
        state = {"n": 0}

        def _flaky_factory(text, voice, **kw):
            calls.append(dict(text=text, voice=voice, **kw))
            state["n"] += 1
            # first 3 constructions fail (the retries); the fallback succeeds
            if state["n"] <= 3:
                raise RuntimeError("connect failed")
            return _mk_fake_communicate(chunks, [])(text, voice, **kw)

        async def _no_sleep(*a, **k):
            pass

        adapter = EdgeTTSAdapter(voice_map={"default": "zh-CN-YunxiNeural"})
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.mp3")
            with mock.patch("modules.tts.edge_tts_adapter._MicrosoftEdgeCommunicate", new=_flaky_factory):
                with mock.patch("asyncio.sleep", new=_no_sleep):
                    asyncio.run(
                        adapter.synthesize("hi", "zh-CN-XiaoxiaoNeural", out)
                    )
            self.assertGreater(os.path.getsize(out), 100)
            # the fallback used the default voice
            self.assertEqual(calls[-1]["voice"], "zh-CN-YunxiNeural")


class MicrosoftTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_endpoint_stream_preserves_boundaries_chunks_and_connector_ownership(self):
        # Exercise the real library chunk loop and our transport with mocked I/O.
        urls, session_options, sockets, connectors = [], [], [], []
        audio = b"\xff\xf3" * 100
        header = b"X-RequestId:test\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n"
        packet = len(header).to_bytes(2, "big") + header + audio
        metadata = {"Metadata": [{"Type": "WordBoundary", "Data": {
            "Offset": 0, "Duration": 1000000, "text": {"Text": "你好", "Length": 2},
        }}]}
        messages = [
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Path:audio.metadata\r\n\r\n" + json.dumps(metadata)),
            SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=packet),
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Path:turn.end\r\n\r\n"),
        ]

        class Socket:
            def __init__(self):
                self.sent = []
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def send_str(self, value):
                self.sent.append(value)
            async def __aiter__(self):
                for message in messages:
                    yield message

        class Session:
            def __init__(self, **kwargs):
                session_options.append(kwargs)
                connectors.append(kwargs["connector"])
                self.connector = kwargs["connector"]
                self.owner = kwargs["connector_owner"]
            async def __aenter__(self):
                if self.connector.closed:
                    raise RuntimeError("Session is closed")
                return self
            async def __aexit__(self, *args):
                if self.owner:
                    await self.connector.close()
            def ws_connect(self, url, **kwargs):
                urls.append((url, kwargs))
                if len(urls) == 1:
                    failed = mock.MagicMock()
                    failed.__aenter__ = mock.AsyncMock(side_effect=aiohttp.WSServerHandshakeError(
                        SimpleNamespace(real_url="wss://api.msedgeservices.com/test"), (),
                        status=403, message="Invalid response status",
                    ))
                    return failed
                socket = Socket()
                sockets.append(socket)
                return socket

        adapter = EdgeTTSAdapter(max_retries=1, default_fallback=False)
        adapter._network_policy = NetworkPolicy(proxy_provider=lambda: {})
        original_url = edge_communicate.WSS_URL
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "probe.mp3")
            # >4096 bytes forces two sessions using the same externally owned connector.
            with mock.patch("modules.tts.edge_tts_adapter.aiohttp.ClientSession", Session), \
                    mock.patch("edge_tts.communicate.DRM.handle_client_response_error") as correction:
                result = await adapter.synthesize("你好 " * 700, "zh-CN-XiaoxiaoNeural", output)
            correction.assert_called_once()
            self.assertEqual(result, output)
            with open(output, "rb") as stream:
                self.assertEqual(stream.read(), audio * 2)
            with open(output + ".boundaries.json", encoding="utf-8") as stream:
                words = json.load(stream)["words"]
            self.assertEqual([word["text"] for word in words], ["你好", "你好"])
            self.assertGreater(words[1]["offset_ms"], words[0]["offset_ms"])
        self.assertEqual(len(urls), 3)
        for url, kwargs in urls:
            parsed = urlsplit(url)
            self.assertEqual(parsed.hostname, "api.msedgeservices.com")
            self.assertEqual(parsed.path, "/tts/cognitiveservices/websocket/v1")
            query = parse_qs(parsed.query)
            self.assertIn("Ocp-Apim-Subscription-Key", query)
            self.assertNotIn("TrustedClientToken", query)
            self.assertIn("Sec-MS-GEC", query)
            self.assertIn("Sec-MS-GEC-Version", query)
            self.assertEqual(kwargs["proxy"], "")
        self.assertTrue(all(not options["connector_owner"] for options in session_options))
        self.assertIs(connectors[0], connectors[1])
        self.assertIs(connectors[1], connectors[2])
        self.assertTrue(connectors[0].closed)
        self.assertIn("https://api.msedgeservices.com", adapter._network_policy._successes)
        self.assertEqual(edge_communicate.WSS_URL, original_url)
        for socket in sockets:
            self.assertIn('"wordBoundaryEnabled": "true"', socket.sent[0])
            self.assertIn("audio-24khz-48kbitrate-mono-mp3", socket.sent[0])
            self.assertIn("zh-CN, XiaoxiaoNeural", socket.sent[1])


if __name__ == "__main__":
    unittest.main()
