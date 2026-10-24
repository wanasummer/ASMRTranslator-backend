"""Edge TTS adapter — Microsoft Edge online TTS (free, no API key).

Activated features (vs the earlier MVP that only passed `rate`):
  * Chinese voice catalog: ``get_available_voices`` queries the live
    ``edge_tts.list_voices()`` endpoint and keeps only the zh-* voices
    (14 total — Mainland/Cantonese/Taiwan/dialects), falling back to the
    curated zh list when offline.
  * Pitch: ``pitch`` is forwarded to ``edge_tts.Communicate``.
  * Word boundaries: synthesis streams via ``Communicate.stream()`` and captures
    WordBoundary events into a ``<output_path>.boundaries.json`` sidecar (offsets
    in ms) for per-word timing.
"""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Optional
import aiohttp
import edge_tts
from edge_tts.communicate import (
    _SSL_CTX, connect_id, date_to_string, get_headers_and_data,
    mkssml, ssml_headers_plus_data,
)
from edge_tts.constants import SEC_MS_GEC_VERSION, TRUSTED_CLIENT_TOKEN, WSS_HEADERS
from edge_tts.drm import DRM
from edge_tts.typing import TTSChunk

from ..network import NetworkPolicy, async_operation
from edge_tts.exceptions import NoAudioReceived, UnexpectedResponse, UnknownResponse, WebSocketError

from ..error import TTSError, descriptor
from ..io import open_write
from .base import TTSAdapter
from .models import VoiceInfo


_SYNTHESIS_ORIGIN = "https://api.msedgeservices.com"
_SYNTHESIS_URL = (
    "wss://api.msedgeservices.com/tts/cognitiveservices/websocket/v1"
    f"?Ocp-Apim-Subscription-Key={TRUSTED_CLIENT_TOKEN}"
)


class _MicrosoftEdgeCommunicate(edge_tts.Communicate):
    """Compatibility bridge for edge-tts 7.2.8's retired synthesis endpoint.

    Keep the library's escaping, chunking, metadata parser, CBR timing and clock
    correction. Only replace its per-chunk transport; no library globals change.
    The connector belongs to modules.network and must survive internal retries
    and subsequent chunks. Remove this bridge when upstream ships the migration.
    """

    async def _Communicate__stream(self) -> AsyncIterator[TTSChunk]:
        audio_received = False
        async with aiohttp.ClientSession(
            connector=self.connector, connector_owner=self.connector is None,
            trust_env=False, timeout=self.session_timeout,
        ) as session, session.ws_connect(
            f"{_SYNTHESIS_URL}&ConnectionId={connect_id()}"
            f"&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
            compress=15, proxy=self.proxy,
            headers=DRM.headers_with_muid(WSS_HEADERS), ssl=_SSL_CTX,
        ) as websocket:
            word_boundary = self.tts_config.boundary == "WordBoundary"
            metadata_options = {
                "sentenceBoundaryEnabled": "false" if word_boundary else "true",
                "wordBoundaryEnabled": "true" if word_boundary else "false",
            }
            config = {"context": {"synthesis": {"audio": {
                "metadataoptions": metadata_options,
                "outputFormat": "audio-24khz-48kbitrate-mono-mp3",
            }}}}
            await websocket.send_str(
                f"X-Timestamp:{date_to_string()}\r\n"
                "Content-Type:application/json; charset=utf-8\r\n"
                "Path:speech.config\r\n\r\n"
                f"{json.dumps(config)}\r\n"
            )
            await websocket.send_str(ssml_headers_plus_data(
                connect_id(), date_to_string(),
                mkssml(self.tts_config, self.state["partial_text"]),
            ))
            async for received in websocket:
                if received.type == aiohttp.WSMsgType.TEXT:
                    encoded = received.data.encode("utf-8")
                    headers, data = get_headers_and_data(encoded, encoded.find(b"\r\n\r\n"))
                    path = headers.get(b"Path")
                    if path == b"audio.metadata":
                        metadata = self._Communicate__parse_metadata(data)
                        yield metadata
                        self.state["last_duration_offset"] = metadata["offset"] + metadata["duration"]
                    elif path == b"turn.end":
                        self._Communicate__compensate_offset()
                        break
                    elif path not in (b"response", b"turn.start"):
                        raise UnknownResponse("Unknown path received")
                elif received.type == aiohttp.WSMsgType.BINARY:
                    if len(received.data) < 2:
                        raise UnexpectedResponse("Audio message is missing the header length")
                    header_length = int.from_bytes(received.data[:2], "big")
                    if header_length > len(received.data) - 2:
                        raise UnexpectedResponse("Audio header length exceeds the message length")
                    headers, data = get_headers_and_data(received.data, header_length)
                    if headers.get(b"Path") != b"audio":
                        raise UnexpectedResponse("Binary message path is not audio")
                    content_type = headers.get(b"Content-Type")
                    if content_type is None and not data:
                        continue
                    if content_type != b"audio/mpeg" or not data:
                        raise UnexpectedResponse("Audio message has invalid content type or empty data")
                    audio_received = True
                    self.state["chunk_audio_bytes"] += len(data)
                    yield {"type": "audio", "data": data}
                elif received.type == aiohttp.WSMsgType.ERROR:
                    raise WebSocketError(received.data or "Unknown error")
            if not audio_received:
                raise NoAudioReceived("No audio was received for the given text.")


class EdgeTTSAdapter(TTSAdapter):
    """Microsoft Edge TTS backend.

    Voices available: https://speech.microsoft.com/portal/voicegallery
    Chinese voices: zh-CN-XiaoxiaoNeural, zh-CN-XiaoyiNeural, etc.
    """

    # Known working Chinese voices — fallback when the live catalog is unreachable.
    # ASMR 适配的中文（普通话）女声白名单 —— 温柔/甜美/清新系。
    # 排除：男声、女童、成熟/知性女声、zh-TW/zh-HK 口音系。
    # 与 MiniMax 后端口径一致（只保留适配 ASMR 的音声）。
    _ASMR_ZH_CN_FEMALE = (
        "zh-CN-XiaoxiaoNeural",  # 晓晓 — 温暖女声
        "zh-CN-XiaoyiNeural",    # 晓伊 — 活泼甜美
        "zh-CN-XiaohanNeural",   # 晓涵 — 温柔女声(可能不可用)
        "zh-CN-XiaochenNeural",  # 晓辰 — 清新女声
        "zh-CN-XiaoyouNeural",   # 晓悠 — 温柔舒缓
        "zh-CN-XiaomengNeural",  # 晓梦 — 活力甜美
        "zh-CN-XiaoxuanNeural",  # 晓萱 — 活泼甜美
    )
    _KNOWN_VOICES = {
        "zh-CN-XiaoxiaoNeural":  "晓晓 (Xiaoxiao) — 温暖女声",
        "zh-CN-XiaoyiNeural":    "晓伊 (Xiaoyi) — 活泼甜美女声",
        "zh-CN-XiaohanNeural":   "晓涵 (Xiaohan) — 温柔女声(可能不可用)",
        "zh-CN-XiaochenNeural":  "晓辰 (Xiaochen) — 清新女声",
        "zh-CN-XiaoyouNeural":   "晓悠 (Xiaoyou) — 温柔舒缓女声",
        "zh-CN-XiaomengNeural":  "晓梦 (Xiaomeng) — 活力甜美女声",
        "zh-CN-XiaoxuanNeural":  "晓萱 (Xiaoxuan) — 活泼甜美女声",
    }

    def __init__(
        self,
        voice_map: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        default_fallback: bool = True,
        boundary: str = "WordBoundary",
    ):
        super().__init__(voice_map)
        self._max_retries = max_retries
        self._default_fallback = default_fallback
        self._boundary = boundary
        self._network_policy = NetworkPolicy()
        self._live_voices_cache: Optional[list[VoiceInfo]] = None

    @property
    def name(self) -> str:
        return "edge-tts"

    @staticmethod
    async def _save_with_boundaries(communicate, output_path: str):
        """Stream audio to ``output_path`` while capturing WordBoundary events.

        ``Communicate.stream()`` yields audio chunks and metadata events. Audio is
        written verbatim (same bytes ``save()`` would write); WordBoundary offsets
        are converted from 100 ns ticks to ms and stored in a JSON sidecar.

        Raises ``NoAudioReceived`` if the stream ends without any audio bytes
        (e.g. empty/whitespace text), so callers see a real error instead of a
        silent zero-byte file.
        """
        boundaries = []
        audio_bytes = 0
        with open_write(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                    audio_bytes += len(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    boundaries.append({
                        "text": chunk.get("text", ""),
                        "offset_ms": chunk.get("offset", 0) / 10000.0,
                        "duration_ms": chunk.get("duration", 0) / 10000.0,
                    })
        if audio_bytes <= 0:
            raise NoAudioReceived("No audio was received for the given text.")
        if boundaries:
            sidecar = output_path + ".boundaries.json"
            with open_write(sidecar, "w", encoding="utf-8") as f:
                json.dump({"type": "word", "unit": "ms", "words": boundaries},
                          f, ensure_ascii=False)

    async def _save_routed(self, text: str, voice: str, output_path: str, rate: str, pitch: str) -> None:
        async def save(connection):
            communicate = _MicrosoftEdgeCommunicate(
                text, voice, rate=rate, pitch=pitch, boundary=self._boundary,
                **connection.request_options, connect_timeout=8,
            )
            await self._save_with_boundaries(communicate, output_path)

        await async_operation(
            _SYNTHESIS_ORIGIN, save,
            network_policy=self._network_policy,
            retry_allowed=lambda: not os.path.exists(output_path) or os.path.getsize(output_path) == 0,
        )

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synthesize with retry + fallback to default voice."""
        last_error = None

        for attempt in range(self._max_retries):
            try:
                await self._save_routed(text, voice, output_path, rate, pitch)
                if os.path.getsize(output_path) > 100:
                    return output_path
            except Exception as e:
                last_error = e
                if isinstance(e, TTSError) and not e.descriptor.retryable:
                    raise
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(f"       [retry {attempt+1}] '{text[:20]}...' -> wait {wait}s ({type(e).__name__})")
                    await asyncio.sleep(wait)

        # Fallback to default voice
        default_voice = self._voice_map.get("default", "")
        if self._default_fallback and voice != default_voice:
            print(f"       [fallback] default voice for: '{text[:30]}...'")
            try:
                await self._save_routed(text, default_voice, output_path, rate, pitch)
                if os.path.getsize(output_path) > 100:
                    return output_path
            except Exception as e:
                last_error = e

        if isinstance(last_error, aiohttp.ClientResponseError) and last_error.status in (401, 403):
            raise TTSError(
                descriptor("TTS_ACCESS_DENIED"),
                detail=f"EdgeTTS failed after retries: {last_error}",
                cause=last_error, context={"backend": self.name, "status": last_error.status},
            ) from last_error
        raise TTSError(f"EdgeTTS failed after retries: {last_error}") from last_error

    def get_available_voices(self) -> list[VoiceInfo]:
        """Return the live Edge Chinese (zh-*) voice catalog.

        The app dubs to Chinese, so only zh-* voices are offered. Falls back to
        the curated zh-CN list when the network query fails.
        """
        if self._live_voices_cache is not None:
            return self._live_voices_cache
        try:
            raw = asyncio.run(async_operation(
                "https://speech.platform.bing.com",
                lambda connection: edge_tts.list_voices(**connection.request_options),
                network_policy=self._network_policy,
            ))
            voices = []
            for v in raw:
                locale = v.get("Locale", "")
                short = v.get("ShortName", "")
                if short not in self._ASMR_ZH_CN_FEMALE:
                    continue  # 只保留 ASMR 适配的中文普通话女声
                friendly = v.get("FriendlyName", "")
                # 中文名优先（晓晓 / 晓伊 ...），英文名兜底
                name = (self._KNOWN_VOICES.get(short, "").split(" — ")[0]
                        or friendly.split(" - ")[0] or short)
                gender = (v.get("Gender") or "").lower() or None
                voices.append(VoiceInfo(
                    id=short, name=name, gender=gender,
                    language=locale, description=friendly or short,
                ))
            voices.sort(key=lambda vi: (vi.language or "", vi.id))
            self._live_voices_cache = voices
            return voices
        except Exception as e:
            # stderr, not stdout — the Electron picker parses stdout as JSON
            print(f"       [warn] Edge voice catalog fetch failed ({e}); using curated zh list",
                  file=sys.stderr)
            return self._fallback_voices()

    def _fallback_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=vid, name=desc.split("—")[0].strip(),
                      description=desc, gender="female" if "女" in desc else "male",
                      language="zh-CN")
            for vid, desc in self._KNOWN_VOICES.items()
        ]
