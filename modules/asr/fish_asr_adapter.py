"""Fish Audio ASR adapter — multipart/form-data speech-to-text.

``POST https://api.fish.audio/v1/asr`` with ``Authorization: Bearer <key>`` and a
multipart body carrying the audio file plus ``language`` and ``ignore_timestamps``
fields. The API accepts multipart/form-data only (no base64/JSON payloads).

``ignore_timestamps=false`` is what makes the API return per-fragment time slices
(``segments``); with ``true`` (the API default) the ``segments`` list comes back
empty, so ``false`` is sent explicitly from config to keep timestamps on.

Response shape::

    {"text", "duration",
     "segments": [{"text", "start", "end"}],
     "language_code", "language"}

Retry policy: 429 (concurrent-request limit), 5xx, and transport errors retry
with exponential backoff + jitter (``max_retries``); other 4xx fail fast.
"""

import asyncio
import os
import random
from pathlib import Path
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import ASRError
from .base import ASRAdapter
from .models import ASRSegment, ASRResult
from .cloud_audio import wav_duration
from .no_speech import response_has_no_speech, is_no_speech

_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".amr": "audio/amr",
}


class FishAudioASRAdapter(ASRAdapter):
    """Fish Audio ASR backend.

    Transcribes one audio clip into timestamped Japanese segments. The audio is
    posted as multipart form data (read into memory up front, so a shared
    ``httpx.AsyncClient`` can be reused across a batch).
    """

    _DEFAULT_ENDPOINT = "https://api.fish.audio/v1/asr"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "ja",
        ignore_timestamps: bool = False,
        base_url: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        self._api_key = api_key or os.environ.get("FISH_API_KEY", "")
        self._language = language
        self._ignore_timestamps = ignore_timestamps
        self._endpoint = (base_url or self._DEFAULT_ENDPOINT).rstrip("/")
        self._timeout = timeout
        self._network_policy = NetworkPolicy()
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "fish-asr"

    @property
    def timeout(self) -> float:
        """Per-request timeout, for callers that want to build a shared client."""
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        """Transcribe one audio clip into an ``ASRResult``.

        ``language`` overrides the configured source language per call;
        ``timestamps=False`` drops the parsed segments (the multipart still sends
        ``ignore_timestamps=false`` so the API keeps returning the slices).
        """
        if not self._api_key:
            raise ASRError("未配置 Fish Audio ASR API key（api_key 为空）")
        lang = language or self._language
        result = await self._post(audio_path, lang, self._ignore_timestamps, client=client)
        if not timestamps:
            result.segments = []
        return result

    # ---- transport / parsing ----

    @staticmethod
    def _mime_for(path: str) -> str:
        return _MIME_BY_EXT.get(Path(path).suffix.lower(), "audio/wav")

    async def _post(
        self,
        audio_path: str,
        language: str,
        ignore_timestamps: bool,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        files = {"audio": (Path(audio_path).name, audio_bytes, self._mime_for(audio_path))}
        form = {
            "language": language,
            "ignore_timestamps": "true" if ignore_timestamps else "false",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if client is not None:
                    response = await client.post(
                        self._endpoint, headers=headers, data=form, files=files
                    )
                else:
                    async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self._timeout) as ac:
                        response = await ac.post(
                            self._endpoint, headers=headers, data=form, files=files
                        )
                if (response.status_code == 429 or response.status_code >= 500) \
                        and attempt < self._max_retries:
                    # Transient concurrency/rate limit (429) or server error (5xx)
                    # — retry with exponential backoff + jitter. 429 is the
                    # concurrent-request limit, which clears as peers finish.
                    last_error = ASRError(
                        f"Fish Audio ASR returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 0.5))
                    continue
                if response.status_code != 200:
                    if response_has_no_speech(response):
                        return ASRResult("", wav_duration(audio_path), language, [])
                    raise ASRError(
                        f"Fish Audio ASR returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                return self._parse_response(response.json())
            except ASRError:
                # 4xx / unparseable responses are deterministic — no retry.
                raise
            except Exception as e:  # noqa: BLE001 — transport errors are retriable
                last_error = e
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
        raise ASRError(
            f"Fish Audio ASR failed after {self._max_retries + 1} tries: {last_error}"
        )

    def _parse_response(self, resp_json) -> ASRResult:
        if not isinstance(resp_json, dict):
            raise ASRError(f"ASR 返回了非 JSON 对象响应: {str(resp_json)[:200]}")
        if resp_json.get("error"):
            if is_no_speech(resp_json["error"]):
                duration = float(resp_json.get("duration") or 0.0)
                return ASRResult("", duration, self._language, [])
            raise ASRError(f"ASR API 返回错误: {str(resp_json['error'])[:300]}")
        for key in ("text", "duration"):
            if key not in resp_json:
                raise ASRError(f"ASR 响应缺少字段: {key}")
        segments: list[ASRSegment] = []
        for s in resp_json.get("segments") or []:
            if not isinstance(s, dict):
                continue
            try:
                segments.append(
                    ASRSegment(
                        text=str(s.get("text", "")),
                        start=float(s["start"]),
                        end=float(s["end"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return ASRResult(
            text=str(resp_json["text"]),
            duration=float(resp_json["duration"]),
            language_code=resp_json.get("language_code"),
            segments=segments,
        )
