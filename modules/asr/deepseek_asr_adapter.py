"""DeepSeek ASR adapter — multipart/form-data speech-to-text.

``POST https://api.deepseek.com/v1/tools/asr`` with ``Authorization: Bearer <key>``
and a multipart ``file`` field carrying a 16-bit PCM WAV or Ogg Opus audio clip
(up to 120s / 8 MiB).

Response shape::

    {"request_id": "...", "text": "...", "duration_ms": 1234}

The DeepSeek ASR endpoint does not return word- or segment-level timestamps;
when ``timestamps=True`` and non-empty text is returned, the whole clip is
emitted as a single ``ASRSegment`` spanning ``0.0 .. duration``.

Retry policy: 429, 5xx, and transport errors retry with exponential backoff +
jitter (``max_retries``); other 4xx responses fail fast.
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from typing import Optional

import httpx

from ..error import ASRError
from ..network import NetworkPolicy, async_client
from .base import ASRAdapter
from .cloud_audio import wav_duration
from .models import ASRResult, ASRSegment
from .no_speech import response_has_no_speech

_MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
}


class DeepSeekASRAdapter(ASRAdapter):
    """DeepSeek standalone ASR tool backend."""

    _DEFAULT_ENDPOINT = "https://api.deepseek.com/v1/tools/asr"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "ja",
        base_url: str = "",
        timeout: float = 120.0,
        max_retries: int = 2,
        **_ignored,
    ) -> None:
        self._api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        self._language = language
        self._endpoint = (base_url or self._DEFAULT_ENDPOINT).rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._network_policy = NetworkPolicy()

    @property
    def name(self) -> str:
        return "deepseek-asr"

    @property
    def timeout(self) -> float:
        """Per-request timeout, for callers that want to build a shared client."""
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def _mime_for(path: str) -> str:
        return _MIME_BY_EXT.get(Path(path).suffix.lower(), "audio/wav")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        """Transcribe one audio clip into an ``ASRResult``."""
        if not self._api_key:
            raise ASRError("DeepSeek ASR API Key 未配置")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        files = {
            "file": (Path(audio_path).name, audio_bytes, self._mime_for(audio_path))
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if client is not None:
                    response = await client.post(
                        self._endpoint, headers=headers, files=files
                    )
                else:
                    async with async_client(
                        client_factory=httpx.AsyncClient,
                        network_policy=self._network_policy,
                        timeout=self._timeout,
                    ) as ac:
                        response = await ac.post(
                            self._endpoint, headers=headers, files=files
                        )

                if (
                    response.status_code == 429 or response.status_code >= 500
                ) and attempt < self._max_retries:
                    last_error = ASRError(
                        f"DeepSeek ASR returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 0.5))
                    continue

                if response.status_code in (401, 403):
                    raise ASRError("DeepSeek ASR API Key 无效")

                if response.status_code != 200:
                    if response_has_no_speech(response):
                        return ASRResult(
                            "", wav_duration(audio_path), language or self._language, []
                        )
                    raise ASRError(
                        f"DeepSeek ASR returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )

                return self._parse_response(
                    response.json(),
                    audio_path=audio_path,
                    language=language or self._language,
                    timestamps=timestamps,
                )
            except ASRError:
                raise
            except Exception as exc:  # noqa: BLE001 — transport errors are retriable
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

        raise ASRError(
            f"DeepSeek ASR failed after {self._max_retries + 1} tries: {last_error}"
        )

    def _parse_response(
        self,
        resp_json: object,
        *,
        audio_path: str,
        language: str,
        timestamps: bool,
    ) -> ASRResult:
        if not isinstance(resp_json, dict):
            raise ASRError(f"DeepSeek ASR 返回了非 JSON 对象响应: {str(resp_json)[:200]}")
        if "text" not in resp_json:
            raise ASRError("DeepSeek ASR 响应缺少字段: text")

        text = str(resp_json["text"] or "").strip()
        duration_ms = resp_json.get("duration_ms")
        if duration_ms is not None:
            try:
                duration = float(duration_ms) / 1000.0
            except (TypeError, ValueError) as exc:
                raise ASRError(f"DeepSeek ASR 响应 duration_ms 无效: {duration_ms}") from exc
        else:
            duration = wav_duration(audio_path)

        segments = (
            [ASRSegment(text=text, start=0.0, end=duration)]
            if text and timestamps
            else []
        )
        return ASRResult(
            text=text,
            duration=duration,
            language_code=language,
            segments=segments,
        )
