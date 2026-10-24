"""Xiaomi MiMo V2.5 speech-recognition adapter."""

from __future__ import annotations

import os
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import ASRError
from .base import ASRAdapter
from .cloud_audio import audio_data_url, wav_duration
from .models import ASRResult, ASRSegment
from .no_speech import response_has_no_speech


class MimoASRAdapter(ASRAdapter):
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.xiaomimimo.com/v1",
        timeout: float = 120.0,
        **_ignored,
    ) -> None:
        self.api_key = (api_key or os.environ.get("MIMO_API_KEY", "")).strip()
        self.base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._network_policy = NetworkPolicy()

    @property
    def name(self) -> str:
        return "mimo-asr"

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        if not self.api_key:
            raise ASRError("MiMo ASR API Key 未配置")
        payload = {
            "model": "mimo-v2.5-asr",
            "messages": [{
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {"data": audio_data_url(audio_path)}}],
            }],
            "asr_options": {"language": "auto"},
            "stream": False,
        }
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        if client is not None:
            response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        else:
            async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self._timeout) as own_client:
                response = await own_client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        if response.status_code in (401, 403):
            raise ASRError("MiMo ASR API Key 无效")
        if response_has_no_speech(response):
            duration = wav_duration(audio_path)
            return ASRResult("", duration, language or "ja", [])
        if not response.is_success:
            raise ASRError(f"MiMo ASR 请求失败（HTTP {response.status_code}）：{response.text[:300]}")
        try:
            text = str(response.json()["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ASRError("MiMo ASR 响应缺少识别文本") from exc
        duration = wav_duration(audio_path)
        segments = [ASRSegment(text=text, start=0.0, end=duration)] if text and timestamps else []
        return ASRResult(text=text, duration=duration, language_code=language or "ja", segments=segments)
