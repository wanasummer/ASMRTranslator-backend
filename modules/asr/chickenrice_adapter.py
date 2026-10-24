"""ChickenRice whole-audio subtitle-service adapter."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import time
from dataclasses import dataclass, replace
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import ASRError
from ..vtt_parser import VTTBlock
from .external_timeline import external_slice_pairs, restore_speech_blocks
from .base import ASRAdapter
from .models import ASRResult
from ..plugin_logs import relay_plugin_logs
from .no_speech import ASRNoSpeechError, response_has_no_speech, is_no_speech


@dataclass(frozen=True)
class ChickenRiceSubtitleResult:
    language: str
    blocks: list[VTTBlock]
    source: str
    timeline_source: str


class ChickenRiceASRAdapter(ASRAdapter):
    """Submit one complete audio file to the asynchronous ChickenRice service."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:7870",
        api_key: str = "",
        vad_provider: str = "external",
        output_language: str = "ja",
        timeout: float = 120.0,
        poll_interval: float = 1.5,
        job_timeout: float = 21600.0,
        **_ignored,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key.strip()
        self.vad_provider = vad_provider
        self.output_language = output_language
        self._timeout = float(timeout)
        self._network_policy = NetworkPolicy()
        self.poll_interval = float(poll_interval)
        self.job_timeout = float(job_timeout)
        if self.vad_provider not in {"external", "chickenrice"}:
            raise ASRError("海南鸡 vad_provider 必须是 external 或 chickenrice")
        if self.output_language not in {"ja", "zh"}:
            raise ASRError("海南鸡 output_language 必须是 ja 或 zh")

    @property
    def name(self) -> str:
        return "chickenrice"

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def uses_project_vad(self) -> bool:
        return self.vad_provider == "external"

    async def transcribe(self, audio_path: str, language: Optional[str] = None,
                         timestamps: bool = True, client=None) -> ASRResult:
        raise ASRError("ChickenRice 必须通过整段音频字幕契约调用")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _external_segments(slices) -> list[dict[str, int]]:
        return [segment for segment, _ in external_slice_pairs(slices)]

    async def transcribe_document(self, audio_path: str, slices=None) -> ChickenRiceSubtitleResult:
        data = {"vad_provider": self.vad_provider, "output_language": self.output_language}
        if self.vad_provider == "external":
            pairs = external_slice_pairs(slices or [])
            segments = [segment for segment, _ in pairs]
            if not segments:
                return ChickenRiceSubtitleResult(self.output_language, [], "chickenrice", "external_vad")
            data["segments"] = json.dumps(segments, ensure_ascii=False)

        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 30.0))
        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=timeout, headers=self._headers) as client:
            try:
                mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
                with open(audio_path, "rb") as audio:
                    response = await client.post(
                        f"{self.endpoint}/v1/jobs",
                        files={"audio": (os.path.basename(audio_path), audio, mime)},
                        data=data,
                    )
                self._raise_for_status(response, "创建字幕任务")
                job_id = response.json().get("id")
                if not job_id:
                    raise ASRError("海南鸡创建任务响应缺少 id")

                stop_logs = asyncio.Event()
                log_task = asyncio.create_task(relay_plugin_logs(
                    client, self.endpoint, str(job_id), self._headers,
                    stop_logs, source="chickenrice",
                    interval=min(self.poll_interval, 0.75),
                ))
                try:
                    deadline = time.monotonic() + self.job_timeout
                    while time.monotonic() < deadline:
                        status_response = await client.get(f"{self.endpoint}/v1/jobs/{job_id}")
                        self._raise_for_status(status_response, "查询字幕任务")
                        job = status_response.json()
                        if job.get("status") == "succeeded":
                            break
                        if job.get("status") == "failed":
                            if is_no_speech(job):
                                return ChickenRiceSubtitleResult(
                                    self.output_language, [], "chickenrice",
                                    "external_vad" if self.vad_provider == "external" else "chickenrice_vad",
                                )
                            raise ASRError(str(job.get("error") or job.get("message") or "海南鸡字幕任务失败"))
                        await asyncio.sleep(self.poll_interval)
                    else:
                        raise ASRError("海南鸡字幕任务等待超时")
                finally:
                    stop_logs.set()
                    await log_task

                result_response = await client.get(f"{self.endpoint}/v1/jobs/{job_id}/result")
                self._raise_for_status(result_response, "获取字幕结果")
                result = self._parse_result(result_response.json())
                if self.vad_provider == "external":
                    result = replace(result, blocks=restore_speech_blocks(result.blocks, pairs))
                return result
            except ASRError:
                raise
            except (httpx.HTTPError, OSError, ValueError) as exc:
                raise ASRError(f"海南鸡服务请求失败：{exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return
        if response.status_code == 401:
            raise ASRError("海南鸡 API Key 无效")
        if response_has_no_speech(response):
            raise ASRNoSpeechError("海南鸡未识别到可用文字")
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:300]
        raise ASRError(f"{action}失败（HTTP {response.status_code}）：{detail or '未知错误'}")

    def _parse_result(self, payload: dict) -> ChickenRiceSubtitleResult:
        if is_no_speech(payload):
            return ChickenRiceSubtitleResult(
                self.output_language, [], "chickenrice",
                "external_vad" if self.vad_provider == "external" else "chickenrice_vad",
            )
        language = str(payload.get("language", ""))
        if language not in {"ja", "zh"}:
            raise ASRError(f"海南鸡返回了无效语言：{language or '空'}")
        if language != self.output_language:
            raise ASRError(f"海南鸡响应语言不符：预期 {self.output_language}，实际 {language}")
        blocks = []
        for raw in payload.get("blocks", []):
            try:
                blocks.append(VTTBlock(
                    index=int(raw["index"]),
                    start_ms=int(raw["start_ms"]),
                    end_ms=int(raw["end_ms"]),
                    text=str(raw.get("text") or ""),
                    skip_tts=bool(raw.get("skip_tts", False)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ASRError(f"海南鸡字幕块契约无效：{raw}") from exc
        return ChickenRiceSubtitleResult(
            language=language,
            blocks=blocks,
            source=str(payload.get("source", "chickenrice")),
            timeline_source=str(payload.get("timeline_source", "")),
        )
