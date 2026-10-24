"""Adapter for the synchronous common ASR ``POST /job`` contract."""

from __future__ import annotations

import json
import mimetypes
import os
import asyncio
from dataclasses import dataclass, replace
from typing import Optional
from uuid import uuid4

import httpx

from ..network import NetworkPolicy, async_client

from ..error import ASRError
from ..vtt_parser import VTTBlock
from .base import ASRAdapter
from .models import ASRResult
from .external_timeline import external_slice_pairs, restore_speech_blocks
from ..plugin_logs import relay_plugin_logs
from .no_speech import ASRNoSpeechError, response_has_no_speech, is_no_speech


@dataclass(frozen=True)
class CommonASRDocumentResult:
    language: str
    blocks: list[VTTBlock]
    source: str
    timeline_source: str


class CommonASRAdapter(ASRAdapter):
    """Send full audio plus the project VAD timeline to a common ASR service."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8000",
        api_key: str = "",
        output_language: str = "ja",
        timeout: float = 21600.0,
        **_ignored,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key.strip()
        self.output_language = output_language.strip().lower()
        self._timeout = float(timeout)
        self._network_policy = NetworkPolicy()
        if not self.endpoint:
            raise ASRError("自定义 ASR 服务地址不能为空")
        if self.output_language not in {"ja", "zh"}:
            raise ASRError("自定义 ASR output_language 必须是 ja 或 zh")

    @property
    def name(self) -> str:
        return "common-asr"

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self.endpoint)

    def uses_project_vad(self) -> bool:
        return True

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client=None,
    ) -> ASRResult:
        raise ASRError("通用 ASR 服务必须通过整段音频文档契约调用")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _external_segments(slices) -> list[dict[str, int]]:
        return [segment for segment, _ in external_slice_pairs(slices)]

    async def transcribe_document(self, audio_path: str, slices=None) -> CommonASRDocumentResult:
        pairs = external_slice_pairs(slices or [])
        segments = [segment for segment, _ in pairs]
        if not segments:
            return CommonASRDocumentResult(
                language=self.output_language,
                blocks=[],
                source="common-asr",
                timeline_source="external_segments",
            )
        form = {
            "language": self.output_language,
            "segments": json.dumps(segments, ensure_ascii=False),
        }
        request_id = str(uuid4())
        form["request_id"] = request_id
        mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        timeout = httpx.Timeout(self._timeout, connect=min(self._timeout, 30.0))
        try:
            async with async_client(client_factory=httpx.AsyncClient,
                network_policy=self._network_policy,
                timeout=timeout,
                headers=self._headers,
            ) as client:
                stop_logs = asyncio.Event()
                log_task = asyncio.create_task(relay_plugin_logs(
                    client, self.endpoint, request_id, self._headers,
                    stop_logs, source="common-asr",
                ))
                try:
                    with open(audio_path, "rb") as audio:
                        response = await client.post(
                            f"{self.endpoint}/job",
                            files={"audio": (os.path.basename(audio_path), audio, mime)},
                            data=form,
                            headers={"X-Request-ID": request_id},
                        )
                finally:
                    stop_logs.set()
                    await log_task
            self._raise_for_status(response)
            result = self._parse_result(response.json(), segments)
            return replace(result, blocks=restore_speech_blocks(result.blocks, pairs))
        except ASRNoSpeechError:
            return CommonASRDocumentResult(
                language=self.output_language,
                blocks=[],
                source="common-asr",
                timeline_source="external_segments",
            )
        except ASRError:
            raise
        except (httpx.HTTPError, OSError, ValueError) as exc:
            raise ASRError(f"自定义 ASR 服务请求失败：{exc}") from exc

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == 401:
            raise ASRError("自定义 ASR API Key 无效")
        if response_has_no_speech(response):
            raise ASRNoSpeechError("自定义 ASR 未识别到可用文字")
        try:
            payload = response.json()
            detail = payload.get("error", {}).get("message") or payload.get("detail")
        except ValueError:
            detail = response.text[:300]
        raise ASRError(
            f"自定义 ASR 请求失败（HTTP {response.status_code}）：{detail or '未知错误'}"
        )

    def _parse_result(
        self,
        payload: dict,
        requested_segments: list[dict[str, int]],
    ) -> CommonASRDocumentResult:
        if not isinstance(payload, dict):
            raise ASRError("自定义 ASR 返回了非 JSON 对象")
        if is_no_speech(payload):
            return CommonASRDocumentResult(
                language=self.output_language,
                blocks=[],
                source=str(payload.get("source") or "common-asr"),
                timeline_source=str(payload.get("timeline_source") or "external_segments"),
            )
        if payload.get("contract_version") != 1 or payload.get("stage") != "asr":
            raise ASRError("自定义 ASR 返回了不兼容的契约版本或阶段")
        language = str(payload.get("language") or "").lower()
        if language != self.output_language:
            raise ASRError(
                f"自定义 ASR 响应语言不符：预期 {self.output_language}，实际 {language or '空'}"
            )
        raw_blocks = payload.get("blocks")
        if not isinstance(raw_blocks, list) or len(raw_blocks) != len(requested_segments):
            raise ASRError("自定义 ASR 返回块数量与外部分段不一致")
        blocks: list[VTTBlock] = []
        for position, (raw, requested) in enumerate(zip(raw_blocks, requested_segments)):
            try:
                index = int(raw["index"])
                start_ms = int(raw["start_ms"])
                end_ms = int(raw["end_ms"])
                text = str(raw.get("text") or "").strip()
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ASRError(f"自定义 ASR 字幕块契约无效：blocks[{position}]") from exc
            if (index, start_ms, end_ms) != (
                requested["index"], requested["start_ms"], requested["end_ms"]
            ):
                raise ASRError(f"自定义 ASR 改写了外部分段边界：blocks[{position}]")
            blocks.append(VTTBlock(index, start_ms, end_ms, text, skip_tts=False))
        return CommonASRDocumentResult(
            language=language,
            blocks=blocks,
            source=str(payload.get("source") or "common-asr"),
            timeline_source=str(payload.get("timeline_source") or "external_segments"),
        )
