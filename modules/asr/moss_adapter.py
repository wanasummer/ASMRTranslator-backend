"""Independent MOSS document adapter retaining native timestamps and speakers."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re

from ..analysis_plugins import AnalysisPluginClient, AnalysisPluginError
from ..error import ASRError
from ..vtt_parser import VTTBlock
from .base import ASRAdapter
from .models import ASRResult


@dataclass(frozen=True)
class MossDocumentResult:
    language: str
    blocks: list[VTTBlock]
    source: str = "moss-transcribe-diarize"
    timeline_source: str = "model"
    raw_response: dict | None = None
    preserve_source_timeline: bool = True


class MossASRAdapter(ASRAdapter):
    def __init__(self, endpoint="http://127.0.0.1:8008", language="ja", *, client=None, **_ignored):
        if language != "ja":
            raise ASRError("MOSS 主程序适配器目前只接入日语转写")
        try:
            self.client = client or AnalysisPluginClient(endpoint, "moss-transcribe-diarize", "moss-transcribe-diarize/1")
        except (AnalysisPluginError, ValueError) as exc:
            raise ASRError(str(exc)) from exc

    @property
    def name(self):
        return "moss-transcribe-diarize"

    @property
    def timeout(self):
        return 21600.0

    def uses_project_vad(self):
        return False

    async def transcribe(self, audio_path, language=None, timestamps=True, client=None) -> ASRResult:
        raise ASRError("MOSS 必须通过整段音频文档契约调用")

    async def transcribe_document(self, audio_path, slices=None) -> MossDocumentResult:
        try:
            body, duration = await asyncio.to_thread(self.client.request, "/transcribe", audio_path, {"language": "ja"})
            return self.parse_result(body, duration)
        except AnalysisPluginError as exc:
            raise ASRError(f"MOSS 转写失败：{exc}") from exc

    @staticmethod
    def parse_result(body, duration_ms) -> MossDocumentResult:
        if body.get("language") != "ja" or body.get("timeline_source") != "model":
            raise ASRError("MOSS 返回了不兼容的语言或时间轴")
        rows = body.get("blocks")
        if not isinstance(rows, list):
            raise ASRError("MOSS 响应缺少字幕块")
        blocks, previous = [], 0
        for position, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ASRError("MOSS 字幕块格式无效")
            start, end = row.get("start_ms"), row.get("end_ms")
            text, speaker = row.get("text"), row.get("speaker")
            if (type(row.get("index")) is not int or row["index"] != position
                    or type(start) is not int or type(end) is not int
                    or not previous <= start < end <= duration_ms
                    or not isinstance(text, str) or not text.strip()
                    or not isinstance(speaker, str) or re.fullmatch(r"S\d+", speaker) is None):
                raise ASRError(f"MOSS 字幕时间轴或说话人无效：blocks[{position - 1}]")
            blocks.append(VTTBlock(position - 1, start, end, text, speaker=speaker))
            previous = start
        return MossDocumentResult("ja", blocks, raw_response=body)
