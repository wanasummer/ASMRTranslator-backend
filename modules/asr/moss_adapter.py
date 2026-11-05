"""MOSS recognition adapter; native blocks feed the dedicated detect stage."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import math
import re
from typing import Sequence

from ..analysis_plugins import AnalysisPluginClient, AnalysisPluginError
from ..error import ASRError
from ..vtt_parser import VTTBlock
from ..vad.contracts import AsrSlice
from ..vad.models import SpeechRegion
from ..subtitle_timeline import calibrate_final_subtitle_end, SubtitleTimelineError
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
    japanese_cleanup_required: bool = False


def merge_slice_blocks(blocks: Sequence[VTTBlock]) -> list[VTTBlock]:
    """Remove exact duplicate observations from overlapping VAD crops only.

    Preserve native timestamps and simultaneous speakers within one crop.
    Raw observations remain available in the per-slice response audit.
    """
    output = []
    for block in sorted(blocks, key=lambda item: item.start_ms):
        duplicate = False
        for previous in output:
            if (previous.asr_slice == block.asr_slice
                    or previous.end_ms <= block.start_ms
                    or previous.start_ms >= block.end_ms):
                continue
            overlap = min(previous.end_ms, block.end_ms) - max(previous.start_ms, block.start_ms)
            if (re.sub(r"\s+", "", previous.text) == re.sub(r"\s+", "", block.text)
                    and overlap * 2 >= min(previous.end_ms - previous.start_ms,
                                            block.end_ms - block.start_ms)):
                duplicate = True
                break
        if not duplicate:
            output.append(replace(block, index=len(output)))
    return output


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
        return True

    async def transcribe(self, audio_path, language=None, timestamps=True, client=None) -> ASRResult:
        raise ASRError("MOSS 必须通过项目 VAD 切片文档契约调用")

    async def transcribe_document(
        self, audio_path: str, slices: Sequence[SpeechRegion] | None = None,
    ) -> MossDocumentResult:
        """Consume only VAD crops; return new blocks on the recording timeline.

        Requests run sequentially to bound model memory. Native speakers are
        scoped to a crop, never inferred to match speakers in another crop.
        The input regions and model response dictionaries are not mutated.
        """
        if not slices:
            raise ASRError("MOSS 需要项目 VAD 切片，禁止回退为整段音频推理")
        crops = []
        previous_start = -1
        for position, region in enumerate(slices, 1):
            if not all(math.isfinite(value) for value in (region.start, region.end)):
                raise ASRError("MOSS VAD 切片时间无效")
            start, end = round(region.start * 1000), round(region.end * 1000)
            if not 0 <= start < end or start < previous_start:
                raise ASRError("MOSS VAD 切片时间或顺序无效")
            parent = region.asr_slice or AsrSlice(
                f"moss-slice-{position:06d}", start, end,
                region.speech_start_ms if region.speech_start_ms is not None else start,
                region.speech_end_ms if region.speech_end_ms is not None else end, (),
            )
            crops.append((start, end, parent))
            previous_start = start

        blocks, responses = [], []
        speaker_count = 0
        for position, (start, end, parent) in enumerate(crops, 1):
            try:
                body, duration = await asyncio.to_thread(
                    self.client.request, "/transcribe", audio_path, {"language": "ja"},
                    start_ms=start, end_ms=end,
                )
                if abs(duration - (end - start)) > 1:
                    raise ASRError("MOSS 返回音频长度与 VAD 切片不符")
                result = self.parse_result(body, duration)
            except AnalysisPluginError as exc:
                raise ASRError(f"MOSS 第 {position}/{len(crops)} 片转写失败：{exc}") from exc
            speakers = {}
            for block in result.blocks:
                if block.speaker not in speakers:
                    speaker_count += 1
                    speakers[block.speaker] = f"S{speaker_count:02d}"
                blocks.append(replace(
                    block, start_ms=block.start_ms + start, end_ms=block.end_ms + start,
                    speaker=speakers[block.speaker], asr_slice=parent, source_text=block.text,
                ))
            responses.append({"slice_id": parent.id, "start_ms": start, "end_ms": end,
                              "speaker_map": speakers, "response": body})
            print(f"       [moss] 切片 {position}/{len(crops)}：{len(result.blocks)} 个字幕块")
        return MossDocumentResult("ja", merge_slice_blocks(blocks),
            raw_response={"policy_version": "project-vad-v5-original-audio", "speaker_scope": "slice",
                          "slices": responses})

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
                    or not previous <= start < end or start >= duration_ms
                    or not isinstance(text, str) or not text.strip()
                    or not isinstance(speaker, str) or re.fullmatch(r"S\d+", speaker) is None):
                raise ASRError(f"MOSS 字幕时间轴或说话人无效：blocks[{position - 1}]")
            blocks.append(VTTBlock(position - 1, start, end, text, speaker=speaker))
            previous = start
        try:
            calibrated = calibrate_final_subtitle_end(blocks, duration_ms)
        except SubtitleTimelineError as exc:
            raise ASRError(f"MOSS 字幕时间轴无效：{exc}") from exc
        return MossDocumentResult("ja", list(calibrated.blocks), raw_response=body)
