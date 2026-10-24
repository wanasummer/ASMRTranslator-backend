"""Explicit request ownership and speech-only output for document ASR adapters."""

from dataclasses import replace
from typing import TypedDict

from ..error import ASRError
from ..vad.models import SpeechRegion
from ..vad.timeline import speech_bounds_ms
from ..vtt_parser import VTTBlock


class ExternalSegment(TypedDict):
    index: int
    start_ms: int
    end_ms: int


def external_slice_pairs(slices) -> list[tuple[ExternalSegment, SpeechRegion]]:
    """Keep existing non-overlapping wire ranges, with their original owner."""
    pairs = []
    previous_end = 0
    for region in sorted(slices, key=lambda item: item.start):
        start_ms = max(round(region.start * 1000), previous_end)
        end_ms = round(region.end * 1000)
        if end_ms <= start_ms:
            continue
        pairs.append(({"index": len(pairs), "start_ms": start_ms, "end_ms": end_ms}, region))
        previous_end = end_ms
    return pairs


def restore_speech_blocks(blocks: list[VTTBlock], pairs) -> list[VTTBlock]:
    """Validate server ownership before returning new blocks without padding."""
    if len(blocks) != len(pairs):
        raise ASRError("ASR 返回块数量与项目 VAD 切片不一致")
    output = []
    previous_end = 0
    for block, (request, source) in zip(blocks, pairs):
        if (block.index, block.start_ms, block.end_ms) != (
                request['index'], request['start_ms'], request['end_ms']):
            raise ASRError("ASR 返回时间轴与项目 VAD 请求不一致")
        start_ms, end_ms = speech_bounds_ms(source)
        start_ms = max(start_ms, previous_end)
        if end_ms <= start_ms:
            raise ASRError("ASR 人声时间边界无效")
        output.append(replace(block, start_ms=start_ms, end_ms=end_ms,
                              asr_slice=source.asr_slice, source_text=block.text))
        previous_end = end_ms
    return output
