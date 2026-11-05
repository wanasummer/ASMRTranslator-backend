"""Subtitle boundary calibration independent of recognition implementations."""
from dataclasses import dataclass, replace
from typing import Sequence

from .vtt_parser import VTTBlock


class SubtitleTimelineError(ValueError):
    """A subtitle timeline cannot be restored to its source audio boundary."""


@dataclass(frozen=True)
class EndAdjustment:
    index: int
    end_ms: int
    restored_end_ms: int


@dataclass(frozen=True)
class FinalSubtitleCalibration:
    blocks: tuple[VTTBlock, ...]
    adjustments: tuple[EndAdjustment, ...]


def calibrate_final_subtitle_end(
    blocks: Sequence[VTTBlock], source_duration_ms: int,
) -> FinalSubtitleCalibration:
    """Return copies, correcting only the final subtitle's out-of-range end.

    The last subtitle must begin inside the real audio. All earlier subtitles
    must already fit it. Do not shift starts, redistribute durations, alter text
    or speakers, or accept a subtitle originating in appended silence.
    """
    if type(source_duration_ms) is not int or source_duration_ms <= 0:
        raise SubtitleTimelineError("源音频长度无效")
    output, adjustments = [], []
    previous_start = 0
    for position, block in enumerate(blocks):
        if (type(block.start_ms) is not int or type(block.end_ms) is not int
                or not previous_start <= block.start_ms < source_duration_ms
                or block.end_ms <= block.start_ms):
            raise SubtitleTimelineError("字幕起点、顺序或时长无效")
        end = block.end_ms
        if end > source_duration_ms:
            if position != len(blocks) - 1:
                raise SubtitleTimelineError("非末条字幕超出原切片边界")
            end = source_duration_ms
            adjustments.append(EndAdjustment(block.index, block.end_ms, end))
        output.append(replace(block, end_ms=end))
        previous_start = block.start_ms
    return FinalSubtitleCalibration(tuple(output), tuple(adjustments))
