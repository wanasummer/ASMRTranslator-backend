"""Immutable subtitle-stage handoff; implementations copy legacy mutable blocks."""

from dataclasses import dataclass
from typing import Literal, Protocol
from ..vtt_parser import VTTBlock


@dataclass(frozen=True)
class SubtitleStageInput:
    blocks: tuple[VTTBlock, ...]
    source: Literal["subtitles", "asr"] = "subtitles"
    preserve_timeline: bool = False
    preserve_source_timeline: bool = False


@dataclass(frozen=True)
class SubtitleStageResult:
    blocks: tuple[VTTBlock, ...]
    stats: dict[str, int | bool]


class SubtitleStage(Protocol):
    def run(self, value: SubtitleStageInput) -> SubtitleStageResult: ...
