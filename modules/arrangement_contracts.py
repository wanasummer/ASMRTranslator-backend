"""Stable values passed from arrangement preparation to rendering and writing."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict


class ArrangementError(RuntimeError):
    """Invalid arrangement data or unavailable source artifacts."""


class Interval(TypedDict):
    start: float
    end: float


class Replacement(TypedDict):
    path: str
    text: str


class ArrangementEdit(TypedDict):
    index: int
    text: str
    japanese: str
    volume: float
    interval: Interval | None
    replacement: Replacement | None
    deleted: bool
    originalInterval: Interval | None
    ttsSample: Interval | None


class ArrangementState(TypedDict):
    schemaVersion: int
    sourceVersion: str
    loudnessScope: str
    globalVolume: float
    segments: list[ArrangementEdit]


@dataclass(frozen=True)
class ArrangedClip:
    index: int
    start_ms: int
    end_ms: int
    text: str
    path: Path
    volume: float
    reference_start_ms: int
    reference_end_ms: int
    sample_start_ms: int = 0
    sample_end_ms: int | None = None


@dataclass(frozen=True)
class ArrangementPlan:
    source_audio: Path
    clips: tuple[ArrangedClip, ...]
    global_loudness: bool


@dataclass(frozen=True)
class ArrangementAudioResult:
    audio: object
    clips: tuple[ArrangedClip, ...]
    origin_shift_ms: int = 0


class ArrangementAudioAdapter(Protocol):
    def render(self, plan: ArrangementPlan) -> ArrangementAudioResult: ...
