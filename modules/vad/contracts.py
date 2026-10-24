"""Immutable millisecond contracts; no downstream stage dependencies."""

from dataclasses import dataclass


class VadAnalysisError(RuntimeError):
    """VAD analysis-stage contract failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class VadRegion:
    id: str
    speech_start_ms: int
    speech_end_ms: int
    source: str = "initial"
    threshold_ms: int = 1500
    parent_region_id: str | None = None


@dataclass(frozen=True)
class AsrSlice:
    id: str
    extraction_start_ms: int
    extraction_end_ms: int
    speech_start_ms: int
    speech_end_ms: int
    region_ids: tuple[str, ...]
    forced_split: bool = False


@dataclass(frozen=True)
class VadAnalysis:
    regions: tuple[VadRegion, ...]
    slices: tuple[AsrSlice, ...]
    initial_regions: tuple[VadRegion, ...] = ()
