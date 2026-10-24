"""VAD data models — no ML-runtime dependency."""

from dataclasses import dataclass
from .contracts import AsrSlice


@dataclass
class SpeechRegion:
    """A detected speech region with absolute timestamps (seconds)."""

    start: float
    end: float
    asr_slice: AsrSlice | None = None
    # Decode padding is carried by start/end; downstream speech uses these.
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start
