"""Typed data models for the ASR pipeline.

Times are float seconds, always on the absolute original-audio timeline once a
chunk has been converted (``AudioChunk.start`` offsets ``ASRSegment.start/end``
which are chunk-relative). ``SentenceSegment`` is the sentence-level unit the
LLM translate-and-chunk stage consumes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRSegment:
    """A single recognized fragment with timestamps (seconds)."""

    text: str
    start: float
    end: float


@dataclass
class ASRResult:
    """Full response from one ASR transcription request."""

    text: str
    duration: float
    language_code: Optional[str] = None
    segments: list[ASRSegment] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


@dataclass
class AudioChunk:
    """A coarse-chunked window extracted from the original audio."""

    id: str
    start: float  # absolute seconds on the original timeline
    end: float
    path: str


@dataclass
class SentenceSegment:
    """A sentence-level utterance with absolute timestamps (seconds)."""

    id: str
    start: float
    end: float
    text: str
