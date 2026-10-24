"""TTS common data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TTSSegmentContext:
    """Source speech returned by the splitting stage; never mutated by TTS."""

    index: int
    speech_file: Optional[str] = None

    @classmethod
    def from_segment(cls, segment: dict) -> "TTSSegmentContext":
        return cls(index=segment["index"], speech_file=segment.get("speech_file"))


@dataclass
class VoiceInfo:
    """Describes a TTS voice."""
    id: str                    # Voice identifier (e.g. "zh-CN-XiaoxiaoNeural")
    name: str                  # Human-readable name
    gender: Optional[str] = None
    language: Optional[str] = None
    description: str = ""


@dataclass
class TTSConfig:
    """Per-segment TTS configuration passed through the pipeline."""
    text: str
    speaker: Optional[str] = None
    voice: Optional[str] = None
    rate: str = "+0%"
    volume: float = 1.0
    metadata: dict = field(default_factory=dict)
