"""Universal subtitle data model — all adapters output this format."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SubtitleBlock:
    """A single subtitle block in universal format.

    All adapters (VTT, SRT, ASS, etc.) convert to this common format.
    """
    index: int             # Sequential index (0-based)
    start_ms: int          # Start time in milliseconds
    end_ms: int            # End time in milliseconds
    text: str               # Clean subtitle text (speaker prefix stripped)
    speaker: Optional[str] = None   # Detected speaker name (if any)
    raw_text: str = ""      # Original text before speaker detection
    metadata: dict = None   # Extra format-specific data

    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class SubtitleDocument:
    """Container for a parsed subtitle file."""
    blocks: list[SubtitleBlock]
    format_name: str          # "vtt", "srt", "ass", etc.
    language: Optional[str] = None
    metadata: dict = None
