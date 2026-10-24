"""Backward-compatible wrapper — delegates to the new adapter system.

    For new code, use `modules.subtitles.parse_subtitle()` directly.
    This module exists so existing code (pipeline.py) doesn't break.
"""

from dataclasses import dataclass
from typing import Optional, List, Protocol

from .subtitles import SubtitleBlock, parse_subtitle
from .vad.contracts import AsrSlice


class AlignmentMetadata(Protocol):
    id: str
    source_block_id: str
    method: str
    confidence: str


@dataclass
class VTTBlock:
    """Legacy VTTBlock — kept for backward compatibility.

    New code should use SubtitleBlock from modules.subtitles.
    """
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = None
    # Interjection marker: when True the translation TTS is skipped and the
    # original audio is kept for this span (set by the pipeline's
    # mark_interjection_skips before splitting).
    skip_tts: bool = False
    asr_slice: AsrSlice | None = None
    source_text: str = ""
    cleaned_source_text: str = ""
    aligned_sentence: AlignmentMetadata | None = None
    translation_available: bool = False
    # The voiced (speech-only) span as-authored (used by loudness matching to
    # measure just the speech of the pink slice). Kept for backward
    # compatibility — the new pink-segment geometry carries the coverage instead.
    voice_start_ms: Optional[int] = None
    voice_end_ms: Optional[int] = None
    # Pink-segment geometry (set by modules.pink_segments.attach_pink_geometry
    # before the audio slicer runs): the continuous slice [pink_start, pink_end]
    # this block belongs to, the TTS trigger offset within it, and the max TTS
    # duration allowed before bleeding into the next slice. None until attached.
    pink_start_ms: Optional[int] = None
    pink_end_ms: Optional[int] = None
    tts_delay_ms: Optional[int] = None
    max_allowed_tts_ms: Optional[int] = None

    @staticmethod
    def from_subtitle_block(b: SubtitleBlock) -> "VTTBlock":
        return VTTBlock(
            index=b.index,
            start_ms=b.start_ms,
            end_ms=b.end_ms,
            text=b.text,
            speaker=b.speaker,
        )


def parse_vtt(path: str) -> List[VTTBlock]:
    """Parse a VTT file (delegates to new adapter system)."""
    doc = parse_subtitle(path)
    return [VTTBlock.from_subtitle_block(b) for b in doc.blocks]
