"""Stable final-subtitle contract used by project artifact exports."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Protocol

from ..io import write_text

if TYPE_CHECKING:
    from ..audio_joiner import SegmentPlacement


class FinalSubtitleSource(Protocol):
    start_ms: int
    end_ms: int
    text: str
    speaker: Optional[str]


class FinalSubtitleError(ValueError):
    """A final subtitle artifact cannot be produced."""

    code = "SUBTITLE_FINAL_EMPTY"


@dataclass(frozen=True)
class FinalSubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: Optional[str]


def build_final_subtitles(blocks: Iterable[FinalSubtitleSource], *,
                          include_skipped: bool = False) -> list[FinalSubtitleCue]:
    """Return cues without mutation; subtitle-only exports include TTS skips."""
    cues: list[FinalSubtitleCue] = []
    for block in blocks:
        text = (getattr(block, "text", "") or "").strip()
        start_ms = int(getattr(block, "start_ms", 0))
        end_ms = int(getattr(block, "end_ms", 0))
        if (not include_skipped and getattr(block, "skip_tts", False)) or not text or end_ms <= start_ms:
            continue
        cues.append(FinalSubtitleCue(
            index=len(cues) + 1,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            speaker=getattr(block, "speaker", None),
        ))
    if not cues:
        raise FinalSubtitleError("最终字幕没有可导出的有效 cue")
    return cues


class IndexedSubtitleSource(FinalSubtitleSource, Protocol):
    index: int


def build_aligned_subtitles(blocks: Iterable[IndexedSubtitleSource],
                            placements: Iterable["SegmentPlacement"]) -> list[FinalSubtitleCue]:
    """Use returned audio spans, preserving real overlaps without mutating blocks.

    Missing segments have no audio; their omissions and placement spans are
    already recorded in output.join_audit.json by the joiner.
    """
    placed = {item.block_id: item for item in placements}
    mapped = []
    for block in blocks:
        item = placed.get(str(block.index))
        if item is not None and not getattr(block, "skip_tts", False):
            mapped.append(FinalSubtitleCue(0, item.subtitle_start_ms, item.subtitle_end_ms,
                                           block.text, block.speaker))
    return build_final_subtitles(mapped)


def _timestamp(milliseconds: int) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def write_final_vtt(cues: Iterable[FinalSubtitleCue], output_dir: str) -> str:
    """Write UTF-8/LF WebVTT to ``llm_subtitles/final.vtt``."""
    cue_list = list(cues)
    if not cue_list:
        raise FinalSubtitleError("最终字幕没有可导出的有效 cue")
    path = Path(output_dir) / "llm_subtitles" / "final.vtt"
    lines = ["WEBVTT", ""]
    for cue in cue_list:
        text = f"{cue.speaker}：{cue.text}" if cue.speaker else cue.text
        lines.extend([
            str(cue.index),
            f"{_timestamp(cue.start_ms)} --> {_timestamp(cue.end_ms)}",
            text,
            "",
        ])
    try:
        write_text(str(path), "\n".join(lines), encoding="utf-8")
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return str(path.resolve())
