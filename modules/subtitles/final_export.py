"""Stable final-subtitle contract used by project artifact exports."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

from ..io import write_text


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
    write_text(str(path), "\n".join(lines), encoding="utf-8")
    return str(path.resolve())
