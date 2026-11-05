"""Persistence adapter for cleaning diagnostics."""

from __future__ import annotations

import os

from ..storage import save_json, save_text
from .contracts import JapaneseCleaningResult


def _timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def write_cleaning_artifacts(result: JapaneseCleaningResult, output_dir: str) -> dict[str, str]:
    """Write cleaned VTT and JSON audit report; return their paths."""
    from ..storage import initialize_project
    initialize_project(output_dir)
    directory = os.path.join(output_dir, "asr_subtitles")
    vtt_path = os.path.join(directory, "cleaned.vtt")
    report_path = os.path.join(directory, "clean_report.json")

    lines = ["WEBVTT", ""]
    for block in result.blocks:
        lines.extend([
            str(block.index + 1),
            f"{_timestamp(block.start_ms)} --> {_timestamp(block.end_ms)}",
            f"{block.speaker}：{block.text or ''}" if block.speaker else block.text or "",
            "",
        ])
    vtt_path = save_text(vtt_path, "\n".join(lines))
    report_path = save_json(report_path, result.as_dict())
    return {"vtt": vtt_path, "report": report_path}
