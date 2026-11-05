"""Preserve imported source text and parsed rows before downstream transformation."""
from pathlib import Path
from typing import Iterable, Protocol

from ..storage import save_bytes, save_json


class ImportedSubtitleRow(Protocol):
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None


def save_subtitle_source(output_dir: str, source: str) -> None:
    path = Path(source)
    if path.is_file():
        save_bytes(Path(output_dir) / 'inputs/subtitle-source', path.read_bytes(), 'application/octet-stream')


def save_imported_subtitles(output_dir: str, blocks: Iterable[ImportedSubtitleRow]) -> None:
    root = Path(output_dir)
    save_json(root / 'inputs/subtitle-blocks', [
        {'index': row.index, 'start_ms': row.start_ms, 'end_ms': row.end_ms,
         'text': row.text, 'speaker': row.speaker} for row in blocks
    ])
