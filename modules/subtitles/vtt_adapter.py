"""VTT (WebVTT) subtitle adapter."""

import re
from typing import Optional
from .base import SubtitleAdapter
from .models import SubtitleBlock, SubtitleDocument


def _ts_to_ms(ts: str) -> int:
    parts = ts.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        raise ValueError(f"Invalid timestamp: {ts}")
    return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)


def _detect_speaker(text: str, known_speakers: set) -> tuple[Optional[str], str]:
    """Detect speaker prefix like '遥：', '瑠衣:' or '[遥]'."""
    m = re.match(r'^([^\s]{1,6})[：:]\s*(.*)', text)
    if m:
        name = m.group(1).strip()
        if not known_speakers or name in known_speakers:
            return name, m.group(2).strip()
    m = re.match(r'^\[([^\]]+)\]\s*(.*)', text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


class VTTAdapter(SubtitleAdapter):
    """Adapter for WebVTT (.vtt) subtitle files."""

    @property
    def format_name(self) -> str:
        return "vtt"

    @property
    def extensions(self) -> list[str]:
        return [".vtt"]

    def parse_text(self, content: str) -> SubtitleDocument:
        lines = content.split("\n")

        # First pass: collect known speakers
        known_speakers = set()
        for line in lines:
            m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', line.strip())
            if m:
                known_speakers.add(m.group(1).strip())

        blocks: list[SubtitleBlock] = []
        start_ms = None
        end_ms = None
        text_lines: list[str] = []

        def _save():
            nonlocal start_ms, end_ms, text_lines
            if start_ms is not None and text_lines:
                raw = " ".join(text_lines).strip()
                if raw:
                    speaker, clean = _detect_speaker(raw, known_speakers)
                    blocks.append(SubtitleBlock(
                        index=len(blocks),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=clean,
                        speaker=speaker,
                        raw_text=raw,
                    ))
            start_ms = None
            end_ms = None
            text_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip WEBVTT header and blank lines (delimiters between blocks)
            if stripped == "WEBVTT" or stripped == "":
                _save()
                i += 1
                continue

            # Try timestamp line
            ts_match = re.match(
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})\s*-->\s*'
                r'(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})',
                stripped
            )
            if ts_match:
                _save()
                start_ms = _ts_to_ms(ts_match.group(1))
                end_ms = _ts_to_ms(ts_match.group(2))
                i += 1
                continue

            # Skip cue identifiers (single line before timestamp, no colon)
            # A cue identifier is a line that is NOT a timestamp and has no colon
            # If we see a line that could be a cue id, skip it — the next line
            # should be the timestamp
            if start_ms is None and stripped and ":" not in stripped:
                # Could be a cue identifier; skip it
                i += 1
                continue

            # Actual subtitle text
            if stripped:
                text_lines.append(stripped)
            i += 1

        _save()  # Last block

        return SubtitleDocument(blocks=blocks, format_name="vtt")
