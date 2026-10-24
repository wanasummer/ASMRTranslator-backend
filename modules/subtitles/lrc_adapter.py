"""LRC (Lyrics) subtitle adapter.

Format:
  [MM:SS.xx]text
  [MM:SS.xxx]text
  [HH:MM:SS.xxx]text

LRC has no end timestamps — end time is inferred from the next line's start.
Metadata tags like [ti:Title] are ignored.
"""

import re
from .base import SubtitleAdapter
from .models import SubtitleBlock, SubtitleDocument


def _lrc_ts_to_ms(ts: str) -> int:
    """Convert LRC timestamp to milliseconds.

    Handles: MM:SS.xx, MM:SS.xxx, HH:MM:SS.xxx
    """
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        # MM:SS.xx
        m, s = parts
        return int(m) * 60000 + int(float(s) * 1000)
    elif len(parts) == 3:
        # HH:MM:SS.xxx
        h, m, s = parts
        return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)
    else:
        raise ValueError(f"Invalid LRC timestamp: {ts}")


def _detect_speaker(text: str, known_speakers: set):
    m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


# Metadata tags to skip
_METADATA_TAGS = {"ti", "ar", "al", "by", "offset", "re", "ve", "length"}


class LRCAdapter(SubtitleAdapter):
    """Adapter for LRC (.lrc) lyric files."""

    @property
    def format_name(self) -> str:
        return "lrc"

    @property
    def extensions(self) -> list[str]:
        return [".lrc"]

    def parse_text(self, content: str) -> SubtitleDocument:
        lines = content.split("\n")

        # First pass: collect speakers
        known_speakers = set()
        for line in lines:
            m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', line.strip())
            if m:
                known_speakers.add(m.group(1).strip())

        # Parse all timestamped lines
        raw_entries: list[tuple[int, str]] = []
        ts_pattern = re.compile(r'\[(\d{1,2}:\d{2}:\d{2}\.\d{1,3}|\d{1,2}:\d{2}\.\d{1,3})\](.*)')

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            m = ts_pattern.match(stripped)
            if not m:
                continue

            ts_str = m.group(1)
            text = m.group(2).strip()

            # Skip metadata tags like [ti:Title]
            tag_match = re.match(r'^(\w+):', text)
            if tag_match and tag_match.group(1).lower() in _METADATA_TAGS:
                continue

            if not text:
                continue

            try:
                start_ms = _lrc_ts_to_ms(ts_str)
                raw_entries.append((start_ms, text))
            except ValueError:
                continue

        # Sort by timestamp (LRC may not be in order)
        raw_entries.sort(key=lambda e: e[0])

        # Build blocks: end time = next entry's start, or start + 3s
        blocks: list[SubtitleBlock] = []
        for i, (start_ms, text) in enumerate(raw_entries):
            if i + 1 < len(raw_entries):
                end_ms = raw_entries[i + 1][0] - 1
            else:
                end_ms = start_ms + 3000  # last line: default 3s

            if end_ms <= start_ms:
                end_ms = start_ms + 1000

            speaker, clean = _detect_speaker(text, known_speakers)
            blocks.append(SubtitleBlock(
                index=len(blocks),
                start_ms=start_ms,
                end_ms=end_ms,
                text=clean,
                speaker=speaker,
                raw_text=text,
            ))

        return SubtitleDocument(blocks=blocks, format_name="lrc")
