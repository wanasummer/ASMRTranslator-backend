"""SRT (SubRip) subtitle adapter — example of how to extend to a new format."""

import re
from .base import SubtitleAdapter
from .models import SubtitleBlock, SubtitleDocument


def _srt_ts_to_ms(ts: str) -> int:
    """Convert SRT timestamp 00:00:01,227 to ms."""
    h, m, s = ts.replace(",", ".").split(":")
    return int(h) * 3600000 + int(m) * 60000 + int(float(s) * 1000)


def _detect_speaker(text: str, known_speakers: set):
    m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text.strip()


class SRTAdapter(SubtitleAdapter):
    """Adapter for SubRip (.srt) subtitle files."""

    @property
    def format_name(self) -> str:
        return "srt"

    @property
    def extensions(self) -> list[str]:
        return [".srt"]

    def parse_text(self, content: str) -> SubtitleDocument:
        lines = content.split("\n")

        known_speakers = set()
        for line in lines:
            m = re.match(r'^([\u4e00-\u9fff·]{1,8})[：:]', line.strip())
            if m:
                known_speakers.add(m.group(1).strip())

        blocks: list[SubtitleBlock] = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Skip blank lines
            if stripped == "":
                i += 1
                continue

            # Sequence number
            if stripped.isdigit():
                i += 1
                continue

            # Timestamp line: 00:00:01,227 --> 00:00:05,043
            ts_match = re.match(
                r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})',
                lines[i] if i < len(lines) else ""
            )
            if ts_match:
                start_ms = _srt_ts_to_ms(ts_match.group(1))
                end_ms = _srt_ts_to_ms(ts_match.group(2))
                i += 1

                # Collect text lines until blank line
                text_lines = []
                while i < len(lines) and lines[i].strip() != "":
                    text_lines.append(lines[i].strip())
                    i += 1

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
                continue

            i += 1

        return SubtitleDocument(blocks=blocks, format_name="srt")
