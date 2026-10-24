"""Subtitle format adapter system.

Usage:
    from modules.subtitles import parse_subtitle, SubtitleBlock

    doc = parse_subtitle("subtitles.vtt")
    for block in doc.blocks:
        print(block.start_ms, block.speaker, block.text)

Extending with a new format:
    1. Create a new adapter class inheriting SubtitleAdapter
    2. Add it to _BUILTIN_ADAPTERS in factory.py
    Done.
"""

from .models import SubtitleBlock, SubtitleDocument
from .base import SubtitleAdapter
from .factory import (
    parse_subtitle,
    get_adapter,
    detect_format,
    register_adapter,
)
from .vtt_adapter import VTTAdapter
from .srt_adapter import SRTAdapter
from .lrc_adapter import LRCAdapter

__all__ = [
    "SubtitleBlock",
    "SubtitleDocument",
    "SubtitleAdapter",
    "parse_subtitle",
    "get_adapter",
    "detect_format",
    "register_adapter",
    "VTTAdapter",
    "SRTAdapter",
    "LRCAdapter",
]
