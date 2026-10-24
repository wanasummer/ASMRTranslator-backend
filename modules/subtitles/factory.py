"""Subtitle adapter factory — auto-detect format and return the right adapter."""

import os
from typing import Optional
from ..error import ConfigurationError, descriptor
from ..text_decoder import TextDecodingError, read_text
from .base import SubtitleAdapter
from .models import SubtitleDocument
from .vtt_adapter import VTTAdapter
from .srt_adapter import SRTAdapter
from .lrc_adapter import LRCAdapter


# Registry of available adapters — add new ones here
_BUILTIN_ADAPTERS: list[type[SubtitleAdapter]] = [
    VTTAdapter,
    SRTAdapter,
    LRCAdapter,
]

_registry: list[SubtitleAdapter] = []


def _ensure_registry():
    if not _registry:
        for cls in _BUILTIN_ADAPTERS:
            _registry.append(cls())


def register_adapter(adapter: SubtitleAdapter):
    """Register a custom adapter instance."""
    _ensure_registry()
    _registry.append(adapter)


def detect_format(path: str) -> Optional[str]:
    """Detect subtitle format from file extension."""
    _ensure_registry()
    ext = os.path.splitext(path)[1].lower()
    for adapter in _registry:
        if ext in adapter.extensions:
            return adapter.format_name
    return None


def get_adapter(path: str) -> Optional[SubtitleAdapter]:
    """Get the appropriate adapter for a subtitle file.

    Matches by file extension first, then by content detection.
    """
    _ensure_registry()
    ext = os.path.splitext(path)[1].lower()

    # Try extension match
    for adapter in _registry:
        if ext in adapter.extensions:
            return adapter

    # Fallback: try content detection
    for adapter in _registry:
        if adapter.detect(path):
            return adapter

    return None


def parse_subtitle(path: str, encoding: str = "auto") -> SubtitleDocument:
    """Parse any supported subtitle file into universal format.

    Args:
        path: Path to subtitle file (.vtt, .srt, etc.)
        encoding: Explicit file encoding, or auto for shared text detection.

    Returns:
        SubtitleDocument with universal SubtitleBlocks.

    Raises:
        ValueError: If format is not supported.
    """
    adapter = get_adapter(path)
    if adapter is None:
        supported = ", ".join(
            f"*.{ext}" for a in _registry for ext in a.extensions
        ) if _registry else "none registered"
        raise ValueError(
            f"Unsupported subtitle format: {os.path.splitext(path)[1]}. "
            f"Supported: {supported}"
        )
    try:
        decoded = read_text(path, encoding)
    except TextDecodingError as error:
        raise ConfigurationError(descriptor("CFG_SUBTITLE_ENCODING"),
                                 detail=str(error), cause=error) from error
    return adapter.parse_text(decoded.text)
