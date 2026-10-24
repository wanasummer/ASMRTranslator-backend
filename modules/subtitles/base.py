"""Abstract base class for subtitle format adapters."""

from abc import ABC, abstractmethod
from .models import SubtitleDocument


class SubtitleAdapter(ABC):
    """Base adapter for subtitle files.

    Subclass this for each format (VTT, SRT, ASS, etc.).
    Register in factory.py when done.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name, e.g. 'vtt', 'srt'."""
        ...

    @property
    @abstractmethod
    def extensions(self) -> list[str]:
        """File extensions this adapter handles, e.g. ['.vtt']."""
        ...

    def parse(self, path: str, encoding: str = "utf-8-sig") -> SubtitleDocument:
        """Compatibility file API; default input is already normalized UTF-8.

        Use factory.parse_subtitle for automatic source encoding detection.
        Explicit encodings remain supported for direct callers.
        """
        with open(path, "r", encoding=encoding) as source:
            return self.parse_text(source.read())

    @abstractmethod
    def parse_text(self, content: str) -> SubtitleDocument:
        """Transform normalized Unicode text into blocks; no I/O or detection.

        The import boundary owns decoding/UTF-8 and LF newline normalization.
        Implementations preserve text and timing according to their format contract.
        """
        ...

    def detect(self, path: str) -> bool:
        """Quick check if this adapter can handle the file.

        Default implementation checks file extension.
        Override for content-based detection.
        """
        import os
        ext = os.path.splitext(path)[1].lower()
        return ext in self.extensions
