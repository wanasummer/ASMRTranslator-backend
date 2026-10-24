"""Extension point for pluggable Japanese subtitle cleaners."""

from abc import ABC, abstractmethod

from .models import JapaneseCleaningResult


class JapaneseCleanerUnavailableError(RuntimeError):
    """Raised when a configured cleaner cannot initialize its runtime data."""


class JapaneseCleaner(ABC):
    """Transform ASR subtitle blocks without knowing later LLM/TTS stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable implementation identifier used in diagnostics."""

    @abstractmethod
    def clean(self, blocks) -> JapaneseCleaningResult:
        """Return new blocks plus an audit report; never mutate caller blocks."""
