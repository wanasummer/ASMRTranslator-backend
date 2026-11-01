"""Shared contract for replaceable voice activity detection adapters."""

from abc import ABC, abstractmethod

from .models import SpeechRegion


class VADEngine(ABC):
    """Detect speech regions in an audio file."""

    @property
    def available(self) -> bool:
        """Whether this engine can actually run detection (True for a live one)."""
        return True

    @abstractmethod
    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        """Return raw speech regions as absolute ``(start, end)`` seconds."""
        ...


class NoOpVAD(VADEngine):
    """Placeholder engine — never runs; signals VAD is unavailable/disabled."""

    def __init__(self, reason: str = "VAD 未启用"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        return []
