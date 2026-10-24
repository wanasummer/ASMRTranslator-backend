"""Abstract base class for TTS adapters."""

from abc import ABC, abstractmethod
from typing import Optional
from .models import VoiceInfo, TTSSegmentContext


class TTSAdapter(ABC):
    """Base class for text-to-speech backends.

    Subclass and implement synthesize() + voices() to add a new TTS engine.
    """

    def __init__(self, voice_map: Optional[dict[str, str]] = None):
        """Initialize with optional speaker→voice mapping.

        Args:
            voice_map: Maps speaker names to TTS voice IDs.
                       {"default": "zh-CN-XiaoxiaoNeural",
                        "遥": "zh-CN-XiaoyiNeural"}
                       Falls back to "default" if speaker not found.
        """
        self._voice_map = dict(voice_map or {})

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name, e.g. 'edge-tts', 'cosyvoice'."""
        ...

    @property
    def output_extension(self) -> str:
        """Preferred segment file extension produced by this backend."""
        return ".mp3"

    def for_segment(self, context: TTSSegmentContext) -> "TTSAdapter":
        """Return an isolated segment configuration, or this stateless adapter."""
        return self

    def cache_inputs(self) -> dict:
        """Extra content-bearing inputs affecting synthesis for this segment."""
        return {}

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synthesize text to speech audio file.

        Args:
            text: Text to speak.
            voice: TTS voice ID (from get_available_voices()).
            output_path: Where to save the audio file.
            rate: Speed adjustment, e.g. "+10%", "-5%", "+0%" (normal).
            pitch: Pitch adjustment, e.g. "+10Hz", "-20Hz", "+0Hz" (normal).
                   Backends that don't support pitch ignore it.

        Returns:
            Path to the generated audio file.

        Raises:
            TTSError: If synthesis fails.
        """
        ...

    @abstractmethod
    def get_available_voices(self) -> list[VoiceInfo]:
        """Return list of voices this backend supports."""
        ...

    async def aclose(self) -> None:
        """Release batch-scoped async resources held by the adapter."""

    # ---- Voice mapping helpers ----

    def set_voice_map(self, voice_map: dict[str, str]):
        """Update the speaker→voice mapping."""
        self._voice_map.update(voice_map)

    def get_voice_for_speaker(self, speaker: Optional[str]) -> str:
        """Resolve voice ID for a speaker name. Falls back to 'default'."""
        if speaker and speaker in self._voice_map:
            return self._voice_map[speaker]
        return self._voice_map.get("default", "")

    # ---- Convenience for sync callers ----

    def synthesize_sync(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synchronous wrapper around synthesize()."""
        import asyncio

        async def run_and_close() -> str:
            try:
                return await self.synthesize(text, voice, output_path, rate, pitch)
            finally:
                await self.aclose()

        return asyncio.run(run_and_close())
