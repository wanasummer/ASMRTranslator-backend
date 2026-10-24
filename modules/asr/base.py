"""Abstract base class for ASR adapters."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from .models import ASRResult


class ASRAdapter(ABC):
    """Base class for speech-to-text backends.

    Subclass and implement ``transcribe()`` to add a new ASR engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name, e.g. 'fish-asr'."""
        ...

    @property
    @abstractmethod
    def timeout(self) -> float:
        """Per-request timeout, for callers that want to build a shared client."""
        ...

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client=None,
    ) -> ASRResult:
        """Transcribe a single audio clip.

        Args:
            audio_path: Path to the audio file (wav/mp3/...).
            language: Optional source-language override (None = adapter default).
            timestamps: When True the result carries per-fragment time slices.
            client: Optional shared ``httpx.AsyncClient`` for keep-alive reuse.

        Returns:
            An ``ASRResult`` with the transcript and, when ``timestamps`` is
            True, the recognized fragments with chunk-relative timestamps.

        Raises:
            ASRError: If the request fails or the payload is malformed.
        """
        ...

    def is_configured(self) -> bool:
        """Whether this adapter has the credentials it needs to run."""
        return True


@runtime_checkable
class DocumentASRAdapter(Protocol):
    """Structural contract for services that transcribe one complete audio file."""

    def uses_project_vad(self) -> bool:
        """Whether the caller must provide the authored project-VAD timeline."""
        ...

    async def transcribe_document(self, audio_path: str, slices=None) -> Any:
        """Return a document result carrying language, blocks, source and timeline."""
        ...
