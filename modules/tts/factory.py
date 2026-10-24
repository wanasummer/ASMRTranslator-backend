"""TTS adapter factory — create TTS backends by name."""

from typing import Optional
from .base import TTSAdapter
from .edge_tts_adapter import EdgeTTSAdapter
from .mimo_tts_adapter import MimoTTSAdapter
from .fish_audio_adapter import FishAudioTTSAdapter
from .minimax_tts_adapter import MiniMaxTTSAdapter
from .indextts_adapter import IndexTTSAdapter
from .common_tts_adapter import CommonTTSAdapter


# Registry of available backends — add new ones here
_BUILTIN_BACKENDS: dict[str, type[TTSAdapter]] = {
    "common-tts": CommonTTSAdapter,
    "edge-tts": EdgeTTSAdapter,
    "mimo-tts": MimoTTSAdapter,
    "fish-audio": FishAudioTTSAdapter,
    "minimax-tts": MiniMaxTTSAdapter,
    "indextts-2.5": IndexTTSAdapter,
}

_registry: dict[str, type[TTSAdapter]] = {}


def _ensure_registry():
    if not _registry:
        _registry.update(_BUILTIN_BACKENDS)


def register_backend(name: str, adapter_cls: type[TTSAdapter]):
    """Register a custom TTS backend."""
    _ensure_registry()
    _registry[name] = adapter_cls


def get_available_backends() -> list[str]:
    """List registered backend names."""
    _ensure_registry()
    return list(_registry.keys())


def create_tts(
    backend: str = "edge-tts",
    voice_map: Optional[dict[str, str]] = None,
    **kwargs,
) -> TTSAdapter:
    """Create a TTS adapter instance.

    Args:
        backend: Backend name, e.g. "edge-tts".
        voice_map: Speaker→voice ID mapping (passed to adapter constructor).
        **kwargs: Additional backend-specific arguments.

    Returns:
        A TTSAdapter instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    _ensure_registry()
    cls = _registry.get(backend)
    if cls is None:
        supported = ", ".join(_registry.keys()) if _registry else "none registered"
        raise ValueError(
            f"Unknown TTS backend: '{backend}'. "
            f"Available: {supported}"
        )
    return cls(voice_map=voice_map, **kwargs)
