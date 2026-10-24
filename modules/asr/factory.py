"""ASR adapter factory — create ASR backends by name."""

from .base import ASRAdapter
from .fish_asr_adapter import FishAudioASRAdapter
from .chickenrice_adapter import ChickenRiceASRAdapter
from .common_asr_adapter import CommonASRAdapter
from .qwen_asr_adapter import FunASRAdapter, QwenAudio31ASRFlashAdapter
from .mimo_asr_adapter import MimoASRAdapter


# Registry of available backends — add new ones here
_BUILTIN_BACKENDS: dict[str, type[ASRAdapter]] = {
    "fish-asr": FishAudioASRAdapter,
    "chickenrice": ChickenRiceASRAdapter,
    "common-asr": CommonASRAdapter,
    "qwen-audio-3.1-asr-flash": QwenAudio31ASRFlashAdapter,
    "fun-asr": FunASRAdapter,
    "mimo-asr": MimoASRAdapter,
}

_registry: dict[str, type[ASRAdapter]] = {}


def _ensure_registry():
    if not _registry:
        _registry.update(_BUILTIN_BACKENDS)


def register_asr_backend(name: str, adapter_cls: type[ASRAdapter]):
    """Register a custom ASR backend."""
    _ensure_registry()
    _registry[name] = adapter_cls


def get_available_asr_backends() -> list[str]:
    """List registered backend names."""
    _ensure_registry()
    return list(_registry.keys())


def create_asr(backend: str = "fish-asr", **kwargs) -> ASRAdapter:
    """Create an ASR adapter instance.

    Args:
        backend: Backend name, e.g. "fish-asr".
        **kwargs: Additional backend-specific arguments.

    Returns:
        An ASRAdapter instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    _ensure_registry()
    cls = _registry.get(backend)
    if cls is None:
        supported = ", ".join(_registry.keys()) if _registry else "none registered"
        raise ValueError(
            f"Unknown ASR backend: '{backend}'. "
            f"Available: {supported}"
        )
    return cls(**kwargs)
