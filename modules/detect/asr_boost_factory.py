"""Standalone factory and hook registry for the ASR-boost detect channel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol, Sequence

from ..vad.contracts import AsrSlice
from .asr_boost_config import ASR_BOOST_POLICY_VERSION, AsrBoostConfig


@dataclass(frozen=True)
class AsrBoostSpan:
    """One timestamped word or sub-clause span clamped to the parent speech timeline."""

    start_ms: int
    end_ms: int
    text: str


class AsrBoostAdapter(Protocol):
    """Per-ASR-backend adapter extracting normalized spans and plugin hook settings."""

    backend: str

    def resolve_config(self, block: object, base: AsrBoostConfig) -> AsrBoostConfig: ...

    def extract_spans(self, block: object, parent: AsrSlice) -> tuple[AsrBoostSpan, ...]: ...


def normalize_boost_spans(
    raw_spans: Sequence[object] | None,
    parent: AsrSlice,
) -> tuple[AsrBoostSpan, ...]:
    """Clamp raw ASR spans into [parent.speech_start_ms, parent.speech_end_ms] monotonically."""
    if not raw_spans:
        return ()
    lower = parent.speech_start_ms
    upper = parent.speech_end_ms
    if upper <= lower:
        return ()
    spans: list[AsrBoostSpan] = []
    previous_start = lower
    for raw in raw_spans:
        if isinstance(raw, AsrBoostSpan):
            start_val, end_val, text = raw.start_ms, raw.end_ms, raw.text
        elif isinstance(raw, dict):
            start_val = raw.get("start_ms")
            end_val = raw.get("end_ms")
            text = str(raw.get("text") or "")
        else:
            start_val = getattr(raw, "start_ms", None)
            end_val = getattr(raw, "end_ms", None)
            text = str(getattr(raw, "text", "") or "")
        text = text.strip()
        if not text or start_val is None or end_val is None:
            continue
        try:
            start_ms = int(round(float(start_val)))
            end_ms = int(round(float(end_val)))
        except (TypeError, ValueError):
            continue
        start_ms = max(previous_start, min(upper, start_ms))
        end_ms = max(start_ms, min(upper, end_ms))
        if end_ms <= start_ms:
            continue
        spans.append(AsrBoostSpan(start_ms=start_ms, end_ms=end_ms, text=text))
        previous_start = start_ms
    return tuple(spans)


class LocalSliceAsrBoostAdapter:
    """Built-in adapter for local slice-based ASR backends (fish-asr, fun-asr)."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def resolve_config(self, block: object, base: AsrBoostConfig) -> AsrBoostConfig:
        plugin_cfg = getattr(block, "asr_boost_config", None)
        return base.with_plugin_overrides(plugin_cfg)

    def extract_spans(self, block: object, parent: AsrSlice) -> tuple[AsrBoostSpan, ...]:
        raw = getattr(block, "asr_boost_spans", None)
        return normalize_boost_spans(raw, parent)


class CommonAsrPluginBoostAdapter:
    """Adapter reading hook configurations and spans from common-asr plugins."""

    backend = "common-asr"

    def resolve_config(self, block: object, base: AsrBoostConfig) -> AsrBoostConfig:
        plugin_cfg = getattr(block, "asr_boost_config", None)
        return base.with_plugin_overrides(plugin_cfg)

    def extract_spans(self, block: object, parent: AsrSlice) -> tuple[AsrBoostSpan, ...]:
        raw = getattr(block, "asr_boost_spans", None)
        return normalize_boost_spans(raw, parent)


class HookAsrBoostAdapter:
    """Adapter wrapping user-supplied hook functions for custom ASR backends."""

    def __init__(
        self,
        backend: str,
        span_hook: Callable[[object, AsrSlice], Sequence[object]] | None = None,
        config_hook: Callable[[object, AsrBoostConfig], dict | AsrBoostConfig | None] | None = None,
    ) -> None:
        self.backend = backend
        self._span_hook = span_hook
        self._config_hook = config_hook

    def resolve_config(self, block: object, base: AsrBoostConfig) -> AsrBoostConfig:
        resolved = base.with_plugin_overrides(getattr(block, "asr_boost_config", None))
        if self._config_hook is None:
            return resolved
        custom = self._config_hook(block, resolved)
        if isinstance(custom, AsrBoostConfig):
            return custom
        if isinstance(custom, dict):
            return resolved.with_plugin_overrides(custom)
        return resolved

    def extract_spans(self, block: object, parent: AsrSlice) -> tuple[AsrBoostSpan, ...]:
        if self._span_hook is not None:
            return normalize_boost_spans(self._span_hook(block, parent), parent)
        return normalize_boost_spans(getattr(block, "asr_boost_spans", None), parent)


_BUILTIN_BOOST_ADAPTERS: dict[str, Callable[[], AsrBoostAdapter]] = {
    "fish-asr": lambda: LocalSliceAsrBoostAdapter("fish-asr"),
    "fun-asr": lambda: LocalSliceAsrBoostAdapter("fun-asr"),
    "common-asr": CommonAsrPluginBoostAdapter,
}

_CUSTOM_BOOST_ADAPTERS: dict[str, Callable[[], AsrBoostAdapter]] = {}


def register_asr_boost_hook(
    backend: str,
    span_hook: Callable[[object, AsrSlice], Sequence[object]] | None = None,
    config_hook: Callable[[object, AsrBoostConfig], dict | AsrBoostConfig | None] | None = None,
) -> None:
    """Register a custom ASR-boost hook for a backend."""
    name = str(backend or "").strip()
    if not name:
        raise ValueError("ASR backend name must not be empty")
    _CUSTOM_BOOST_ADAPTERS[name] = lambda: HookAsrBoostAdapter(
        name,
        span_hook=span_hook,
        config_hook=config_hook,
    )


def asr_boost_supports(backend: str) -> bool:
    """Return True when the ASR backend supports the asr-boost detect channel."""
    return backend in _CUSTOM_BOOST_ADAPTERS or backend in _BUILTIN_BOOST_ADAPTERS


def has_custom_boost_hook(backend: str) -> bool:
    """Return True when a custom ASR-boost hook was explicitly registered for ``backend``."""
    return backend in _CUSTOM_BOOST_ADAPTERS


def create_asr_boost_adapter(backend: str) -> AsrBoostAdapter:
    """Instantiate the ASR-boost adapter for ``backend``."""
    factory = _CUSTOM_BOOST_ADAPTERS.get(backend) or _BUILTIN_BOOST_ADAPTERS.get(backend)
    if factory is None:
        from .contracts import DetectError
        raise DetectError(
            "DETECT_BACKEND_INCOMPATIBLE",
            f"Detect pipeline asr-boost does not support ASR backend {backend}",
        )
    return factory()


def asr_boost_cache_signature(config) -> dict:
    """Compute cache identity for the asr-boost detect channel."""
    settings = config.asr_boost_settings()
    return {
        "pipeline": "asr-boost",
        "version": ASR_BOOST_POLICY_VERSION,
        "settings": asdict(settings),
    }


def create_asr_boost_pre_stage(config):
    """Create the ASR-boost interval pre-stage when the ASR backend supports it."""
    from .asr_boost_stage import AsrBoostPreStage

    settings = config.asr_boost_settings()
    if not asr_boost_supports(settings.backend):
        return None
    adapter = create_asr_boost_adapter(settings.backend)
    return AsrBoostPreStage(settings, adapter=adapter)


def create_asr_boost_stage(config, audio_path: str):
    """Create the standalone ASR-boost detect stage."""
    from .asr_boost_stage import AsrBoostDetectStage

    settings = config.asr_boost_settings()
    adapter = create_asr_boost_adapter(settings.backend)
    return AsrBoostDetectStage(settings, audio_path, adapter=adapter)
