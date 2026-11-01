"""Composition root for native FSMN and local ASMR VAD adapters.

Each registration owns construction and effective settings. Orchestrators use
the same registration for cache identity, slice policy, and inference.
"""

from dataclasses import dataclass
from typing import Any, Callable

from ..error import VADUnavailableError
from .base import NoOpVAD, VADEngine


VadConfig = dict[str, Any]


@dataclass(frozen=True)
class VadAdapterRegistration:
    create: Callable[[VadConfig], VADEngine]
    configure: Callable[[VadConfig], VadConfig] = dict


def _create_fsmn_vad(config: VadConfig) -> VADEngine:
    from .engine import FSMNVAD

    try:
        return FSMNVAD(
            model=config.get("model", "fsmn-vad"),
            max_single_segment_time=config.get("max_single_segment_time", 25000),
            max_end_silence_time=config.get("max_end_silence_time", 1500),
            strict=config.get("strict", False),
        )
    except VADUnavailableError as error:
        return NoOpVAD(str(error))


def _create_asmr_vad(config: VadConfig) -> VADEngine:
    from .plugin import ASMRPluginVAD

    return ASMRPluginVAD(config)


def _configure_asmr_vad(config: VadConfig) -> VadConfig:
    from .plugin import ASMRPluginVAD

    return ASMRPluginVAD.configure(config)


VAD_ADAPTERS: dict[str, VadAdapterRegistration] = {
    "fsmn-vad": VadAdapterRegistration(_create_fsmn_vad),
    "asmr-vad": VadAdapterRegistration(_create_asmr_vad, _configure_asmr_vad),
}


def _registration(config: VadConfig) -> VadAdapterRegistration:
    backend = config.get("backend", "fsmn-vad")
    try:
        return VAD_ADAPTERS[backend]
    except KeyError as error:
        raise ValueError(f"Unknown VAD backend: {backend}") from error


def resolve_vad_config(config: VadConfig) -> VadConfig:
    """Return adapter-owned settings without mutating the caller's config."""
    return _registration(config).configure(dict(config))


def create_vad(config: VadConfig) -> VADEngine:
    """Build the selected adapter; plugin errors never switch to native FSMN.

    Disabled detection and an unavailable native component return NoOpVAD for
    the caller to handle. ASMR configuration and inference failures propagate.
    """
    if not config.get("enabled", True):
        return NoOpVAD("vad.enabled=false")
    registration = _registration(config)
    return registration.create(registration.configure(dict(config)))
