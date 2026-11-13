"""Configuration contract for the standalone ASR-boost detect channel."""

from __future__ import annotations

from dataclasses import dataclass, replace

ASR_BOOST_POLICY_VERSION = 1


@dataclass(frozen=True)
class AsrBoostConfig:
    """User and plugin parameters governing ASR-boost sentence boundary refinement."""

    enabled: bool = True
    padding_ms: int = 250
    min_duration_ms: int = 280
    backend: str = "fish-asr"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.enabled, bool)
            or isinstance(self.padding_ms, bool)
            or isinstance(self.min_duration_ms, bool)
            or not isinstance(self.padding_ms, int)
            or not isinstance(self.min_duration_ms, int)
            or self.padding_ms <= 0
            or self.min_duration_ms <= 0
            or not str(self.backend or "").strip()
        ):
            raise ValueError("Invalid ASR-boost detect configuration")

    def with_plugin_overrides(self, plugin_config: dict | None) -> "AsrBoostConfig":
        """Apply plugin-declared hook parameters when present and valid."""
        if not isinstance(plugin_config, dict):
            return self
        enabled = plugin_config.get("enabled", self.enabled)
        padding_ms = plugin_config.get("padding_ms", plugin_config.get("paddingMs", self.padding_ms))
        min_duration_ms = plugin_config.get(
            "min_duration_ms",
            plugin_config.get("minDurationMs", self.min_duration_ms),
        )
        return replace(
            self,
            enabled=bool(enabled),
            padding_ms=int(padding_ms),
            min_duration_ms=int(min_duration_ms),
        )
