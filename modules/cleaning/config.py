"""Validated, frontend-toggleable configurations for each cleaning capability."""

from __future__ import annotations

from dataclasses import dataclass

_DISABLED_CLEANER_NAMES = frozenset({"none", "off", "disabled", "false"})


@dataclass(frozen=True)
class JapaneseCleanConfig:
    """Configuration for pre-translation Japanese ASR cleaning (capJapaneseClean)."""

    enabled: bool = True
    cleaner: str = "mecab-regex"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("JapaneseCleanConfig.enabled 必须是布尔值")
        if not isinstance(self.cleaner, str) or not self.cleaner.strip():
            raise ValueError("JapaneseCleanConfig.cleaner 必须是非空字符串")

    @property
    def is_enabled(self) -> bool:
        return self.enabled and self.cleaner.strip().lower() not in _DISABLED_CLEANER_NAMES


@dataclass(frozen=True)
class InterjectionSkipConfig:
    """Configuration for post-translation Chinese short-interjection skip (capInterjectionSkip)."""

    enabled: bool = True
    max_chars: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("InterjectionSkipConfig.enabled 必须是布尔值")
        if isinstance(self.max_chars, bool) or not isinstance(self.max_chars, int) or self.max_chars < 0:
            raise ValueError("InterjectionSkipConfig.max_chars 必须是非负整数")
