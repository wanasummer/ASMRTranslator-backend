"""Unified cleaning domain with independently toggleable capabilities (capXxx)."""

from .artifacts import write_cleaning_artifacts
from .capabilities import (
    InterjectionSkipCapability,
    JapaneseCleanCapability,
    KEEP_GREETING_SURFACES,
    MecabRegexJapaneseCleaner,
    MecabTokenizer,
    RegexJapaneseCleaner,
    RegexTokenizer,
    count_cjk_chars,
    create_japanese_cleaner,
    mark_interjection_skips,
    register_japanese_cleaner,
)
from .config import InterjectionSkipConfig, JapaneseCleanConfig
from .contracts import (
    CleaningRecord,
    InterjectionRecord,
    InterjectionSkipResult,
    JapaneseCleaner,
    JapaneseCleanerUnavailableError,
    JapaneseCleaningResult,
    MorphToken,
    MorphTokenizer,
)

__all__ = [
    "CleaningRecord",
    "InterjectionRecord",
    "InterjectionSkipCapability",
    "InterjectionSkipConfig",
    "InterjectionSkipResult",
    "JapaneseCleanCapability",
    "JapaneseCleanConfig",
    "JapaneseCleaner",
    "JapaneseCleanerUnavailableError",
    "JapaneseCleaningResult",
    "KEEP_GREETING_SURFACES",
    "MecabRegexJapaneseCleaner",
    "MecabTokenizer",
    "MorphToken",
    "MorphTokenizer",
    "RegexJapaneseCleaner",
    "RegexTokenizer",
    "count_cjk_chars",
    "create_japanese_cleaner",
    "mark_interjection_skips",
    "register_japanese_cleaner",
    "write_cleaning_artifacts",
]
