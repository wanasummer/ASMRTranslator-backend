"""Japanese reading-length estimates used by local sentence alignment."""

from dataclasses import dataclass
import unicodedata

from .contracts import (
    DetectError,
    JapaneseToken,
    SentenceDurationEstimator,
)


_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮヵヶ")
_KANA = (
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
)


def count_japanese_mora(reading: str, surface: str = "") -> float:
    """Estimate spoken units conservatively when UniDic reading is unavailable."""
    value = reading if reading and reading != "*" else surface
    mora = 0.0
    for char in value:
        codepoint = ord(char)
        if char in _SMALL_KANA:
            continue
        if any(start <= codepoint <= end for start, end in _KANA):
            mora += 1.0
        elif "\u4e00" <= char <= "\u9fff":
            # Unknown kanji commonly expands to roughly two kana. This fallback
            # affects only ASR terms UniDic could not read.
            mora += 2.0
        elif char.isdigit():
            mora += 1.5
        elif unicodedata.category(char)[0] in ("L", "N"):
            mora += 1.0
    return mora


@dataclass(frozen=True)
class JapaneseMoraDurationEstimator:
    """Return relative speech load; the optimizer calibrates it per ASR slice."""

    unknown_token_discount: float = 0.82

    def estimate_units(self, tokens) -> float:
        total = 0.0
        for token in tokens:
            units = token.mora or count_japanese_mora(token.reading, token.surface)
            total += units * (self.unknown_token_discount if token.unknown else 1.0)
        return max(0.0, total)


DURATION_ESTIMATORS = {"japanese-mora": JapaneseMoraDurationEstimator}


def create_duration_estimator(name: str) -> SentenceDurationEstimator:
    try:
        return DURATION_ESTIMATORS[name]()
    except KeyError as exc:
        raise DetectError(
            "DETECT_IMPLEMENTATION_UNKNOWN",
            f"Unknown duration estimator: {name}",
        ) from exc
