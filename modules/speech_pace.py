"""Estimate where a Chinese voice should land from the original speech pace."""
from __future__ import annotations

import re

from .dict import SpeechRate

_CHINESE = re.compile(r"[\u4e00-\u9fff]")


def chinese_char_count(text: str) -> int:
    return len(_CHINESE.findall(text))


def classify_speech_rate(chars: int, original_ms: float) -> SpeechRate | None:
    """The tier closest to speaking ``chars`` Chinese characters within the original speech."""
    if chars <= 0 or original_ms <= 0:
        return None
    pace = chars / (original_ms / 1000)
    return min(SpeechRate, key=lambda tier: abs(tier.value - pace))


def estimate_tts_ms(text: str, original_ms: float) -> int | None:
    chars = chinese_char_count(text)
    tier = classify_speech_rate(chars, original_ms)
    return None if tier is None else round(chars / tier.value * 1000)


def landing_hint_ms(text: str, speech_start_ms: int, speech_end_ms: int) -> int | None:
    """Covers the original speech and the voice's estimated duration at its tier."""
    estimate = estimate_tts_ms(text, speech_end_ms - speech_start_ms)
    return None if estimate is None else max(speech_end_ms, speech_start_ms + estimate)
