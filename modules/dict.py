from enum import Enum


class SpeechRate(Enum):
    """Speaking-style tiers in Chinese characters per second."""
    NORMAL = 2.5
    SOFT = 2.2
    WHISPER = 1.9
    SLOW = 1.7
    FAST = 3.2
    EXCITED = 3.0
