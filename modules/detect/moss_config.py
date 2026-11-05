"""Independent policy for refining native MOSS subtitle blocks."""
from dataclasses import dataclass


MOSS_POLICY_VERSION = 1


@dataclass(frozen=True)
class MossDetectConfig:
    alignment_endpoint: str = "http://127.0.0.1:8007"
    min_sentence_duration_ms: int = 280

    def __post_init__(self):
        if type(self.min_sentence_duration_ms) is not int or self.min_sentence_duration_ms <= 0:
            raise ValueError("Invalid MOSS minimum sentence duration")


