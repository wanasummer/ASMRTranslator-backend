"""Independent policy for refining native MOSS subtitle blocks."""
from dataclasses import dataclass


MOSS_POLICY_VERSION = 2


@dataclass(frozen=True)
class MossDetectConfig:
    min_sentence_duration_ms: int = 280
    alignment_endpoint: str | None = None
    recalibrate: bool = False
    recalibrate_endpoint: str | None = None
    min_recalibrate_duration_ms: int = 1500

    def __post_init__(self):
        if type(self.min_sentence_duration_ms) is not int or self.min_sentence_duration_ms <= 0:
            raise ValueError("Invalid MOSS minimum sentence duration")
        if type(self.min_recalibrate_duration_ms) is not int or self.min_recalibrate_duration_ms <= 0:
            raise ValueError("Invalid MOSS minimum recalibrate duration")
