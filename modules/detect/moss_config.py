"""Independent policy for refining native MOSS subtitle blocks."""
from dataclasses import dataclass


MOSS_POLICY_VERSION = 3


@dataclass(frozen=True)
class MossDetectConfig:
    min_sentence_duration_ms: int = 280
    alignment_endpoint: str | None = None
    # Experimental: re-recognize both sides of each acoustic boundary with
    # MOSS; boundaries whose sides do not match their sentences are recomputed.
    recalibrate: bool = False
    recalibrate_endpoint: str | None = None
    min_verify_duration_ms: int = 800
    verify_similarity: float = 0.8
    max_verify_candidates: int = 3

    def __post_init__(self):
        if type(self.min_sentence_duration_ms) is not int or self.min_sentence_duration_ms <= 0:
            raise ValueError("Invalid MOSS minimum sentence duration")
        if type(self.min_verify_duration_ms) is not int or self.min_verify_duration_ms <= 0:
            raise ValueError("Invalid MOSS minimum verification duration")
        if not 0 < self.verify_similarity <= 1:
            raise ValueError("Invalid MOSS verification similarity")
        if type(self.max_verify_candidates) is not int or self.max_verify_candidates < 0:
            raise ValueError("Invalid MOSS verification candidate count")
