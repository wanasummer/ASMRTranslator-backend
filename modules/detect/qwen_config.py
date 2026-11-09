"""Configuration owned by the Qwen word-alignment pipeline.

FSMN atom detection, mora estimates and joint partition scores do not enter
this configuration. Version changes invalidate completed alignment caches.
"""

from dataclasses import dataclass
import math


QWEN_PIPELINE_VERSION = 3


@dataclass(frozen=True)
class QwenConfig:
    endpoint: str = "http://127.0.0.1:8007"
    min_sentence_duration_ms: int = 280
    max_sentence_duration_ms: int = 12000
    max_sentence_count: int = 12
    boundary_bias: float = 0.72
    language_weight: float = 0.55
    acoustic_weight: float = 0.75
    boundary_synergy_weight: float = 0.30
    vocalization_chars: str = "嗯啊阿哈哦喔呃唔呀哼欸诶あぁうぅんはぁえぇおぉっッー"

    def __post_init__(self):
        sizes = (self.min_sentence_duration_ms, self.max_sentence_duration_ms,
                 self.max_sentence_count)
        weights = (self.boundary_bias, self.language_weight,
                   self.acoustic_weight, self.boundary_synergy_weight)
        if (any(type(value) is not int or value <= 0 for value in sizes)
                or self.max_sentence_duration_ms < self.min_sentence_duration_ms
                or any(not math.isfinite(value) or value < 0 for value in weights)):
            raise ValueError("Invalid Qwen pipeline configuration")
