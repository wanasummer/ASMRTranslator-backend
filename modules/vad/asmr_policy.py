"""ASMR VAD's own coarse-slice and fine-atom policies (milliseconds)."""

from dataclasses import dataclass


ASMR_VAD_POLICY_VERSION = 1


@dataclass(frozen=True)
class ASMRVadPolicy:
    """Tune the plugin independently of FSMN's segmentation defaults.

    Initial detection produces ASR-sized windows. Fine detection owns only
    acoustic atoms; its parent crop, rather than ASR's 20s cap, bounds length.
    """

    initial_silence_ms: int = 600
    initial_max_segment_ms: int = 20000
    detect_silence_ms: int = 200
    detect_max_segment_ms: int = 300000
    min_speech_ms: int = 100
    max_merge_gap_sec: float = 0.6

    def __post_init__(self):
        ranges = ((self.initial_silence_ms, 20, 30000),
                  (self.detect_silence_ms, 20, 30000),
                  (self.initial_max_segment_ms, 100, 300000),
                  (self.detect_max_segment_ms, 100, 300000),
                  (self.min_speech_ms, 20, 5000))
        if (any(type(value) is not int or not low <= value <= high
                for value, low, high in ranges)
                or not 0 <= self.max_merge_gap_sec <= 30):
            raise ValueError("Invalid ASMR VAD splitting policy")


DEFAULT_ASMR_VAD_POLICY = ASMRVadPolicy()
