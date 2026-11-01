"""Local ASMR VAD adapter; model dependencies stay in the installed plugin."""
import asyncio
import math
from typing import Any

from ..analysis_plugins import validate_timeline
from .local_runtime import LocalVADClient
from ..error import VadError, descriptor
from .base import VADEngine
from .models import SpeechRegion
from .asmr_policy import ASMRVadPolicy, ASMR_VAD_POLICY_VERSION, DEFAULT_ASMR_VAD_POLICY


class ASMRPluginVAD(VADEngine):
    def __init__(self, config: dict[str, Any], *, client=None,
                 policy: ASMRVadPolicy = DEFAULT_ASMR_VAD_POLICY):
        self.config = self.configure(config, policy=policy)
        self.client = client if client is not None else LocalVADClient()

    @staticmethod
    def configure(config: dict[str, Any], *,
                  policy: ASMRVadPolicy = DEFAULT_ASMR_VAD_POLICY) -> dict[str, Any]:
        """Return effective stage settings without mutating caller configuration.

        Generic VAD controls remain upper bounds; model-specific defaults live
        here. Inject a policy to customize ASMR splitting, independently of FSMN.
        """
        result = dict(config)
        threshold = result.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
            raise ValueError("ASMR VAD threshold must be between 0 and 1")
        purpose = result.get("purpose", "initial")
        if purpose not in {"initial", "detect"}:
            raise ValueError("Unknown ASMR VAD detection purpose")
        for key, low, high in (("max_end_silence_time", 20, 30000),
                               ("max_single_segment_time", 100, 300000)):
            value = result.get(key)
            if value is not None and (type(value) is not int or not low <= value <= high):
                raise ValueError(f"Invalid ASMR VAD {key}")
        for key in ("min_sec", "target_min_sec", "target_max_sec", "max_sec",
                    "max_merge_gap_sec", "split_window_sec", "split_overlap_sec"):
            value = result.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0
                                      or value == 0 and key not in {"max_merge_gap_sec", "split_overlap_sec"}):
                raise ValueError(f"Invalid ASMR VAD {key}")
        silence = policy.detect_silence_ms if purpose == "detect" else policy.initial_silence_ms
        maximum = policy.detect_max_segment_ms if purpose == "detect" else policy.initial_max_segment_ms
        result["max_end_silence_time"] = min(result.get("max_end_silence_time", silence), silence)
        result["max_single_segment_time"] = min(result.get("max_single_segment_time", maximum), maximum)
        if purpose == "initial":
            max_sec = result["max_single_segment_time"] / 1000
            result["max_sec"] = min(result.get("max_sec", max_sec), max_sec)
            result["target_max_sec"] = min(result.get("target_max_sec", max_sec), result["max_sec"])
            result["target_min_sec"] = min(result.get("target_min_sec", 5), result["target_max_sec"])
            result["max_merge_gap_sec"] = min(result.get("max_merge_gap_sec", policy.max_merge_gap_sec), policy.max_merge_gap_sec)
            result["split_window_sec"] = min(result.get("split_window_sec", max_sec), result["max_sec"])
            result["split_overlap_sec"] = min(result.get("split_overlap_sec", 1), result["split_window_sec"] / 4)
        result["min_speech_ms"] = policy.min_speech_ms
        result["asmr_policy_version"] = ASMR_VAD_POLICY_VERSION
        return result

    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        try:
            return await asyncio.to_thread(self.detect_sync, audio_path)
        except Exception as error:
            raise VadError(descriptor("VAD_INFERENCE_FAILED"), detail=str(error), cause=error,
                           context={"backend": "asmr-vad"}) from error

    def detect_sync(self, audio_path: str, *, start_ms: int | None = None,
                    end_ms: int | None = None, silence_ms: int | None = None) -> list[SpeechRegion]:
        body, duration = self.client.request("/detect", audio_path, {
            "threshold": self.config.get("threshold", 0.5),
            "min_silence_ms": self.config["max_end_silence_time"] if silence_ms is None else silence_ms,
            "max_segment_ms": self.config["max_single_segment_time"],
            "min_speech_ms": self.config["min_speech_ms"],
        }, start_ms=start_ms, end_ms=end_ms)
        return [SpeechRegion(row["start_ms"] / 1000, row["end_ms"] / 1000)
                for row in validate_timeline(body, "segments", duration)]
