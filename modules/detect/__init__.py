"""Japanese sentence detection and audio alignment before cleanup/translation."""

from .contracts import DetectConfig, DetectError, DetectStageResult
from .factory import create_detect_stage, detect_cache_signature
from .qwen_config import QwenConfig
from .stage import detect_eligible

__all__ = ["DetectConfig", "DetectError", "DetectStageResult",
           "create_detect_stage", "detect_eligible", "detect_cache_signature", "QwenConfig"]
