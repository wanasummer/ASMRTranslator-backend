"""Japanese sentence detection and audio alignment before cleanup/translation."""

from .asr_boost_config import AsrBoostConfig
from .asr_boost_factory import asr_boost_supports, register_asr_boost_hook
from .asr_boost_stage import AsrBoostDetectStage
from .contracts import DetectConfig, DetectError, DetectStageInput, DetectStageResult
from .factory import (create_detect_stage, detect_cache_signature,
                      detect_pipeline_supports, validate_detect_selection)
from .qwen_config import QwenConfig
from .stage import detect_eligible

__all__ = ["AsrBoostConfig", "AsrBoostDetectStage", "DetectConfig", "DetectError",
           "DetectStageInput", "DetectStageResult", "asr_boost_supports",
           "create_detect_stage", "detect_eligible", "detect_cache_signature",
           "detect_pipeline_supports", "register_asr_boost_hook",
           "validate_detect_selection", "QwenConfig"]
