"""Voice-activity detection preprocessing for ASR.

The selected adapter finds speech before ASR. Native FSMN uses its configured
pause interval and duration cap; the local ASMR plugin owns separate coarse and
fine segmentation policies. Slice optimization consumes their effective config.

Public API:
    from modules.vad import create_vad, VADEngine, SpeechRegion

    vad = create_vad(cfg)   # returns a NoOpVAD (available=False) when unusable
    regions = await vad.detect("audio.wav")

Data models (``models``) and the pure slice-optimization functions
(``segments``) have no funasr/torch dependency and are testable offline.
"""

from ..error import VADUnavailableError
from .base import NoOpVAD, VADEngine
from .engine import FSMNVAD
from .factory import create_vad, resolve_vad_config
from .models import SpeechRegion
from .segments import optimize_asmr_segments, split_long_segment

__all__ = [
    "VADEngine",
    "NoOpVAD",
    "FSMNVAD",
    "create_vad",
    "resolve_vad_config",
    "SpeechRegion",
    "VADUnavailableError",
    "optimize_asmr_segments",
    "split_long_segment",
]
