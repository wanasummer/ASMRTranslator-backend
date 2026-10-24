"""Voice-activity detection preprocessing for ASR.

The VAD stage runs before the ASR transcribe step: it finds real speech
intervals with FSMN-VAD (FunASR), splits the track at the configured pause
interval (1500 ms) and force-cuts runs past the max segment length (25000 ms),
then the ASR orchestration shapes the raw regions into slices and feeds them
to the transcribe pipeline.

Public API:
    from modules.vad import create_vad, VADEngine, SpeechRegion

    vad = create_vad(cfg)   # returns a NoOpVAD (available=False) when unusable
    regions = await vad.detect("audio.wav")

Data models (``models``) and the pure slice-optimization functions
(``segments``) have no funasr/torch dependency and are testable offline.
"""

from ..error import VADUnavailableError
from .engine import FSMNVAD, NoOpVAD, VADEngine, create_vad
from .models import SpeechRegion
from .segments import optimize_asmr_segments, split_long_segment

__all__ = [
    "VADEngine",
    "NoOpVAD",
    "FSMNVAD",
    "create_vad",
    "SpeechRegion",
    "VADUnavailableError",
    "optimize_asmr_segments",
    "split_long_segment",
]
