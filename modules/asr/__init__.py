"""ASR adapter system.

Usage:
    from modules.asr import create_asr, FishAudioASRAdapter

    asr = create_asr("fish-asr", api_key="...")
    result = await asr.transcribe("slice_000.wav")

Extending with a new backend:
    1. Create a new adapter class inheriting ASRAdapter
    2. Add it to _BUILTIN_BACKENDS in factory.py
    Done.
"""

from ..error import ASRError
from .base import ASRAdapter, DocumentASRAdapter
from .factory import (
    create_asr,
    register_asr_backend,
    get_available_asr_backends,
)
from .fish_asr_adapter import FishAudioASRAdapter
from .chickenrice_adapter import ChickenRiceASRAdapter, ChickenRiceSubtitleResult
from .common_asr_adapter import CommonASRAdapter, CommonASRDocumentResult
from .qwen_asr_adapter import FunASRAdapter, QwenAudio31ASRFlashAdapter
from .mimo_asr_adapter import MimoASRAdapter
from .models import ASRSegment, ASRResult, AudioChunk, SentenceSegment
from .no_speech import ASRNoSpeechError

__all__ = [
    "ASRAdapter",
    "DocumentASRAdapter",
    "ASRError",
    "ASRNoSpeechError",
    "ASRSegment",
    "ASRResult",
    "AudioChunk",
    "SentenceSegment",
    "create_asr",
    "register_asr_backend",
    "get_available_asr_backends",
    "FishAudioASRAdapter",
    "ChickenRiceASRAdapter",
    "ChickenRiceSubtitleResult",
    "CommonASRAdapter",
    "CommonASRDocumentResult",
    "QwenAudio31ASRFlashAdapter",
    "FunASRAdapter",
    "MimoASRAdapter",
]
