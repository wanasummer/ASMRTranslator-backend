"""TTS adapter system.

Usage:
    from modules.tts import create_tts, EdgeTTSAdapter

    tts = create_tts("edge-tts", voice_map={"default": "zh-CN-XiaoxiaoNeural"})
    path = await tts.synthesize("你好", "zh-CN-XiaoxiaoNeural", "output.mp3")

Extending with a new backend:
    1. Create a new adapter class inheriting TTSAdapter
    2. Add it to _BUILTIN_BACKENDS in factory.py
    Done.
"""

from ..error import TTSError
from .models import VoiceInfo, TTSConfig
from .base import TTSAdapter
from .factory import create_tts, register_backend, get_available_backends
from .edge_tts_adapter import EdgeTTSAdapter
from .mimo_tts_adapter import MimoTTSAdapter
from .fish_audio_adapter import FishAudioTTSAdapter
from .minimax_tts_adapter import MiniMaxTTSAdapter
from .indextts_adapter import IndexTTSAdapter

__all__ = [
    "VoiceInfo",
    "TTSConfig",
    "TTSAdapter",
    "TTSError",
    "create_tts",
    "register_backend",
    "get_available_backends",
    "EdgeTTSAdapter",
    "MimoTTSAdapter",
    "FishAudioTTSAdapter",
    "MiniMaxTTSAdapter",
    "IndexTTSAdapter",
]
