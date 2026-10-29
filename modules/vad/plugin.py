"""Local ASMR VAD adapter; model dependencies stay in the installed plugin."""
import asyncio
from typing import Any

from ..analysis_plugins import validate_timeline
from .local_runtime import LocalVADClient
from ..error import VadError, descriptor
from .engine import VADEngine
from .models import SpeechRegion


class ASMRPluginVAD(VADEngine):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client = LocalVADClient()

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
            "min_silence_ms": silence_ms or self.config.get("max_end_silence_time", 1500),
            "max_segment_ms": self.config.get("max_single_segment_time", 25000),
            "min_speech_ms": 100,
        }, start_ms=start_ms, end_ms=end_ms)
        return [SpeechRegion(row["start_ms"] / 1000, row["end_ms"] / 1000)
                for row in validate_timeline(body, "segments", duration)]
