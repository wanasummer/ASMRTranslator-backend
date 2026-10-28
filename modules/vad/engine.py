"""Native FSMN and external ASMR VAD engines, plus the unavailable fallback.

VAD runs entirely through the bundled native component (``modules/vad/native``):
a ctypes wrapper around the compiled ``fsmn_vad.dll`` (C++ port of FSMN-VAD,
sources under ``modules/vad/cpp``) plus the GGUF weights (``model/fsmn-vad.gguf``,
~1.7 MB). There is no funasr/torch dependency — the torch fallback was removed
deliberately (it added ~400 MB to the packaged EXE).

``create_vad`` returns a ``NoOpVAD`` (``available=False``) whenever the native
component is missing (no DLL or no GGUF) or fails to load. The caller (the ASR
orchestration) decides whether to error out on an unavailable engine.

FSMN-VAD performs native sentence segmentation: ``detect`` returns the
``[start_ms, end_ms]`` voice segments the model emits directly, split where a
pause exceeds ``max_end_silence_time`` (interval) and force-cut past
``max_single_segment_time`` (max length).
"""

import asyncio
from abc import ABC, abstractmethod

from ..error import VADUnavailableError
from .models import SpeechRegion


def _load_native_vad():
    """Build the ctypes native backend when the DLL + GGUF are present.

    Isolated so tests can patch it (and so the heavy pydub/ctypes import is
    lazy). Returns ``None`` when the native component is not built yet.
    """
    from .native import NativeVAD, native_available

    if not native_available():
        return None
    return NativeVAD()


class VADEngine(ABC):
    """Detect speech regions in an audio file."""

    @property
    def available(self) -> bool:
        """Whether this engine can actually run detection (True for a live one)."""
        return True

    @abstractmethod
    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        """Return raw speech regions as absolute ``(start, end)`` seconds."""
        ...


class NoOpVAD(VADEngine):
    """Placeholder engine — never runs; signals VAD is unavailable/disabled."""

    def __init__(self, reason: str = "VAD 未启用"):
        self.reason = reason

    @property
    def available(self) -> bool:
        return False

    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        return []


class FSMNVAD(VADEngine):
    """FSMN-VAD sentence segmentation via the bundled native C++ backend.

    The native backend (``fsmn_vad.dll`` + ``fsmn-vad.gguf``, no funasr/torch)
    runs on 16 kHz audio and splits the track at pauses longer than
    ``max_end_silence_time`` ms, force-cutting any run that exceeds
    ``max_single_segment_time`` ms — so a single returned region never spans
    more than the max length.

    ``model`` and ``device`` are kept for config compatibility but ignored:
    weights and compute backend are fixed by the bundled native component.
    """

    def __init__(
        self,
        model: str = "fsmn-vad",
        max_single_segment_time: int = 25000,
        max_end_silence_time: int = 1500,
        device: str = "cpu",
        strict: bool = False,
    ):
        self.max_single_segment_time = max_single_segment_time
        self.strict = strict
        self.max_end_silence_time = max_end_silence_time
        self.model = None  # no funasr model object — native backend only
        try:
            self._native = _load_native_vad()
        except Exception as e:  # noqa: BLE001 — surface as VADUnavailableError
            self._native = None
            raise VADUnavailableError(f"native VAD 后端加载失败: {e}") from e
        if self._native is None:
            raise VADUnavailableError(
                "native VAD 组件缺失：未找到 fsmn_vad.dll（需先编译 "
                "modules/vad/cpp）或 fsmn-vad.gguf"
            )

    async def detect(self, audio_path: str) -> list[SpeechRegion]:
        """Run FSMN-VAD on ``audio_path`` and return the voice segments.

        Inference is CPU-bound and can take seconds, so it runs on a worker
        thread to keep the event loop responsive.
        """
        return await asyncio.to_thread(self._detect_native, audio_path)

    def _detect_native(self, audio_path: str) -> list[SpeechRegion]:
        """Native backend inference: decode 16k mono → C++ VAD → regions."""
        try:
            from .native import load_wav_16k_mono

            wav = load_wav_16k_mono(audio_path)
            return self._native.run(
                wav,
                max_seg_ms=self.max_single_segment_time,
                max_end_silence_ms=self.max_end_silence_time,
                **({"strict": True} if self.strict else {}),
            )
        except Exception as error:
            from ..error import VadError, descriptor
            raise VadError(
                descriptor("VAD_INFERENCE_FAILED"), detail=str(error), cause=error,
                context={"backend": "fsmn-vad"},
            ) from error


def create_vad(cfg: dict) -> VADEngine:
    """Build the VAD engine for a config dict, falling back to ``NoOpVAD``.

    Returns a ``NoOpVAD`` (``available=False``) when VAD is disabled in the
    config or the native component (fsmn_vad.dll / fsmn-vad.gguf) is missing
    or fails to load. Plugin configuration errors are raised at construction;
    local plugin inference errors are raised by the selected adapter.
    """
    if not cfg.get("enabled", True):
        return NoOpVAD("vad.enabled=false")
    backend = cfg.get("backend", "fsmn-vad")
    if backend == "asmr-vad":
        from .plugin import ASMRPluginVAD
        return ASMRPluginVAD(cfg)
    if backend != "fsmn-vad":
        raise ValueError(f"Unknown VAD backend: {backend}")
    try:
        return FSMNVAD(
            model=cfg.get("model", "fsmn-vad"),
            max_single_segment_time=cfg.get("max_single_segment_time", 25000),
            max_end_silence_time=cfg.get("max_end_silence_time", 1500),
            strict=cfg.get("strict", False),
        )
    except VADUnavailableError as e:
        return NoOpVAD(str(e))
