"""Native FSMN-VAD backend — ctypes bridge to the extracted C++ DLL.

The C++ port (``modules/vad/cpp/``, built from the fsmn-vad-tool extraction)
runs the same fsmn-vad model on ggml without funasr/torch: it loads a 1.7 MB
GGUF weight file and returns ``[start_ms, end_ms]`` speech spans. This module
wraps that DLL via ctypes and mirrors the ``FSMNVAD`` engine contract so the
pipeline keeps its existing ``vad_*`` parameters:

    max_single_segment_time  ->  max_seg_ms
    max_end_silence_time     ->  max_end_silence_ms  (fixed pause-cut threshold)

The DLL is optional: when either the DLL or the GGUF is missing (e.g. the
user has not built the native component yet), ``available()`` is False and
the engine falls back to funasr.
"""

import os
import sys
import ctypes
from ctypes import POINTER, byref, c_char_p, c_float, c_int

from .models import SpeechRegion

# Bundled GGUF weights (shipped with the package).
_GGUF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "fsmn-vad.gguf"
)


def _candidate_dll_paths() -> list[str]:
    """Where the built ``fsmn_vad.dll`` may live (source layout + frozen EXE)."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "cpp", "build", "bin", "fsmn_vad.dll"),
        os.path.join(base, "native", "fsmn_vad.dll"),
    ]
    if getattr(sys, "frozen", False):  # PyInstaller onefile: _MEIPASS
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(os.path.join(meipass, "modules", "vad", "native", "fsmn_vad.dll"))
    return candidates


def _find_dll() -> str | None:
    for path in _candidate_dll_paths():
        if os.path.isfile(path):
            return path
    return None


def native_available() -> bool:
    """True when both the built DLL and the GGUF weights are present."""
    return _find_dll() is not None and os.path.isfile(_GGUF_PATH)


class NativeVAD:
    """ctypes wrapper around the C API ``funasr_vad``."""

    def __init__(self, dll_path: str | None = None):
        self.dll_path = dll_path or _find_dll()
        if not self.dll_path or not os.path.isfile(self.dll_path):
            raise FileNotFoundError(f"fsmn_vad.dll not found: {self.dll_path}")
        self._lib = ctypes.CDLL(self.dll_path)
        fn = self._lib.funasr_vad
        fn.restype = c_int
        fn.argtypes = [
            c_char_p,              # gguf_path
            POINTER(c_float),      # wav_16k
            c_int,                 # nsamples
            c_int,                 # max_seg_ms
            c_int,                 # max_end_silence_ms
            POINTER(c_int),        # segs_ms_out
            c_int,                 # out_capacity (pairs)
            c_int,                 # nthreads
        ]
        self._fn = fn

    def run(
        self,
        wav_16k: list[float],
        max_seg_ms: int,
        max_end_silence_ms: int,
        strict: bool = False,
    ) -> list[SpeechRegion]:
        """Run VAD on a 16 kHz mono float waveform.

        Returns speech regions as absolute ``(start, end)`` seconds, matching
        the funasr engine's ``detect`` output.
        """
        if not wav_16k:
            return []
        n = len(wav_16k)
        arr = (c_float * n)(*wav_16k)
        # Enough capacity for any realistic clip (1 pair per 1.6 s of audio).
        cap = max(4096, n // 1600 + 8)
        segs = (c_int * (cap * 2))()
        cnt = self._fn(
            _GGUF_PATH.encode("utf-8"),
            arr,
            n,
            max_seg_ms,
            max_end_silence_ms,
            segs,
            cap,
            8,
        )
        if cnt < 0:
            raise RuntimeError(f"native fsmn-vad failed with code {cnt}")
        if strict and cnt > cap:
            raise ValueError("Native VAD output exceeded capacity")
        regions: list[SpeechRegion] = []
        for i in range(min(cnt, cap)):
            start_ms, end_ms = segs[i * 2], segs[i * 2 + 1]
            if strict and (start_ms < 0 or end_ms <= start_ms or end_ms > n / 16):
                from .contracts import VadAnalysisError
                raise VadAnalysisError("VAD_RESULT_INVALID", "Invalid native VAD region")
            if end_ms > start_ms:
                regions.append(SpeechRegion(start_ms / 1000.0, end_ms / 1000.0))
        return regions


def load_wav_16k_mono(audio_path: str) -> list[float]:
    """Decode any audio file to 16 kHz mono float samples.

    Uses the pipeline's bundled ffmpeg (via pydub) so no extra audio library
    is required; mp3/wav/m4a etc. all work.
    """
    from pydub import AudioSegment
    from ..audio_splitter import get_ffmpeg_path

    AudioSegment.converter = get_ffmpeg_path()
    seg = AudioSegment.from_file(audio_path)
    seg = seg.set_frame_rate(16000).set_channels(1)
    return [s / 32768.0 for s in seg.get_array_of_samples()]
