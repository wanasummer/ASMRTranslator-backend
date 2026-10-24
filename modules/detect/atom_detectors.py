"""Speech-atom adapters for the detect stage."""

from .contracts import DetectError, SpeechAtomDetector
from ..vad.contracts import VadRegion


class NativeVadAtomDetector:
    def __init__(self, audio_path: str, *, loader=None, native_factory=None):
        self.audio_path = audio_path
        self.loader = loader
        self.native_factory = native_factory
        self._samples = None
        self._native = None

    def detect(self, parent, threshold_ms):
        try:
            from ..vad.native import NativeVAD, load_wav_16k_mono
            if self._samples is None:
                self._samples = (self.loader or load_wav_16k_mono)(self.audio_path)
                self._native = (self.native_factory or NativeVAD)()
            start, end = parent.speech_start_ms, parent.speech_end_ms
            if start < 0 or end * 16 > len(self._samples):
                raise DetectError("DETECT_ALIGNMENT_INVALID", "Parent exceeds source audio")
            regions = self._native.run(
                self._samples[start * 16:end * 16], max_seg_ms=end - start + 1,
                max_end_silence_ms=threshold_ms, strict=True)
            return tuple(VadRegion(
                f"{parent.id}-atom-{threshold_ms}-{round(region.start * 1000)}",
                start + round(region.start * 1000), start + round(region.end * 1000),
                "detect-stage", threshold_ms) for region in regions)
        except DetectError:
            raise
        except Exception as exc:
            raise DetectError("DETECT_ALIGNMENT_FAILED", "Native VAD atom detection failed") from exc


ATOM_DETECTORS = {"native-vad": NativeVadAtomDetector}


def create_atom_detector(name: str, audio_path: str) -> SpeechAtomDetector:
    try:
        return ATOM_DETECTORS[name](audio_path)
    except KeyError as exc:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN", f"Unknown atom detector: {name}") from exc
