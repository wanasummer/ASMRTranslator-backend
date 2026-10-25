"""Speech-atom adapters for the detect stage."""

from .contracts import DetectError, SpeechAtomDetector
from ..vad.contracts import VadRegion


class NativeVadAtomDetector:
    def __init__(self, audio_path: str, config=None, *, loader=None, native_factory=None):
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


class ASMRVadAtomDetector:
    def __init__(self, audio_path, config):
        from ..vad.plugin import ASMRPluginVAD
        self.audio_path = audio_path
        self.engine = ASMRPluginVAD({"endpoint": config.vad_endpoint, "threshold": config.vad_threshold,
                                     "max_single_segment_time": 300000})

    def detect(self, parent, threshold_ms):
        start, end = parent.speech_start_ms, parent.speech_end_ms
        regions = self.engine.detect_sync(self.audio_path, start_ms=start, end_ms=end, silence_ms=threshold_ms)
        return tuple(VadRegion(f"{parent.id}-asmr-atom-{index}", start + round(row.start * 1000),
                              start + round(row.end * 1000), "detect-stage", threshold_ms)
                     for index, row in enumerate(regions))


ATOM_DETECTORS = {"native-vad": NativeVadAtomDetector, "asmr-vad": ASMRVadAtomDetector}


def create_atom_detector(name: str, audio_path: str, config=None) -> SpeechAtomDetector:
    try:
        return ATOM_DETECTORS[name](audio_path, config)
    except KeyError as exc:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN", f"Unknown atom detector: {name}") from exc
