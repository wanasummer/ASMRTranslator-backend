"""Speech-atom adapters for the detect stage."""

from .contracts import DetectError, SpeechAtomDetector
from ..vad.contracts import VadRegion
from ..vad.asmr_policy import ASMRVadPolicy, DEFAULT_ASMR_VAD_POLICY


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
    def __init__(self, audio_path: str, config=None, *, client=None,
                 policy: ASMRVadPolicy = DEFAULT_ASMR_VAD_POLICY):
        from ..vad.plugin import ASMRPluginVAD
        from .contracts import DetectConfig
        config = config if config is not None else DetectConfig()
        self.audio_path = audio_path
        self.engine = ASMRPluginVAD({"threshold": config.vad_threshold, "purpose": "detect"},
                                    client=client, policy=policy)

    def detect(self, parent, threshold_ms):
        start, end = parent.speech_start_ms, parent.speech_end_ms
        # Fine ASMR segmentation owns its silence policy, independent of FSMN.
        try:
            regions = self.engine.detect_sync(self.audio_path, start_ms=start, end_ms=end)
        except Exception as error:
            raise DetectError("DETECT_ALIGNMENT_FAILED", "ASMR VAD atom detection failed") from error
        return tuple(VadRegion(f"{parent.id}-asmr-atom-{index}", start + round(row.start * 1000),
                              start + round(row.end * 1000), "detect-stage",
                              self.engine.config["max_end_silence_time"])
                     for index, row in enumerate(regions))


ATOM_DETECTORS = {"native-vad": NativeVadAtomDetector, "asmr-vad": ASMRVadAtomDetector}

ATOM_BACKENDS = {"fsmn-vad": "native-vad", "asmr-vad": "asmr-vad"}


def atom_detector_for_backend(backend: str) -> str:
    """Use the selected voice detector in the joint-detect stage as well."""
    try:
        return ATOM_BACKENDS[backend]
    except KeyError as error:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN", f"Unknown VAD backend: {backend}") from error


def create_atom_detector(name: str, audio_path: str, config=None) -> SpeechAtomDetector:
    try:
        factory = ATOM_DETECTORS[name]
    except KeyError as exc:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN", f"Unknown atom detector: {name}") from exc
    return factory(audio_path, config)
