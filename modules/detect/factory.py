"""Composition root for the replaceable detect stage."""

from dataclasses import asdict, replace

from .aligners import create_sentence_aligner
from .atom_detectors import create_atom_detector
from .contracts import AlignmentDependencies, DetectConfig, DetectError, DetectStage
from .duration import create_duration_estimator
from .partitioners import create_region_partitioner
from .qwen_aligner import QwenForceAlignDetectStage
from .qwen_config import QWEN_PIPELINE_VERSION
from .sentence_detectors import create_sentence_detector
from .stage import DefaultDetectStage
from .moss_stage import MossDetectStage
from .moss_config import MOSS_POLICY_VERSION


def select_detect_config(config: DetectConfig, backend: str) -> DetectConfig:
    """Native document backends select their own refinement policy."""
    return replace(config, aligner=BACKEND_PIPELINES.get(backend, config.aligner))


BACKEND_PIPELINES = {"moss-transcribe-diarize": "moss-native"}


def _create_builtin_mecab_stage(config: DetectConfig, audio_path: str) -> DetectStage:
    sentence_detector = create_sentence_detector(config.sentence_detector)
    atom_detector = create_atom_detector(config.atom_detector, audio_path, config)
    estimator = create_duration_estimator(config.duration_estimator)
    partitioner = create_region_partitioner(config.region_partitioner, config)
    dependencies = AlignmentDependencies(
        audio_path,
        atom_detector,
        estimator,
        partitioner,
    )
    aligner = create_sentence_aligner(config.aligner, config, dependencies)
    return DefaultDetectStage(config, sentence_detector, aligner)


def _create_qwen_force_align_stage(config: DetectConfig, audio_path: str) -> DetectStage:
    return QwenForceAlignDetectStage(config.qwen_settings(), audio_path)


def _joint_cache_signature(config: DetectConfig) -> dict:
    settings = asdict(config)
    settings.pop("qwen")
    settings.pop("qwen_endpoint")
    settings.pop("moss")
    signature = {"pipeline": "joint-vad", "version": 1, "settings": settings}
    if config.atom_detector == "asmr-vad":
        from ..vad.factory import resolve_vad_config
        signature["vad_adapter"] = resolve_vad_config({
            "backend": "asmr-vad", "purpose": "detect", "threshold": config.vad_threshold,
        })
    return signature


def _qwen_cache_signature(config: DetectConfig) -> dict:
    return {"pipeline": "qwen-force-align", "version": QWEN_PIPELINE_VERSION,
            "settings": asdict(config.qwen_settings())}


STAGE_FACTORIES = {
    "moss-native": lambda config, audio_path: MossDetectStage(config.moss_settings(), audio_path),
    "joint-vad": _create_builtin_mecab_stage,
    "qwen-force-align": _create_qwen_force_align_stage,
}

CACHE_SIGNATURES = {
    "moss-native": lambda config: {"pipeline": "moss-native", "version": MOSS_POLICY_VERSION,
                                  "settings": asdict(config.moss_settings())},
    "joint-vad": _joint_cache_signature,
    "qwen-force-align": _qwen_cache_signature,
}


def detect_cache_signature(config: DetectConfig) -> dict:
    """Only the selected pipeline's settings affect its cached output."""
    try:
        signature = CACHE_SIGNATURES[config.aligner]
    except KeyError as exc:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN",
                          f"Unknown sentence aligner: {config.aligner}") from exc
    return signature(config)


def create_detect_stage(config: DetectConfig, audio_path: str) -> DetectStage:
    try:
        factory = STAGE_FACTORIES[config.aligner]
    except KeyError as exc:
        raise DetectError(
            "DETECT_IMPLEMENTATION_UNKNOWN",
            f"Unknown sentence aligner: {config.aligner}",
        ) from exc
    return factory(config, audio_path)
