"""Composition root for the replaceable detect stage."""

from dataclasses import asdict, replace

from .aligners import create_sentence_aligner
from .asr_boost_factory import (
    asr_boost_cache_signature,
    asr_boost_supports,
    create_asr_boost_pre_stage,
    create_asr_boost_stage,
)
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


# Native document backends own a dedicated detect pipeline. The user still
# chooses whether detect runs; when it does, their blocks never enter the
# generic pipelines and dedicated pipelines never receive other backends' blocks.
DEDICATED_PIPELINES = {"moss-transcribe-diarize": "moss-native"}


def detect_pipeline_supports(aligner: str, backend: str) -> bool:
    """moss-native is exclusive to MOSS; asr-boost requires a boost-capable ASR backend."""
    if aligner == "moss-native":
        return backend == "moss-transcribe-diarize"
    if aligner == "asr-boost":
        return asr_boost_supports(backend)
    return True


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
    return DefaultDetectStage(config, sentence_detector, aligner, asr_boost=create_asr_boost_pre_stage(config))


def _create_qwen_force_align_stage(config: DetectConfig, audio_path: str) -> DetectStage:
    return QwenForceAlignDetectStage(config.qwen_settings(), audio_path, asr_boost=create_asr_boost_pre_stage(config))


def _joint_cache_signature(config: DetectConfig) -> dict:
    settings = asdict(config)
    settings.pop("qwen")
    settings.pop("qwen_endpoint")
    settings.pop("moss")
    settings.pop("asr_boost", None)
    settings.pop("asr_backend", None)
    settings.pop("asr_boost_padding_ms", None)
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
    "asr-boost": create_asr_boost_stage,
}

CACHE_SIGNATURES = {
    "moss-native": lambda config: {"pipeline": "moss-native", "version": MOSS_POLICY_VERSION,
                                  "settings": asdict(config.moss_settings())},
    "joint-vad": _joint_cache_signature,
    "qwen-force-align": _qwen_cache_signature,
    "asr-boost": asr_boost_cache_signature,
}


def validate_detect_selection(config: DetectConfig, backend: str) -> DetectConfig:
    """Reject a user-selected detect pipeline that the ASR backend cannot feed."""
    if config.aligner not in STAGE_FACTORIES:
        raise DetectError("DETECT_IMPLEMENTATION_UNKNOWN",
                          f"Unknown sentence aligner: {config.aligner}")
    if not detect_pipeline_supports(config.aligner, backend):
        raise DetectError(
            "DETECT_BACKEND_INCOMPATIBLE",
            f"Detect pipeline {config.aligner} does not support ASR backend {backend}",
        )
    if config.asr_backend != backend:
        config = replace(config, asr_backend=backend)
    return config


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
