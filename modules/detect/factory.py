"""Composition root for the replaceable detect stage."""

from .aligners import create_sentence_aligner
from .atom_detectors import create_atom_detector
from .contracts import AlignmentDependencies, DetectConfig, DetectStage
from .duration import create_duration_estimator
from .partitioners import create_region_partitioner
from .sentence_detectors import create_sentence_detector
from .stage import DefaultDetectStage


def create_detect_stage(config: DetectConfig, audio_path: str) -> DetectStage:
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
