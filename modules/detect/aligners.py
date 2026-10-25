"""Pluggable Japanese sentence-to-audio alignment implementations."""

from dataclasses import replace
import time

from .contracts import (
    AlignedSentenceBlock,
    AlignmentDependencies,
    AlignmentResult,
    DetectConfig,
    DetectError,
    SentenceAligner,
)
from ..vad.contracts import VadRegion


def validate_regions(regions, parent):
    """Validate ordered atom ownership and normalize a 1ms rounding overlap."""
    result, previous, seen = [], parent.speech_start_ms, set()
    for region in regions:
        if (
            not isinstance(region, VadRegion)
            or region.id in seen
            or region.speech_start_ms < parent.speech_start_ms
            or region.speech_end_ms > parent.speech_end_ms
            or region.speech_start_ms < previous - 1
        ):
            raise DetectError(
                "DETECT_ALIGNMENT_INVALID",
                "Invalid speech atom boundary or order",
            )
        region = replace(
            region,
            speech_start_ms=max(previous, region.speech_start_ms),
        )
        if region.speech_start_ms >= region.speech_end_ms:
            raise DetectError("DETECT_ALIGNMENT_INVALID", "Empty speech atom")
        result.append(region)
        seen.add(region.id)
        previous = region.speech_end_ms
    return tuple(result)


class JointVadSentenceAligner:
    """Jointly choose Japanese token cuts and 200ms VAD atom groups."""

    def __init__(self, config, dependencies: AlignmentDependencies, clock=time.monotonic):
        self.config = config
        self.audio_path = dependencies.audio_path
        self.atom_detector = dependencies.atom_detector
        self.estimator = dependencies.duration_estimator
        self.partitioner = dependencies.region_partitioner
        self.clock = clock

    def align(self, value):
        parent, detection, config = value.parent, value.detection, self.config
        if (
            not parent.region_ids
            or not (
                0 <= parent.extraction_start_ms <= parent.speech_start_ms
                < parent.speech_end_ms <= parent.extraction_end_ms
            )
        ):
            raise DetectError(
                "DETECT_LINEAGE_INVALID",
                "Invalid parent ASR slice lineage",
            )
        attempts, timings = [], []

        def fallback(reason, *, outcome="fallback"):
            final_type = (
                detection.boundaries[-1].sentence_type
                if detection.boundaries
                else "fragment"
            )
            return AlignmentResult(
                (
                    AlignedSentenceBlock(
                        f"{parent.id}-sentence-0",
                        value.source_block_id,
                        parent.speech_start_ms,
                        parent.speech_end_ms,
                        detection.normalized_text,
                        parent.region_ids,
                        "fallback-unsplit",
                        "low",
                        final_type,
                    ),
                ),
                tuple(attempts),
                outcome,
                reason,
                tuple(timings),
            )

        if value.bypass_reason:
            return fallback(value.bypass_reason, outcome="bypassed")
        if detection.reason or not detection.tokens:
            return fallback(detection.reason or "no_tokens")

        started = self.clock()
        try:
            detected = self.atom_detector.detect(parent, config.atom_silence_ms)
            atoms = validate_regions(tuple(detected), parent)
        except Exception as exc:
            code = exc.code if isinstance(exc, DetectError) else "DETECT_ALIGNMENT_INVALID"
            attempts.append({"error": code, "thresholdMs": config.atom_silence_ms})
            timings.append({
                "thresholdMs": config.atom_silence_ms,
                "elapsedMs": round((self.clock() - started) * 1000),
            })
            return fallback("atom_detection_failed")
        timings.append({
            "thresholdMs": config.atom_silence_ms,
            "elapsedMs": round((self.clock() - started) * 1000),
        })
        attempts.append({
            "thresholdMs": config.atom_silence_ms,
            "regionCount": len(atoms),
            "regions": atoms,
            "parseConfidence": detection.parse_confidence,
        })
        if (
            not atoms
            or any(
                region.speech_end_ms - region.speech_start_ms
                < config.min_region_duration_ms
                for region in atoms
            )
        ):
            return fallback("invalid_atoms")

        partition = self.partitioner.partition(detection, atoms, self.estimator)
        if partition is None or not partition.segments:
            return fallback("no_joint_partition")
        if any(
            max(segment.duration_ratio, 1 / segment.duration_ratio)
            > config.max_duration_ratio_error
            for segment in partition.segments
        ):
            return fallback("duration_mismatch")

        attempts[-1].update({
            "partitionScore": partition.score,
            "segments": partition.segments,
        })
        blocks = []
        for ordinal, segment in enumerate(partition.segments):
            token_group = detection.tokens[segment.token_start:segment.token_end]
            atom_group = atoms[segment.atom_start:segment.atom_end]
            start_char = token_group[0].start_char
            end_char = token_group[-1].end_char
            text = detection.normalized_text[start_char:end_char]
            boundary = detection.boundaries[segment.token_end - 1]
            if not text or not atom_group:
                return fallback("empty_joint_segment")
            if len(partition.segments) == 1:
                confidence = "high" if detection.parse_confidence >= 0.6 else "medium"
                method = "joint-vad-exact"
            elif segment.boundary_score >= 0.9 and detection.parse_confidence >= 0.65:
                confidence = "high"
                method = "joint-vad"
            elif segment.boundary_score >= 0.55:
                confidence = "medium"
                method = "joint-vad"
            else:
                confidence = "low"
                method = "joint-vad"
            blocks.append(AlignedSentenceBlock(
                f"{parent.id}-sentence-{ordinal}",
                value.source_block_id,
                atom_group[0].speech_start_ms,
                atom_group[-1].speech_end_ms,
                text,
                tuple(region.id for region in atom_group),
                method,
                confidence,
                boundary.sentence_type,
            ))

        outcome = "exact" if len(blocks) == 1 else "aligned"
        return AlignmentResult(
            tuple(blocks),
            tuple(attempts),
            outcome,
            timings=tuple(timings),
        )


from .qwen_aligner import QwenForceAlignSentenceAligner

ALIGNERS = {"joint-vad": JointVadSentenceAligner, "qwen-force-align": QwenForceAlignSentenceAligner}


def create_sentence_aligner(
    name: str,
    config: DetectConfig,
    dependencies: AlignmentDependencies,
) -> SentenceAligner:
    try:
        return ALIGNERS[name](config, dependencies)
    except KeyError as exc:
        raise DetectError(
            "DETECT_IMPLEMENTATION_UNKNOWN",
            f"Unknown sentence aligner: {name}",
        ) from exc
