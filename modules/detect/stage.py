"""ASR-to-Japanese-sentence detect stage."""

from dataclasses import replace

from .artifacts import write_alignment_audit, write_detect_artifact
from .contracts import (
    AlignmentRequest,
    DetectConfig,
    DetectError,
    DetectStageInput,
    DetectStageResult,
    SentenceDetectionInput,
)
from ..vad.contracts import AsrSlice


def detect_eligible(
    enabled,
    subtitle_path,
    backend,
    language,
    llm_config=None,
    project_vad=True,
):
    return bool(
        enabled
        and not subtitle_path
        and language == "ja"
    )


def _vocalization_reason(text, config):
    chars = [char for char in text if char.isalnum() or char in "ーっッ"]
    if not chars:
        return "empty_asr"
    run = 0
    for char in chars:
        run = run + 1 if char in config.vocalization_chars else 0
        if run >= config.vocalization_run:
            return "repeated_vocalization"
    ratio = sum(char in config.vocalization_chars for char in chars) / len(chars)
    return "vocalization_ratio" if ratio >= config.vocalization_ratio else None


class DefaultDetectStage:
    """Coordinate injected analysis/alignment implementations and persistence."""

    def __init__(self, config: DetectConfig, sentence_detector, aligner):
        self.config = config
        self.sentence_detector = sentence_detector
        self.aligner = aligner

    def run(self, value: DetectStageInput, output_dir: str) -> DetectStageResult:
        asr_blocks = value.asr_blocks
        write_detect_artifact(
            output_dir,
            "sentence_detection_input.json",
            {"config": self.config, "blocks": asr_blocks},
        )
        summary = {
            "eligibleSlices": 0,
            "exactSlices": 0,
            "alignedSlices": 0,
            "fallbackSlices": 0,
            "bypassedSlices": 0,
            "outputSentences": 0,
        }
        records, output, seen = [], [], set()

        for position, block in enumerate(asr_blocks):
            # Document ASR owns its timestamps and need not provide project VAD
            # lineage. Derive a stable parent without mutating the source block.
            parent = block.asr_slice or AsrSlice(
                f"document-asr:{position}:{block.index}",
                block.start_ms, block.end_ms, block.start_ms, block.end_ms,
                (f"document-asr:{position}:{block.index}",),
            )
            if parent.id in seen:
                raise DetectError(
                    "DETECT_LINEAGE_INVALID",
                    "ASR block lost or duplicated its slice identity",
                )
            seen.add(parent.id)
            source_text = block.source_text or block.text or ""
            detection = self.sentence_detector.detect(
                SentenceDetectionInput(parent, source_text)
            )
            bypass_reason = (
                "skip_tts" if block.skip_tts
                else _vocalization_reason(source_text, self.config)
            )
            request = AlignmentRequest(
                parent,
                parent.id,
                detection,
                bypass_reason,
            )
            result = self.aligner.align(request)
            records.append((parent.id, detection, request, result))

            summary_key = {
                "exact": "exactSlices",
                "aligned": "alignedSlices",
                "fallback": "fallbackSlices",
                "bypassed": "bypassedSlices",
            }.get(result.outcome)
            if summary_key is None:
                raise DetectError(
                    "DETECT_ALIGNMENT_INVALID",
                    "Aligner returned an unknown outcome",
                )
            summary[summary_key] += 1
            summary["eligibleSlices"] += result.outcome != "bypassed"

            for aligned in result.blocks:
                sentence_slice = AsrSlice(
                    aligned.id,
                    aligned.start_ms,
                    aligned.end_ms,
                    aligned.start_ms,
                    aligned.end_ms,
                    aligned.source_region_ids,
                )
                output.append(replace(
                    block,
                    index=len(output),
                    start_ms=aligned.start_ms,
                    end_ms=aligned.end_ms,
                    text=aligned.text,
                    skip_tts=block.skip_tts,
                    asr_slice=sentence_slice,
                    source_text=aligned.text,
                    cleaned_source_text="",
                    aligned_sentence=aligned,
                    translation_available=False,
                ))

        previous_end = -1
        # Document backends may author overlapping speaker turns. Preserve those
        # ranges; project VAD slices retain their strict non-overlap contract.
        document_timeline = any(block.asr_slice is None for block in asr_blocks)
        for block in output:
            if (not document_timeline and block.start_ms < previous_end) or block.end_ms <= block.start_ms:
                raise DetectError(
                    "DETECT_ALIGNMENT_INVALID",
                    "Invalid global sentence timeline",
                )
            previous_end = block.end_ms
        summary["outputSentences"] = len(output)
        path = write_alignment_audit(output_dir, self.config, summary, records)
        print(
            f"       [detect] 日语句界对齐 {summary['alignedSlices']} / "
            f"单句 {summary['exactSlices']} / 回退 {summary['fallbackSlices']} / "
            f"豁免 {summary['bypassedSlices']}，输出 {len(output)} 句"
        )
        print(f"       [detect] {path}")
        return DetectStageResult(tuple(output), summary)
