"""Standalone Qwen Force Align channel for Japanese sentence detection and alignment."""

from dataclasses import replace
import time

from ..analysis_plugins import AnalysisPluginClient, validate_timeline
from ..vad.contracts import AsrSlice
from .artifacts import write_alignment_audit, write_detect_artifact
from .qwen_config import QwenConfig
from .qwen_sentences import QwenSentencePartitioner, QwenWordPartitioner, lexical, _normalize_text
from .contracts import (
    AlignedSentenceBlock,
    AlignmentRequest,
    AlignmentResult,
    DetectConfig,
    DetectError,
    DetectStageInput,
    DetectStageResult,
    SentenceDetection,
)

def _expand_short_span(
    start_ms: int,
    end_ms: int,
    min_duration_ms: int,
    lower_bound_ms: int,
    upper_bound_ms: int,
) -> tuple[int, int]:
    """Expand a sub-minimum aligned sentence span within [lower_bound_ms, upper_bound_ms]."""
    duration = end_ms - start_ms
    if duration >= min_duration_ms or upper_bound_ms <= lower_bound_ms:
        return start_ms, end_ms
    target = min(min_duration_ms, upper_bound_ms - lower_bound_ms)
    if duration >= target:
        return start_ms, end_ms
    needed = target - duration
    grow_left = min(start_ms - lower_bound_ms, needed // 2)
    start_ms -= grow_left
    grow_right = min(upper_bound_ms - end_ms, needed - grow_left)
    end_ms += grow_right
    remaining = target - (end_ms - start_ms)
    if remaining > 0:
        extra_left = min(start_ms - lower_bound_ms, remaining)
        start_ms -= extra_left
    return start_ms, end_ms


class QwenForceAlignSentenceAligner:
    """Standalone Qwen forced-alignment sentence splitter."""

    def __init__(self, config: QwenConfig | DetectConfig, audio_path, *, client=None,
                 partitioner: QwenWordPartitioner | None = None):
        # Accept old construction sites while isolating their FSMN settings.
        self.config = config.qwen_settings() if isinstance(config, DetectConfig) else config
        self.audio_path = audio_path if isinstance(audio_path, str) else audio_path.audio_path
        self.partitioner = partitioner or QwenSentencePartitioner(self.config)
        self.client = client or AnalysisPluginClient(
            self.config.endpoint, "qwen-force-align", "qwen-force-align/1"
        )

    def align(self, value: AlignmentRequest) -> AlignmentResult:
        parent, detection = value.parent, value.detection
        text = _normalize_text(detection.normalized_text)
        attempts, timings = [], []
        final_type = "fragment"

        def fallback(reason: str, outcome: str = "fallback") -> AlignmentResult:
            return AlignmentResult(
                (
                    AlignedSentenceBlock(
                        f"{parent.id}-sentence-0",
                        value.source_block_id,
                        parent.speech_start_ms,
                        parent.speech_end_ms,
                        text or detection.normalized_text,
                        parent.region_ids,
                        "fallback-unsplit",
                        "low",
                        final_type,
                        parent.speech_start_ms,
                        parent.speech_end_ms,
                    ),
                ),
                tuple(attempts),
                outcome,
                reason,
                tuple(timings),
            )

        if value.bypass_reason:
            return fallback(value.bypass_reason, "bypassed")
        if detection.reason or not lexical(text):
            return fallback(detection.reason or "no_tokens")
        started = time.monotonic()
        try:
            body, duration = self.client.request(
                "/align",
                self.audio_path,
                {"text": text, "language": "Japanese"},
                start_ms=parent.speech_start_ms,
                end_ms=parent.speech_end_ms,
            )
            words = validate_timeline(body, "words", duration)
            if (
                not words
                or any(
                    not isinstance(row.get("text"), str) or not lexical(row["text"])
                    for row in words
                )
                or "".join(lexical(row["text"]) for row in words) != lexical(text)
            ):
                raise ValueError("Forced alignment transcript mismatch")
            attempts.append({"method": "qwen-force-align", "words": words})
            segment_ends, word_meta = self.partitioner.partition(words, text)
            raw_spans: list[tuple[int, int, str, dict]] = []
            word_start = 0
            char_start = 0
            for idx, word_end in enumerate(segment_ends):
                meta = word_meta[word_end]
                char_end = (
                    len(text)
                    if idx == len(segment_ends) - 1
                    else meta["char_end"]
                )
                start = parent.speech_start_ms + words[word_start]["start_ms"]
                end = parent.speech_start_ms + words[word_end]["end_ms"]
                if end > parent.speech_end_ms or end <= start:
                    raise ValueError("Forced sentence exceeds parent or has non-positive duration")
                text_slice = text[char_start:char_end]
                if not text_slice:
                    raise ValueError("Forced sentence text slice is empty")
                raw_spans.append((start, end, text_slice, meta))
                word_start = word_end + 1
                char_start = char_end

            groups: list[AlignedSentenceBlock] = []
            for idx, (start, end, text_slice, meta) in enumerate(raw_spans):
                lower_bound = (
                    parent.speech_start_ms if idx == 0 else groups[-1].end_ms
                )
                upper_bound = (
                    parent.speech_end_ms
                    if idx == len(raw_spans) - 1
                    else raw_spans[idx + 1][0]
                )
                start, end = _expand_short_span(
                    start,
                    end,
                    self.config.min_sentence_duration_ms,
                    lower_bound,
                    upper_bound,
                )
                high_conf = meta["score"] >= 0.90 or meta["gap_ms"] >= 350
                confidence = "high" if high_conf else "medium"
                groups.append(
                    AlignedSentenceBlock(
                        f"{parent.id}-sentence-{len(groups)}",
                        value.source_block_id,
                        start,
                        end,
                        text_slice,
                        parent.region_ids,
                        "qwen-force-align",
                        confidence,
                        meta["sentence_type"],
                        parent.speech_start_ms,
                        parent.speech_end_ms,
                    )
                )
            if not groups:
                raise ValueError("Forced sentence partition produced no groups")
        except Exception as error:
            code = getattr(error, "code", type(error).__name__)
            attempts.append({"error": code, "method": "qwen-force-align"})
            print(f"       [detect] Qwen 强制对齐失败，保留原片段（{code}）")
            timings.append({
                "method": "qwen-force-align",
                "elapsedMs": round((time.monotonic() - started) * 1000),
            })
            return fallback("forced_alignment_failed")
        timings.append({
            "method": "qwen-force-align",
            "elapsedMs": round((time.monotonic() - started) * 1000),
        })
        return AlignmentResult(
            tuple(groups),
            tuple(attempts),
            "exact" if len(groups) == 1 else "aligned",
            timings=tuple(timings),
        )


class QwenForceAlignDetectStage:
    """Dedicated detect stage for Qwen Force Align without MeCab or VAD atom splitting."""

    def __init__(
        self,
        config: QwenConfig | DetectConfig,
        audio_path: str,
        *,
        aligner: QwenForceAlignSentenceAligner | None = None,
    ):
        self.config = config.qwen_settings() if isinstance(config, DetectConfig) else config
        self.audio_path = audio_path
        self.aligner = aligner or QwenForceAlignSentenceAligner(self.config, audio_path)

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
            parent = block.asr_slice or AsrSlice(
                f"document-asr:{position}:{block.index}",
                block.start_ms,
                block.end_ms,
                block.start_ms,
                block.end_ms,
                (f"document-asr:{position}:{block.index}",),
            )
            if parent.id in seen:
                raise DetectError(
                    "DETECT_LINEAGE_INVALID",
                    "ASR block lost or duplicated its slice identity",
                )
            seen.add(parent.id)
            raw_text = block.source_text or block.text or ""
            normalized = _normalize_text(raw_text)
            has_lexical = bool(lexical(normalized))
            detection = SentenceDetection(
                normalized_text=normalized,
                tokens=(),
                boundaries=(),
                method="qwen-force-align",
                parse_confidence=1.0 if has_lexical else 0.0,
                reason=None if has_lexical else "empty_asr",
            )
            bypass_reason = (
                "skip_tts"
                if block.skip_tts
                else ("empty_asr" if not has_lexical else None)
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
                if aligned.source_start_ms is None or aligned.source_end_ms is None:
                    aligned = replace(
                        aligned,
                        source_start_ms=parent.speech_start_ms,
                        source_end_ms=parent.speech_end_ms,
                    )
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
            f"       [detect] Qwen 强制对齐 {summary['alignedSlices']} / "
            f"单句 {summary['exactSlices']} / 回退 {summary['fallbackSlices']} / "
            f"豁免 {summary['bypassedSlices']}，输出 {len(output)} 句"
        )
        print(f"       [detect] {path}")
        return DetectStageResult(tuple(output), summary)
