"""ASR-boost sentence detection stage using coarse VAD outer bounds and padded internal cuts."""

from __future__ import annotations

from dataclasses import asdict, replace

from ..vad.contracts import AsrSlice
from .artifacts import write_detect_artifact
from .asr_boost_config import ASR_BOOST_POLICY_VERSION, AsrBoostConfig
from .asr_boost_factory import AsrBoostAdapter, AsrBoostSpan, create_asr_boost_adapter
from .contracts import (
    AlignedSentenceBlock,
    DetectError,
    DetectStageInput,
    DetectStageResult,
)
from .duration import count_japanese_mora
from .moss_stage import _lexical, _refine_boundary_by_energy, native_sentences


def _ends_with_sentence_punctuation(text: str) -> bool:
    stripped = text.rstrip("」』）)\"’” \t\n")
    return bool(stripped) and stripped[-1] in "。！？!?"


def _subdivide_span_by_punctuation(span: AsrBoostSpan) -> tuple[AsrBoostSpan, ...]:
    """Split a span containing multiple punctuation-delimited sentences proportionally by mora."""
    parts = native_sentences(span.text)
    if len(parts) <= 1:
        return (span,)
    duration = span.end_ms - span.start_ms
    if duration < len(parts):
        return (span,)
    weights = [
        max(1.0, count_japanese_mora("", part) or float(len(_lexical(part)) or 1))
        for part in parts
    ]
    total_weight = sum(weights)
    out: list[AsrBoostSpan] = []
    cursor = span.start_ms
    acc_weight = 0.0
    for idx, (part, weight) in enumerate(zip(parts, weights)):
        acc_weight += weight
        if idx == len(parts) - 1:
            end_ms = span.end_ms
        else:
            remaining = len(parts) - idx - 1
            raw_end = span.start_ms + round(duration * acc_weight / total_weight)
            end_ms = max(cursor + 1, min(span.end_ms - remaining, raw_end))
        out.append(AsrBoostSpan(start_ms=cursor, end_ms=end_ms, text=part.strip()))
        cursor = end_ms
    return tuple(out)


def _group_spans_into_sentences(
    spans: tuple[AsrBoostSpan, ...],
    cfg: AsrBoostConfig,
) -> tuple[AsrBoostSpan, ...]:
    """Expand multi-sentence spans and merge word/fragment spans into sentence units."""
    expanded: list[AsrBoostSpan] = []
    for span in spans:
        expanded.extend(_subdivide_span_by_punctuation(span))
    if len(expanded) <= 1:
        return tuple(expanded)

    grouped: list[AsrBoostSpan] = []
    cur_start = expanded[0].start_ms
    cur_end = expanded[0].end_ms
    cur_texts = [expanded[0].text]

    for next_span in expanded[1:]:
        joined_text = "".join(cur_texts)
        gap_ms = max(0, next_span.start_ms - cur_end)
        cur_duration = cur_end - cur_start
        should_cut = (
            _ends_with_sentence_punctuation(joined_text)
            or (gap_ms >= cfg.padding_ms and cur_duration >= cfg.min_duration_ms)
        )
        if should_cut and cur_duration >= cfg.min_duration_ms:
            grouped.append(AsrBoostSpan(start_ms=cur_start, end_ms=cur_end, text=joined_text))
            cur_start = next_span.start_ms
            cur_end = next_span.end_ms
            cur_texts = [next_span.text]
        else:
            cur_end = max(cur_end, next_span.end_ms)
            cur_texts.append(next_span.text)

    tail_text = "".join(cur_texts)
    if grouped and (cur_end - cur_start) < cfg.min_duration_ms and not _ends_with_sentence_punctuation(grouped[-1].text):
        prev = grouped.pop()
        grouped.append(AsrBoostSpan(start_ms=prev.start_ms, end_ms=cur_end, text=prev.text + tail_text))
    else:
        grouped.append(AsrBoostSpan(start_ms=cur_start, end_ms=cur_end, text=tail_text))
    return tuple(grouped)


def align_sentence_spans(
    parent: AsrSlice,
    sentence_spans: tuple[AsrBoostSpan, ...],
    cfg: AsrBoostConfig,
    audio_path: str | None = None,
) -> tuple[tuple[tuple[int, int, str], ...], list[dict]]:
    """Lock first start to speech_start_ms, last end to speech_end_ms, and refine middle cuts with ±padding_ms."""
    speech_start = parent.speech_start_ms
    speech_end = parent.speech_end_ms
    n = len(sentence_spans)
    if n <= 1 or (speech_end - speech_start) < n * cfg.min_duration_ms:
        text = "".join(s.text for s in sentence_spans).strip()
        return ((speech_start, speech_end, text),), []

    cuts: list[int] = [speech_start]
    cut_audits: list[dict] = []
    for idx in range(1, n):
        prev_span = sentence_spans[idx - 1]
        next_span = sentence_spans[idx]
        remaining_sentences = n - idx
        min_allowed = cuts[-1] + cfg.min_duration_ms
        max_allowed = speech_end - remaining_sentences * cfg.min_duration_ms
        if max_allowed < min_allowed:
            return ((speech_start, speech_end, "".join(s.text for s in sentence_spans).strip()),), []

        raw_prev_end = max(speech_start, min(speech_end, prev_span.end_ms))
        raw_next_start = max(raw_prev_end, min(speech_end, next_span.start_ms))
        anchor_ms = (raw_prev_end + raw_next_start) // 2
        window_start = max(min_allowed, raw_prev_end - cfg.padding_ms)
        window_end = min(max_allowed, raw_next_start + cfg.padding_ms)
        if window_end < window_start:
            window_start = window_end = max(min_allowed, min(max_allowed, anchor_ms))

        clamped_anchor = max(window_start, min(window_end, anchor_ms))
        if audio_path and window_end > window_start:
            refined = _refine_boundary_by_energy(
                audio_path,
                window_start,
                window_end,
                clamped_anchor,
                search_window_ms=cfg.padding_ms,
            )
            cut_ms = max(window_start, min(window_end, refined))
        else:
            cut_ms = clamped_anchor

        cuts.append(cut_ms)
        cut_audits.append({
            "boundaryIndex": idx,
            "rawPrevEndMs": raw_prev_end,
            "rawNextStartMs": raw_next_start,
            "anchorMs": anchor_ms,
            "windowStartMs": window_start,
            "windowEndMs": window_end,
            "paddingMs": cfg.padding_ms,
            "cutMs": cut_ms,
            "driftMs": cut_ms - anchor_ms,
        })

    cuts.append(speech_end)
    aligned = tuple(
        (cuts[i], cuts[i + 1], sentence_spans[i].text)
        for i in range(n)
    )
    return aligned, cut_audits


class AsrBoostDetectStage:
    """Refine coarse ASR blocks using backend/plugin timestamps and padded boundary alignment."""

    def __init__(
        self,
        config: AsrBoostConfig,
        audio_path: str,
        *,
        adapter: AsrBoostAdapter | None = None,
    ) -> None:
        self.config = config
        self.audio_path = audio_path
        self.adapter = adapter or create_asr_boost_adapter(config.backend)

    def run(self, value: DetectStageInput, output_dir: str) -> DetectStageResult:
        asr_blocks = value.asr_blocks
        write_detect_artifact(
            output_dir,
            "sentence_detection_input.json",
            {
                "pipeline": "asr-boost",
                "version": ASR_BOOST_POLICY_VERSION,
                "config": asdict(self.config),
                "blocks": asr_blocks,
            },
        )
        summary = {
            "eligibleSlices": 0,
            "exactSlices": 0,
            "alignedSlices": 0,
            "fallbackSlices": 0,
            "bypassedSlices": 0,
            "outputSentences": 0,
        }
        records: list[dict] = []
        output = []
        seen: set[str] = set()

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

            source_text = (block.source_text or block.text or "").strip()
            effective_cfg = self.adapter.resolve_config(block, self.config)

            if block.skip_tts or not source_text or not _lexical(source_text) or not effective_cfg.enabled:
                reason = (
                    "skip_tts"
                    if block.skip_tts
                    else "plugin_disabled"
                    if not effective_cfg.enabled
                    else "empty_asr"
                )
                summary["bypassedSlices"] += 1
                aligned_block = self._build_output_block(
                    block,
                    parent,
                    len(output),
                    0,
                    parent.speech_start_ms,
                    parent.speech_end_ms,
                    source_text,
                    method="asr-boost-bypass",
                    confidence="low",
                    sentence_type="fragment",
                )
                output.append(aligned_block)
                records.append({
                    "sliceId": parent.id,
                    "outcome": "bypassed",
                    "reason": reason,
                    "effectiveConfig": asdict(effective_cfg),
                    "outputBlocks": [aligned_block.aligned_sentence],
                })
                continue

            summary["eligibleSlices"] += 1
            raw_spans = self.adapter.extract_spans(block, parent)
            if not raw_spans:
                raw_spans = (
                    AsrBoostSpan(
                        start_ms=parent.speech_start_ms,
                        end_ms=parent.speech_end_ms,
                        text=source_text,
                    ),
                )

            sentence_spans = _group_spans_into_sentences(raw_spans, effective_cfg)
            if len(sentence_spans) <= 1:
                summary["exactSlices"] += 1
                aligned_block = self._build_output_block(
                    block,
                    parent,
                    len(output),
                    0,
                    parent.speech_start_ms,
                    parent.speech_end_ms,
                    source_text,
                    method=f"asr-boost:{self.adapter.backend}:single",
                    confidence="high",
                    sentence_type="statement",
                )
                output.append(aligned_block)
                records.append({
                    "sliceId": parent.id,
                    "outcome": "exact",
                    "reason": "single_sentence",
                    "effectiveConfig": asdict(effective_cfg),
                    "inputSpans": [asdict(s) for s in raw_spans],
                    "outputBlocks": [aligned_block.aligned_sentence],
                })
                continue

            aligned_ranges, cut_audits = align_sentence_spans(
                parent,
                sentence_spans,
                effective_cfg,
                audio_path=self.audio_path,
            )
            if len(aligned_ranges) <= 1:
                summary["fallbackSlices"] += 1
                aligned_block = self._build_output_block(
                    block,
                    parent,
                    len(output),
                    0,
                    parent.speech_start_ms,
                    parent.speech_end_ms,
                    source_text,
                    method=f"asr-boost:{self.adapter.backend}:fallback-unsplit",
                    confidence="low",
                    sentence_type="statement",
                )
                output.append(aligned_block)
                records.append({
                    "sliceId": parent.id,
                    "outcome": "fallback",
                    "reason": "duration_below_minimum",
                    "effectiveConfig": asdict(effective_cfg),
                    "inputSpans": [asdict(s) for s in raw_spans],
                    "outputBlocks": [aligned_block.aligned_sentence],
                })
                continue

            summary["alignedSlices"] += 1
            created = []
            for local_idx, (start_ms, end_ms, part_text) in enumerate(aligned_ranges):
                aligned_block = self._build_output_block(
                    block,
                    parent,
                    len(output),
                    local_idx,
                    start_ms,
                    end_ms,
                    part_text,
                    method=f"asr-boost:{self.adapter.backend}",
                    confidence="high",
                    sentence_type="statement",
                )
                output.append(aligned_block)
                created.append(aligned_block.aligned_sentence)
            records.append({
                "sliceId": parent.id,
                "outcome": "aligned",
                "effectiveConfig": asdict(effective_cfg),
                "inputSpans": [asdict(s) for s in raw_spans],
                "cutAudits": cut_audits,
                "outputBlocks": created,
            })

        previous_end = -1
        document_timeline = any(block.asr_slice is None for block in asr_blocks)
        for item in output:
            if (not document_timeline and item.start_ms < previous_end) or item.end_ms <= item.start_ms:
                raise DetectError(
                    "DETECT_ALIGNMENT_INVALID",
                    "Invalid global sentence timeline",
                )
            previous_end = item.end_ms

        summary["outputSentences"] = len(output)
        audit_path = write_detect_artifact(
            output_dir,
            "alignment_audit.json",
            {
                "pipeline": "asr-boost",
                "version": ASR_BOOST_POLICY_VERSION,
                "config": asdict(self.config),
                "summary": summary,
                "slices": records,
            },
        )
        print(
            f"       [detect] ASR-Boost 句界检测完成: 输入 {len(asr_blocks)} 片 -> "
            f"输出 {len(output)} 句 (单句 {summary['exactSlices']}, "
            f"对齐 {summary['alignedSlices']}, 退化 {summary['fallbackSlices']}, "
            f"跳过 {summary['bypassedSlices']}): {audit_path}"
        )
        return DetectStageResult(tuple(output), summary)

    @staticmethod
    def _build_output_block(
        source_block,
        parent: AsrSlice,
        global_index: int,
        local_index: int,
        start_ms: int,
        end_ms: int,
        text: str,
        *,
        method: str,
        confidence: str,
        sentence_type: str,
    ):
        aligned_id = f"{parent.id}-sentence-{local_index}"
        aligned = AlignedSentenceBlock(
            id=aligned_id,
            source_block_id=parent.id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            source_region_ids=parent.region_ids,
            method=method,
            confidence=confidence,
            sentence_type=sentence_type,
            source_start_ms=parent.speech_start_ms,
            source_end_ms=parent.speech_end_ms,
        )
        sentence_slice = AsrSlice(
            aligned_id,
            start_ms,
            end_ms,
            start_ms,
            end_ms,
            parent.region_ids,
        )
        return replace(
            source_block,
            index=global_index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            skip_tts=source_block.skip_tts,
            asr_slice=sentence_slice,
            source_text=text,
            cleaned_source_text="",
            aligned_sentence=aligned,
            translation_available=False,
        )
