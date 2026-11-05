"""Refine native MOSS blocks without crossing their timestamps or speakers.

Punctuation owns text boundaries. An injected word-alignment port supplies only
internal times; unavailable or ambiguous alignment retains the native block.
Inputs are never mutated, including overlapping turns sharing one VAD slice.
"""
from dataclasses import asdict, replace
import re
import unicodedata
from typing import Protocol

from ..analysis_plugins import AnalysisPluginClient, AnalysisPluginError, validate_timeline
from .artifacts import write_detect_artifact
from .contracts import AlignedSentenceBlock, DetectStageInput, DetectStageResult


from .moss_config import MOSS_POLICY_VERSION, MossDetectConfig


class WordAlignmentPort(Protocol):
    def request(self, route: str, audio_path: str, data: dict, *,
                start_ms: int, end_ms: int) -> tuple[dict, int]: ...


def _lexical(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def native_sentences(text: str) -> tuple[str, ...]:
    """Preserve every character, closing quotes, and bracketed event tags."""
    ends, depth = [], 0
    for position, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if not depth and char in "。！？!?":
            end = position + 1
            while end < len(text) and text[end] in "。！？!?」』）)\"’” \t\n":
                end += 1
            if not ends or end > ends[-1]:
                ends.append(end)
    parts, start = [], 0
    for end in ends:
        parts.append(text[start:end])
        start = end
    if start < len(text):
        parts.append(text[start:])
    # Non-speech tags and punctuation-only tails remain native observations.
    if len(parts) < 2 or any(not _lexical(part) or re.search(r"\[[^\]]*\]", part) for part in parts):
        return (text,)
    return tuple(parts)


class MossDetectStage:
    def __init__(self, config: MossDetectConfig, audio_path: str, *,
                 client: WordAlignmentPort | None = None):
        self.config, self.audio_path, self.client = config, audio_path, client

    def _request_alignment(self, block) -> tuple[dict, int]:
        client = self.client or AnalysisPluginClient(
            self.config.alignment_endpoint, "qwen-force-align", "qwen-force-align/1")
        return client.request("/align", self.audio_path,
            {"text": block.text, "language": "Japanese"},
            start_ms=block.start_ms, end_ms=block.end_ms)

    def _spans(self, block, parts, body: dict, duration: int) -> tuple[tuple[int, int], ...]:
        if abs(duration - (block.end_ms - block.start_ms)) > 1:
            raise ValueError("alignment_duration_mismatch")
        words = validate_timeline(body, "words", duration)
        if not words or any(not isinstance(w.get("text"), str) or not _lexical(w["text"]) for w in words):
            raise ValueError("invalid_alignment_words")
        if "".join(_lexical(w["text"]) for w in words) != _lexical(block.text):
            raise ValueError("alignment_text_mismatch")
        offsets, count = {}, 0
        for word in words:
            count += len(_lexical(word["text"]))
            offsets[count] = word["end_ms"] + block.start_ms
        edges, count = [block.start_ms], 0
        for part in parts[:-1]:
            count += len(_lexical(part))
            if count not in offsets:
                raise ValueError("sentence_boundary_inside_word")
            edges.append(offsets[count])
        edges.append(block.end_ms)
        if any(end - start < self.config.min_sentence_duration_ms
               for start, end in zip(edges, edges[1:])):
            raise ValueError("sentence_too_short")
        return tuple(zip(edges, edges[1:]))

    def run(self, value: DetectStageInput, output_dir: str) -> DetectStageResult:
        write_detect_artifact(output_dir, "moss_detection_input.json", {
            "policyVersion": MOSS_POLICY_VERSION, "config": asdict(self.config),
            "blocks": value.asr_blocks})
        output, records = [], []
        summary = {"eligibleSlices": 0, "exactSlices": 0, "alignedSlices": 0,
                   "fallbackSlices": 0, "bypassedSlices": 0, "outputSentences": 0}
        for position, block in enumerate(value.asr_blocks):
            parts = native_sentences(block.text)
            outcome, reason, spans = "exact", None, ()
            if block.skip_tts:
                outcome = "bypassed"
            elif len(parts) > 1:
                try:
                    body, duration = self._request_alignment(block)
                    write_detect_artifact(output_dir, f"moss_alignment_raw_{position}.json", {
                        "blockIndex": block.index, "durationMs": duration, "response": body})
                    spans = self._spans(block, parts, body, duration)
                    outcome = "aligned"
                except (AnalysisPluginError, ValueError, TypeError, KeyError) as error:
                    outcome = "fallback"
                    reason = getattr(error, "code", str(error))
                    print(f"       [detect:moss] 块 {block.index} 对齐失败，保留原块（{reason}）")
            summary[outcome + "Slices"] += 1
            summary["eligibleSlices"] += outcome != "bypassed"
            source_id = f"{block.asr_slice.id if block.asr_slice else 'moss'}:block:{position}:{block.index}"
            if outcome != "aligned":
                output.append(replace(block, index=len(output)))
            else:
                for child, (text, (start, end)) in enumerate(zip(parts, spans)):
                    aligned = AlignedSentenceBlock(
                        f"{source_id}:sentence:{child}", source_id, start, end, text,
                        block.asr_slice.region_ids if block.asr_slice else (),
                        "moss-native-word-align", "high", "sentence",
                        block.start_ms, block.end_ms)
                    output.append(replace(block, index=len(output), start_ms=start, end_ms=end,
                        text=text, source_text=text, cleaned_source_text="", aligned_sentence=aligned,
                        translation_available=False))
            records.append({"sourceBlockId": source_id, "outcome": outcome, "reason": reason,
                            "parts": parts, "spans": spans})
        summary["outputSentences"] = len(output)
        write_detect_artifact(output_dir, "moss_detection.json", {
            "policyVersion": MOSS_POLICY_VERSION, "config": self.config,
            "summary": summary, "records": records, "blocks": output})
        print(f"       [detect:moss] 细分 {summary['alignedSlices']} 块，"
              f"回退 {summary['fallbackSlices']} 块，输出 {len(output)} 句")
        return DetectStageResult(tuple(output), summary)
