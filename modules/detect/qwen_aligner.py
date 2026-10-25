"""Map forced word timestamps to detected Japanese sentence boundaries."""
import time
import unicodedata

from ..analysis_plugins import AnalysisPluginClient, validate_timeline
from .contracts import (AlignedSentenceBlock, AlignmentDependencies, AlignmentRequest,
                        AlignmentResult, DetectConfig, DetectError)


def lexical(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())


class QwenForceAlignSentenceAligner:
    def __init__(self, config: DetectConfig, dependencies: AlignmentDependencies, *, client=None):
        self.config = config
        self.audio_path = dependencies.audio_path
        self.client = client or AnalysisPluginClient(config.qwen_endpoint, "qwen-force-align", "qwen-force-align/1")

    def align(self, value: AlignmentRequest) -> AlignmentResult:
        parent, detection = value.parent, value.detection
        if (not parent.region_ids or not 0 <= parent.extraction_start_ms <= parent.speech_start_ms
                < parent.speech_end_ms <= parent.extraction_end_ms):
            raise DetectError("DETECT_LINEAGE_INVALID", "Invalid parent ASR slice lineage")
        attempts, timings = [], []

        def fallback(reason, outcome="fallback"):
            return AlignmentResult((AlignedSentenceBlock(f"{parent.id}-sentence-0", value.source_block_id,
                parent.speech_start_ms, parent.speech_end_ms, detection.normalized_text, parent.region_ids,
                "fallback-unsplit", "low"),), tuple(attempts), outcome, reason, tuple(timings))

        if value.bypass_reason:
            return fallback(value.bypass_reason, "bypassed")
        if detection.reason or not lexical(detection.normalized_text):
            return fallback(detection.reason or "no_tokens")
        started = time.monotonic()
        try:
            body, duration = self.client.request("/align", self.audio_path,
                {"text": detection.normalized_text, "language": "Japanese"},
                start_ms=parent.speech_start_ms, end_ms=parent.speech_end_ms)
            words = validate_timeline(body, "words", duration)
            if (not words or any(not isinstance(row.get("text"), str) or not lexical(row["text"]) for row in words)
                    or "".join(lexical(row["text"]) for row in words) != lexical(detection.normalized_text)):
                raise ValueError("Forced alignment transcript mismatch")
            attempts.append({"method": "qwen-force-align", "words": words})
            # Only cut on real word ends with positive linguistic evidence.
            boundaries = {len(lexical(detection.normalized_text[:row.char_index])): row
                          for row in detection.boundaries if row.language_score >= self.config.boundary_bias}
            groups, word_start, char_start, count = [], 0, 0, 0
            for index, word in enumerate(words):
                count += len(lexical(word["text"]))
                boundary = boundaries.get(count)
                if boundary is None and index != len(words) - 1:
                    continue
                char_end = boundary.char_index if boundary and index != len(words) - 1 else len(detection.normalized_text)
                start = parent.speech_start_ms + words[word_start]["start_ms"]
                end = parent.speech_start_ms + word["end_ms"]
                if end > parent.speech_end_ms or end - start < self.config.min_sentence_duration_ms:
                    raise ValueError("Forced sentence is too short or exceeds parent")
                confidence = "high" if boundary and boundary.language_score >= 0.9 and detection.parse_confidence >= 0.65 else "medium"
                groups.append(AlignedSentenceBlock(f"{parent.id}-sentence-{len(groups)}", value.source_block_id,
                    start, end, detection.normalized_text[char_start:char_end], parent.region_ids,
                    "qwen-force-align", confidence, boundary.sentence_type if boundary else "fragment"))
                word_start, char_start = index + 1, char_end
            if (not groups or len(groups) > self.config.max_sentence_count
                    or any(row.end_ms - row.start_ms > self.config.max_sentence_duration_ms for row in groups)):
                raise ValueError("Forced sentence partition exceeds configured limits")
        except Exception as error:
            code = getattr(error, "code", type(error).__name__)
            attempts.append({"error": code, "method": "qwen-force-align"})
            print(f"       [detect] Qwen 强制对齐失败，保留原片段（{code}）")
            timings.append({"method": "qwen-force-align", "elapsedMs": round((time.monotonic() - started) * 1000)})
            return fallback("forced_alignment_failed")
        timings.append({"method": "qwen-force-align", "elapsedMs": round((time.monotonic() - started) * 1000)})
        return AlignmentResult(tuple(groups), tuple(attempts), "exact" if len(groups) == 1 else "aligned",
                               timings=tuple(timings))
