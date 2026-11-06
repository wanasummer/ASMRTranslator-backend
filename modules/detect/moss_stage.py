"""Refine native MOSS blocks without crossing their timestamps or speakers.

Punctuation owns text boundaries. When an external word-alignment port is
injected, it supplies internal times; in standard operation, native Japanese
mora-proportional subdivision with acoustic energy refinement splits multi-sentence
blocks without external network dependencies. Inputs are never mutated.
"""
from dataclasses import asdict, replace
import re
import unicodedata
from typing import Protocol

from ..analysis_plugins import AnalysisPluginClient, AnalysisPluginError, validate_timeline
from .artifacts import write_detect_artifact
from .contracts import AlignedSentenceBlock, DetectStageInput, DetectStageResult
from .duration import count_japanese_mora
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


def _refine_boundary_by_energy(audio_path: str, start_ms: int, end_ms: int,
                               target_ms: int, search_window_ms: int = 250) -> int:
    """Find the local acoustic energy minimum (pause/breath valley) around target_ms."""
    try:
        import struct, wave
        with wave.open(audio_path, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                return target_ms
            scan_start_ms = max(start_ms + 100, target_ms - search_window_ms)
            scan_end_ms = min(end_ms - 100, target_ms + search_window_ms)
            if scan_end_ms <= scan_start_ms:
                return target_ms
            start_frame = int(scan_start_ms * framerate / 1000)
            num_frames = int((scan_end_ms - scan_start_ms) * framerate / 1000)
            wf.setpos(start_frame)
            raw = wf.readframes(num_frames)
            total_samples = len(raw) // (2 * nchannels)
            win_size = int(0.020 * framerate)  # 20ms sliding window
            step = int(0.005 * framerate)       # 5ms step
            if win_size >= total_samples or total_samples <= 0:
                return target_ms
            fmt = f"<{total_samples * nchannels}h"
            data = struct.unpack(fmt, raw)
            mono = [data[i * nchannels] for i in range(total_samples)]

            min_rms = float("inf")
            rms_list = []
            for offset in range(0, total_samples - win_size, step):
                window = mono[offset : offset + win_size]
                rms = sum(s * s for s in window) / win_size
                rms_list.append((offset, rms))
                if rms < min_rms:
                    min_rms = rms

            threshold = min_rms * 1.05 + 1e-6
            valley_offsets = [offset + win_size // 2 for offset, rms in rms_list if rms <= threshold]
            if not valley_offsets:
                return target_ms
            mid_offset = (valley_offsets[0] + valley_offsets[-1]) // 2
            return scan_start_ms + int(mid_offset * 1000 / framerate)
    except Exception:
        return target_ms


def proportional_spans(start_ms: int, end_ms: int, parts: tuple[str, ...],
                       min_duration: int = 280,
                       audio_path: str | None = None) -> tuple[tuple[int, int], ...]:
    """Divide a multi-sentence duration proportionally by Japanese mora units with acoustic energy refinement."""
    n = len(parts)
    if n <= 1:
        return ((start_ms, end_ms),)
    total_duration = end_ms - start_ms
    if total_duration < n * min_duration:
        return ()

    weights = []
    for part in parts:
        mora = count_japanese_mora("", part)
        if mora <= 0:
            mora = max(1.0, float(len(_lexical(part))))
        weights.append(mora)

    total_weight = sum(weights)
    if total_weight <= 0:
        total_weight = float(n)
        weights = [1.0] * n

    edges = [start_ms]
    accumulated_weight = 0.0
    for i in range(n - 1):
        accumulated_weight += weights[i]
        target = start_ms + round(total_duration * (accumulated_weight / total_weight))
        min_bound = edges[-1] + min_duration
        max_bound = end_ms - (n - 1 - i) * min_duration
        if audio_path and max_bound > min_bound:
            refined = _refine_boundary_by_energy(audio_path, edges[-1], end_ms, target)
            target = max(min_bound, min(refined, max_bound))
        else:
            target = max(min_bound, min(target, max_bound))
        edges.append(target)
    edges.append(end_ms)

    return tuple(zip(edges, edges[1:]))


class MossDetectStage:
    def __init__(self, config: MossDetectConfig, audio_path: str, *,
                 client: WordAlignmentPort | None = None,
                 recalibrate_client: WordAlignmentPort | None = None):
        self.config, self.audio_path, self.client = config, audio_path, client
        self.recalibrate_client = recalibrate_client

    def _request_alignment(self, block) -> tuple[dict, int]:
        if self.client is not None:
            return self.client.request("/align", self.audio_path,
                {"text": block.text, "language": "Japanese"},
                start_ms=block.start_ms, end_ms=block.end_ms)
        if self.config.alignment_endpoint:
            client = AnalysisPluginClient(
                self.config.alignment_endpoint, "qwen-force-align", "qwen-force-align/1")
            return client.request("/align", self.audio_path,
                {"text": block.text, "language": "Japanese"},
                start_ms=block.start_ms, end_ms=block.end_ms)
        raise ValueError("no_alignment_endpoint")

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

    def _recalibrate_spans(self, block, parts: tuple[str, ...],
                           spans: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
        recalibrate_client = self.recalibrate_client
        if recalibrate_client is None and self.config.recalibrate_endpoint:
            try:
                recalibrate_client = AnalysisPluginClient(
                    self.config.recalibrate_endpoint, "moss-transcribe-diarize", "moss-transcribe-diarize/1"
                )
            except Exception:
                recalibrate_client = None

        if recalibrate_client is None:
            return spans

        new_spans = []
        for child_idx, (text, (start, end)) in enumerate(zip(parts, spans)):
            duration = end - start
            if duration < self.config.min_recalibrate_duration_ms:
                new_spans.append((start, end))
                continue

            try:
                body, _ = recalibrate_client.request(
                    "/transcribe", self.audio_path, {"language": "ja"},
                    start_ms=start, end_ms=end,
                )
                child_blocks = body.get("blocks", []) if isinstance(body, dict) else []
                if not child_blocks:
                    new_spans.append((start, end))
                    continue

                recal_text = "".join(b.get("text", "") for b in child_blocks)
                if _lexical(recal_text) != _lexical(text):
                    new_spans.append((start, end))
                    continue

                first_start_rel = child_blocks[0].get("start_ms", 0)
                last_end_rel = child_blocks[-1].get("end_ms", duration)
                cal_start = start + first_start_rel
                cal_end = start + last_end_rel

                if (cal_end - cal_start >= self.config.min_sentence_duration_ms
                        and start <= cal_start < cal_end <= end):
                    print(f"       [detect:moss] 二次微调句 {child_idx}: [{start}, {end}] -> [{cal_start}, {cal_end}]")
                    new_spans.append((cal_start, cal_end))
                else:
                    new_spans.append((start, end))
            except Exception:
                new_spans.append((start, end))

        return tuple(new_spans)

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
            method = "moss-native"
            if block.skip_tts:
                outcome = "bypassed"
            elif len(parts) > 1:
                if self.client is not None or self.config.alignment_endpoint:
                    try:
                        body, duration = self._request_alignment(block)
                        write_detect_artifact(output_dir, f"moss_alignment_raw_{position}.json", {
                            "blockIndex": block.index, "durationMs": duration, "response": body})
                        spans = self._spans(block, parts, body, duration)
                        outcome = "aligned"
                        method = "moss-native-word-align"
                    except (AnalysisPluginError, ValueError, TypeError, KeyError) as error:
                        if self.client is not None:
                            outcome = "fallback"
                            reason = getattr(error, "code", str(error))
                            print(f"       [detect:moss] 块 {block.index} 对齐失败，保留原块（{reason}）")
                        else:
                            outcome = "fallback"
                            reason = getattr(error, "code", str(error))

                if outcome != "aligned" and self.client is None:
                    spans = proportional_spans(block.start_ms, block.end_ms, parts,
                                               self.config.min_sentence_duration_ms,
                                               audio_path=self.audio_path)
                    if spans:
                        outcome = "aligned"
                        reason = None
                        method = "moss-native"
                        if self.config.recalibrate:
                            spans = self._recalibrate_spans(block, parts, spans)
                    else:
                        outcome = "fallback"
                        reason = "sentence_too_short"

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
                        method, "high", "sentence",
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
              f"单句 {summary['exactSlices']} 块，回退 {summary['fallbackSlices']} 块，输出 {len(output)} 句")
        return DetectStageResult(tuple(output), summary)
