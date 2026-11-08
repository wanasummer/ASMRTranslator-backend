"""Resolve a validated arrangement draft into an immutable audio plan."""
from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

from .assets import cached_tts, pipeline_tts_rates, resolve_replacement, tts_max_rate, tts_volume
from .contracts import ArrangementError, ArrangementState, ArrangementPlan, ArrangedClip, RenderedTake
from .state import aligned_rows, read_baseline, validate_state
from .synthesis import synthesize_edit
from ..speech_pace import landing_hint_ms


def fit_default_placement(start: int, sample_length: int, upper: float | None,
                          natural_rate: float, max_rate: float,
                          landing: int | None = None) -> tuple[int, int]:
    """Return ``(end_ms, kept_sample_ms)`` for an unedited voice starting at ``start``.

    The voice plays at ``natural_rate``; when that would reach ``upper`` it is
    sped up to at most ``max(max_rate, natural_rate)``. If it still does not
    fit, it ends at ``landing`` (bounded by ``upper``) and only the leading part
    that fits at the capped rate is kept.
    """
    end = start + round(sample_length / natural_rate)
    if upper is None or not start < upper < end:
        return end, sample_length
    cap = max(1.0, max_rate, natural_rate)
    room = round(upper) - start
    if sample_length / room > cap and landing is not None and start < landing < upper:
        room = landing - start
    rate = min(sample_length / room, cap)
    return start + room, min(sample_length, round(room * rate))


def prepare_arrangement_plan(work: Path, source_audio: Path, state: ArrangementState, config: dict, kwargs: dict) -> ArrangementPlan:
    """Resolve cached and regenerated assets into an explicit, immutable audio plan."""
    from pydub import AudioSegment
    validate_state(work, state)
    rows, _ = read_baseline(work)
    pipeline_rates = pipeline_tts_rates(work)
    max_rate = tts_max_rate(config)
    voice_volume = tts_volume(config)
    clips, takes = [], []
    reused = replaced = 0
    previous_end = -math.inf
    pairs = aligned_rows(rows, state["segments"])
    timeline = [row for _, row in pairs]
    for position, ((source_position, baseline), edit) in enumerate(zip(pairs, state["segments"])):
        if edit.get("deleted", False):
            continue
        replacement = edit.get("replacement")
        sample = edit.get("ttsSample")
        interval = edit.get("interval")
        fresh_take = False
        if replacement and replacement["text"] == edit["text"]:
            path = resolve_replacement(work, replacement["path"])
            replaced += 1
        elif source_position is None and not edit["text"].strip():
            continue
        elif edit["text"] != baseline["text"]:
            replacement = asyncio.run(synthesize_edit(work, {**baseline, **edit}, config, kwargs))
            path = resolve_replacement(work, replacement["path"])
            fresh_take = True
            replaced += 1
        else:
            path = cached_tts(work, source_position)
            if path is None:
                # Original-only interjections have no generated TTS cache.
                if baseline.get("skip_tts", len(re.findall(r"[\u4e00-\u9fff]", baseline["text"])) <= 3):
                    continue
                raise ArrangementError(f"片段 {baseline['index']} 的 TTS 缓存不存在")
            reused += 1
        anchor = round(interval["start"]) if interval is not None else baseline["start_ms"]
        if fresh_take:
            # The saved sample, interval end and pipeline tempo were all fitted to the previous take;
            # the draft keeps them for restore, but the new take only inherits where the fragment starts.
            sample, interval = None, None
        audio_length = len(AudioSegment.from_file(path)) if sample is not None or interval is None else None
        sample_start = round(sample["start"]) if sample else 0
        sample_end = round(sample["end"]) if sample else audio_length
        if sample is not None and (sample_start < 0 or sample_end > audio_length or sample_end <= sample_start):
            raise ArrangementError(f"片段 {baseline['index']} 的 TTS 取样超出音频范围")
        if interval is None:
            start = anchor
            next_sentence_start = next(
                (
                    round(later["interval"]["start"] if later.get("interval") is not None else later_baseline["start_ms"])
                    for later_baseline, later in zip(timeline[position + 1:], state["segments"][position + 1:])
                    if not later.get("deleted", False)
                ),
                None,
            )
            natural_rate = 1 if fresh_take else pipeline_rates.get(baseline["index"], 1)
            end, kept = fit_default_placement(start, sample_end - sample_start, next_sentence_start,
                                              natural_rate, max_rate,
                                              landing_hint_ms(edit["text"], start, start + baseline["end_ms"] - baseline["start_ms"]))
            sample_end = sample_start + kept
        else:
            start, end = round(interval["start"]), round(interval["end"])
        if start < previous_end:
            raise ArrangementError(f"片段 {baseline['index']} 与前一片段重叠")
        if not clips and end < -5000:
            raise ArrangementError("首段 TTS 与原声的间隔不能超过 5 秒")
        previous_end = end
        volume = edit["volume"] * state["globalVolume"] * voice_volume
        reference = edit.get("originalInterval") or baseline
        clips.append(ArrangedClip(baseline["index"], start, end, edit["text"], path, volume,
                                  round(reference.get("start", reference.get("start_ms"))),
                                  round(reference.get("end", reference.get("end_ms"))), sample_start, sample_end))
        if fresh_take:
            takes.append(RenderedTake(edit["index"], edit["text"], replacement["path"], start, end, sample_end))
    deleted = sum(bool(edit.get("deleted", False)) for edit in state["segments"])
    inserted = len(state["segments"]) - len(rows)
    print(f"[arrangement] 共 {len(rows)} 段，新增 {inserted} 段，删除 {deleted} 段，复用 TTS 缓存 {reused} 段，替换 {replaced} 段")
    return ArrangementPlan(source_audio, tuple(clips), False, tuple(takes))
