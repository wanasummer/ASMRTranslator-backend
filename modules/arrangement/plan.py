"""Resolve a validated arrangement draft into an immutable audio plan."""
from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

from .assets import cached_original, cached_tts, pipeline_tts_rates, resolve_replacement, tts_max_rate, tts_volume
from .contracts import ArrangementError, ArrangementState, ArrangementPlan, ArrangedClip, RenderedTake
from .state import aligned_rows, block_id, fragment_tts, read_baseline, replacement_matches, validate_state
from .synthesis import resolve_voice, synthesize_edit
from ..speech_pace import landing_hint_ms


def resolve_listen_mode(config: dict) -> str:
    """Resolve the arrangement listen mode from the serialized app config."""
    if config.get("phantomMode") or config.get("syncEnabled"):
        return "overlay"
    if config.get("retainOriginal"):
        return "dual"
    return "overlay"


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


def _resolve_dual_entries(entries: list[dict], gap_ms: int = 200) -> list[ArrangedClip]:
    """Assign [orig -> gap -> tts] coarse-group timeline coordinates for retain-original mode."""
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["block_id"], []).append(entry)
    clips: list[ArrangedClip] = []
    for grp_id, group in groups.items():
        cursor = 0
        for entry in group:
            dual = entry["dual"] or {}
            orig_deleted = bool(dual.get("originalDeleted", False)) or entry["is_inserted"]
            orig_len = entry["orig_length"]
            if not entry["is_inserted"] and orig_len > 0:
                orig_iv = dual.get("originalInterval")
                if orig_iv is not None:
                    d_orig_start, d_orig_end = round(orig_iv["start"]), round(orig_iv["end"])
                else:
                    d_orig_start = cursor
                    d_orig_end = cursor + (entry["orig_sample_end"] - entry["orig_sample_start"])
                if not orig_deleted:
                    cursor = max(cursor, d_orig_end)
            else:
                d_orig_start = d_orig_end = None

            tts_deleted = bool(dual.get("ttsDeleted", False)) or entry["path"] is None
            if entry["path"] is not None:
                tts_iv = dual.get("ttsInterval")
                if tts_iv is not None:
                    d_tts_start, d_tts_end = round(tts_iv["start"]), round(tts_iv["end"])
                else:
                    default_start = cursor + gap_ms if (not orig_deleted and not entry["is_inserted"] and orig_len > 0) else cursor
                    d_tts_start = default_start
                    d_tts_end = default_start + max(1, entry["sample_end"] - entry["sample_start"])
                if not tts_deleted:
                    cursor = max(cursor, d_tts_end)
            else:
                d_tts_start = d_tts_end = None

            trailing_gap = max(0, round(dual.get("trailingGapMs") or 0))
            clips.append(ArrangedClip(
                entry["index"], entry["start"], entry["end"], entry["text"], entry["path"], entry["volume"],
                entry["ref_start"], entry["ref_end"], entry["sample_start"], entry["sample_end"],
                block_id=grp_id,
                dual_original_start_ms=d_orig_start,
                dual_original_end_ms=d_orig_end,
                dual_tts_start_ms=d_tts_start,
                dual_tts_end_ms=d_tts_end,
                original_sample_start_ms=entry["orig_sample_start"],
                original_sample_end_ms=entry["orig_sample_end"],
                original_deleted=orig_deleted,
                tts_deleted=tts_deleted,
                trailing_gap_ms=trailing_gap,
                original_path=entry["orig_path"],
            ))
    return clips


def prepare_arrangement_plan(work: Path, source_audio: Path, state: ArrangementState, config: dict, kwargs: dict,
                             overrides: dict | None = None) -> ArrangementPlan:
    """Resolve cached and regenerated assets into an explicit, immutable audio plan.

    ``config``/``kwargs`` describe the generation profile used for fragments without their own TTS profile;
    ``overrides`` holds the resolved settings of fragments that have one (see ``resolve_voice``).
    """
    from pydub import AudioSegment
    validate_state(work, state)
    rows, _ = read_baseline(work)
    pipeline_rates = pipeline_tts_rates(work)
    max_rate = tts_max_rate(config)
    voice_volume = tts_volume(config)
    listen_mode = resolve_listen_mode(config)
    clips, takes, dual_entries = [], [], []
    reused = replaced = 0
    previous_end = -math.inf
    pairs = aligned_rows(rows, state["segments"])
    timeline = [row for _, row in pairs]
    for position, ((source_position, baseline), edit) in enumerate(zip(pairs, state["segments"])):
        if edit.get("deleted", False):
            continue
        dual = (edit.get("dual") or {}) if listen_mode == "dual" else {}
        tts_deleted = bool(dual.get("ttsDeleted", False))
        replacement = edit.get("replacement")
        sample = edit.get("ttsSample")
        interval = edit.get("interval")
        fresh_take = False
        if tts_deleted:
            path = None
        elif replacement_matches(edit):
            path = resolve_replacement(work, replacement["path"])
            replaced += 1
        elif source_position is None and not edit["text"].strip():
            if listen_mode != "dual":
                continue
            path = None
        elif edit["text"] != baseline["text"] or fragment_tts(edit) is not None:
            voice = resolve_voice(edit, config, kwargs, overrides)
            replacement = asyncio.run(synthesize_edit(work, {**baseline, **edit}, voice))
            path = resolve_replacement(work, replacement["path"])
            fresh_take = True
            replaced += 1
        else:
            path = cached_tts(work, source_position)
            if path is None:
                # Original-only interjections have no generated TTS cache.
                if baseline.get("skip_tts", len(re.findall(r"[\u4e00-\u9fff]", baseline["text"])) <= 3):
                    if listen_mode != "dual":
                        continue
                else:
                    raise ArrangementError(f"片段 {baseline['index']} 的 TTS 缓存不存在")
            else:
                reused += 1
        anchor = round(interval["start"]) if interval is not None else baseline["start_ms"]
        if fresh_take:
            # The saved sample, interval end and pipeline tempo were all fitted to the previous take;
            # the draft keeps them for restore, but the new take only inherits where the fragment starts.
            sample, interval = None, None
        if path is not None:
            audio_length = len(AudioSegment.from_file(path)) if sample is not None or interval is None else None
            sample_start = round(sample["start"]) if sample else 0
            sample_end = round(sample["end"]) if sample else audio_length
            if sample is not None and (sample_start < 0 or sample_end > audio_length or sample_end <= sample_start):
                raise ArrangementError(f"片段 {baseline['index']} 的 TTS 取样超出音频范围")
            if interval is None:
                start = anchor
                next_sentence_start = None if listen_mode == "dual" else next(
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
        else:
            sample_start, sample_end = 0, None
            start, end = round(baseline["start_ms"]), round(baseline["end_ms"])
        if listen_mode != "dual":
            if start < previous_end:
                raise ArrangementError(f"片段 {baseline['index']} 与前一片段重叠")
            if not clips and end < -5000:
                raise ArrangementError("首段 TTS 与原声的间隔不能超过 5 秒")
        previous_end = end
        volume = edit["volume"] * state["globalVolume"] * voice_volume
        reference = edit.get("originalInterval") or baseline
        ref_start = round(reference.get("start", reference.get("start_ms")))
        ref_end = round(reference.get("end", reference.get("end_ms")))
        if listen_mode == "dual":
            orig_path = cached_original(work, source_position) if source_position is not None else None
            orig_length = len(AudioSegment.from_file(orig_path)) if orig_path is not None else max(0, ref_end - ref_start)
            orig_sample = dual.get("originalSample")
            orig_sample_start = round(orig_sample["start"]) if orig_sample else 0
            orig_sample_end = round(orig_sample["end"]) if orig_sample else orig_length
            if orig_sample is not None and (orig_sample_start < 0 or orig_sample_end > orig_length or orig_sample_end <= orig_sample_start):
                raise ArrangementError(f"片段 {baseline['index']} 的原声取样超出音频范围")
            dual_entries.append({
                "index": baseline["index"],
                "block_id": block_id(edit, baseline),
                "is_inserted": source_position is None,
                "dual": dual,
                "start": start,
                "end": end,
                "text": edit["text"],
                "path": path,
                "volume": volume,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "orig_path": orig_path,
                "orig_length": orig_length,
                "orig_sample_start": orig_sample_start,
                "orig_sample_end": orig_sample_end,
            })
        else:
            clips.append(ArrangedClip(baseline["index"], start, end, edit["text"], path, volume,
                                      ref_start, ref_end, sample_start, sample_end))
        if fresh_take and path is not None:
            takes.append(RenderedTake(edit["index"], edit["text"], replacement["path"], start, end, sample_end,
                                      fragment_tts(edit)))
    if listen_mode == "dual":
        clips = _resolve_dual_entries(dual_entries)
    deleted = sum(bool(edit.get("deleted", False)) for edit in state["segments"])
    inserted = len(state["segments"]) - len(rows)
    print(f"[arrangement] 共 {len(rows)} 段，新增 {inserted} 段，删除 {deleted} 段，复用 TTS 缓存 {reused} 段，替换 {replaced} 段")
    return ArrangementPlan(source_audio, tuple(clips), False, tuple(takes), listen_mode=listen_mode)
