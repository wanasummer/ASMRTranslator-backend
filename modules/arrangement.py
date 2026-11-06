"""Project-backed arrangement contracts and subtitle/TTS editing adapters."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .arrangement_repository import ArrangementRepository, create_arrangement_repository
from .arrangement_contracts import ArrangementError, ArrangementState, ArrangementPlan, ArrangedClip
from .speech_pace import landing_hint_ms
from .storage import save_json, read_bytes, read_json, address
from .llm import LLMConfig, is_configured
from .subtitle_processing.editor import create_subtitle_editor
from .tts_engine import get_tts_adapter, prepare_text_for_synthesis
from .vtt_parser import VTTBlock


class SubtitleOptimizer(Protocol):
    async def optimize_from_japanese(self, block: VTTBlock, japanese: str) -> str: ...
    async def optimize_japanese_text(self, block: VTTBlock, japanese: str) -> str: ...
    async def optimize_chinese_text(self, block: VTTBlock, chinese: str) -> str: ...


def read_baseline(work: Path) -> tuple[list[dict], str]:
    path = work / "debug/_parsed_vtt.json"
    try:
        raw = read_bytes(path)
        rows = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ArrangementError("没有可读取的片段字幕，请先完成自动配音") from exc
    if not isinstance(rows, list) or not rows:
        raise ArrangementError("片段字幕清单为空或无效")
    return rows, hashlib.sha256(raw).hexdigest()


def initial_state(work: Path) -> ArrangementState:
    rows, version = read_baseline(work)
    return {"schemaVersion": 1, "sourceVersion": version, "loudnessScope": "segment",
            "globalVolume": 1, "segments": [
                {"index": row["index"], "text": row["text"],
                 "japanese": row.get("source_text", ""), "volume": 1,
                 "interval": None, "originalInterval": None, "ttsSample": None,
                 "replacement": None, "deleted": False} for row in rows]}


def validate_state(work: Path, state: ArrangementState) -> ArrangementState:
    rows, version = read_baseline(work)
    if state.get("schemaVersion") != 1 or state.get("sourceVersion") != version:
        raise ArrangementError("来源字幕已变化，请重新载入后编排")
    edits = state.get("segments", [])
    if len(edits) != len(rows) or [r.get("index") for r in edits] != [r["index"] for r in rows]:
        raise ArrangementError("编排片段标识或顺序与来源不一致")
    if state.get("loudnessScope") not in ("segment", "global"):
        raise ArrangementError("响度模式无效")
    for value in [state.get("globalVolume"), *[row.get("volume") for row in edits]]:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ArrangementError("响度比例必须在 0%–200% 之间")
    previous_end = -math.inf
    first_active = next((i for i, edit in enumerate(edits) if not edit.get("deleted", False)), None)
    for position, edit in enumerate(edits):
        if not isinstance(edit.get("deleted", False), bool):
            raise ArrangementError("片段删除状态无效")
        original_interval = edit.get("originalInterval")
        if original_interval is not None:
            start, end = original_interval.get("start"), original_interval.get("end")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
                raise ArrangementError("原声取样区间格式无效")
        tts_sample = edit.get("ttsSample")
        if tts_sample is not None:
            start, end = tts_sample.get("start"), tts_sample.get("end")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
                raise ArrangementError("TTS 取样区间格式无效")
        if not isinstance(edit.get("text"), str) or not isinstance(edit.get("japanese"), str):
            raise ArrangementError("字幕草稿格式无效")
        interval = edit.get("interval")
        if interval is not None:
            start, end = interval.get("start"), interval.get("end")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
                raise ArrangementError("TTS 区间格式无效")
            if end <= start or (not edit.get("deleted", False) and (start < previous_end or (position != first_active and start < 0))):
                raise ArrangementError("TTS 区间重叠或长度无效")
            if position == first_active and end < -5000:
                raise ArrangementError("首段 TTS 与原声的间隔不能超过 5 秒")
            if not edit.get("deleted", False):
                previous_end = end
        replacement = edit.get("replacement")
        if replacement:
            resolve_replacement(work, replacement["path"])
            if not isinstance(replacement.get("text"), str):
                raise ArrangementError("重新生成记录格式无效")
    return state


def load_state(work: Path, repository: ArrangementRepository | None = None) -> ArrangementState:
    saved = (repository or create_arrangement_repository(work)).load()
    baseline = initial_state(work)
    if saved is None or saved.get("sourceVersion") != baseline["sourceVersion"]:
        return baseline
    return validate_state(work, saved)


def save_state(work: Path, state: ArrangementState, repository: ArrangementRepository | None = None) -> ArrangementState:
    validate_state(work, state)
    (repository or create_arrangement_repository(work)).save(state)
    return state


def draft_lock(work: Path):
    return create_arrangement_repository(work).transaction()


def commit_optimized_fragment(work: Path, payload: dict) -> dict:
    """Merge a background result into the latest draft, preserving concurrent edits."""
    with draft_lock(work) as repository:
        state = load_state(work, repository)
        position = payload["position"]
        expected, optimized = payload["fragment"], payload["optimizedFragment"]
        if (state["sourceVersion"] != payload["sourceVersion"] or not isinstance(position, int)
                or not 0 <= position < len(state["segments"])):
            raise ArrangementError("来源字幕已变化，请重新载入")
        current = state["segments"][position]
        if current.get("deleted", False):
            return {"edit": current, "applied": False}
        if current["index"] != expected["index"] or optimized["index"] != expected["index"]:
            raise ArrangementError("优化片段标识不匹配")
        mode = payload["mode"]
        if mode == "auto":
            mode = "translate" if expected["japanese"].strip() else "chinese"
        fields = ("japanese",) if mode == "japanese" else ("text", "japanese") if mode == "translate" else ("text",)
        if mode not in ("translate", "chinese", "japanese"):
            raise ArrangementError("未知字幕优化模式")
        if any(current[field] != expected[field] for field in fields):
            return {"edit": current, "applied": False}
        target = "japanese" if mode == "japanese" else "text"
        if not isinstance(optimized.get(target), str) or not optimized[target].strip():
            raise ArrangementError("优化结果为空")
        current[target] = optimized[target]
        save_state(work, state, repository)
        return {"edit": current, "applied": True}


def resolve_replacement(work: Path, relative: str) -> Path:
    path = (work / relative).resolve()
    if not path.is_relative_to((work / "arrangement/audio").resolve()) or not path.is_file():
        raise ArrangementError("替换音频不存在或超出编排目录")
    return path


def voice_asset_path(work: Path, position: int) -> Path:
    """The processed (artifact-checked, mode-DSP) voice that the timeline renders."""
    return work / f"chunk_0/02_tts/voice_{position:04d}.wav"


def cached_tts(work: Path, position: int) -> Path | None:
    # Matches the studio audio track lookup order; never confuse index with position.
    candidates = (voice_asset_path(work, position),
                  *(work / f"chunk_0/02_tts/tts_{position:04d}.{extension}" for extension in ("mp3", "wav")))
    return next((path for path in candidates if path.is_file()), None)


def pipeline_tts_rates(work: Path) -> dict[int, float]:
    target = address(work / "chunk_0/02_tts/tts_tempo_")
    rates = {}
    for key in target.store.keys(target.scope, target.key):
        value = read_json(work / key)
        index = int(Path(key).stem.removeprefix("tts_tempo_"))
        rate = float(value["rate"])
        if math.isfinite(rate) and rate > 0:
            rates[index] = rate
    return rates


def tts_volume(config: dict) -> float:
    """The user's TTS volume, the same multiplier the pipeline applies to every voice."""
    try:
        value = float(config.get("ttsVolume", 1))
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(2.0, value)) if math.isfinite(value) else 1.0


def tts_max_rate(config: dict) -> float:
    """The user's "最大加速倍率", clamped to the same range as the pipeline."""
    try:
        value = float(config.get("followOriginalRate", 1.15))
    except (TypeError, ValueError):
        return 1.15
    return max(1.0, min(1.25, value)) if math.isfinite(value) else 1.15


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


def create_optimizer(config: dict) -> SubtitleOptimizer:
    cfg = LLMConfig(provider=config.get("llmProvider", ""),
                    base_url=config.get("llmBaseUrl", ""),
                    api_key=config.get("llmApiKey", ""), model=config.get("llmModel", ""),
                    tts_backend=config.get("ttsBackend", ""),
                    tts_model=config.get("minimaxModel", "") if config.get("ttsBackend") == "minimax-tts" else config.get("fishModel", ""))
    if not is_configured(cfg):
        raise ArrangementError("请先在识别与实验设置中配置字幕优化大模型")
    return create_subtitle_editor(cfg)


async def optimize_fragment(row: dict, client: SubtitleOptimizer, mode: str) -> dict:
    """Independent fragment output, safe to merge after concurrent requests finish."""
    result = json.loads(json.dumps(row))
    block = VTTBlock(row["index"], 0, 1, row["text"])
    if mode == "auto":
        mode = "translate" if row["japanese"].strip() else "chinese"
    if mode == "translate":
        result["text"] = await client.optimize_from_japanese(block, row["japanese"])
    elif mode == "chinese":
        if not row["text"].strip():
            raise ArrangementError("中文字幕为空，无法优化")
        result["text"] = await client.optimize_chinese_text(block, row["text"])
    elif mode == "japanese":
        result["japanese"] = await client.optimize_japanese_text(block, row["japanese"])
    else:
        raise ArrangementError("未知字幕优化模式")
    return result


async def optimize_subtitles(state: ArrangementState, client: SubtitleOptimizer, position: int | None = None) -> ArrangementState:
    """Return new values; a failed request never partially changes the input draft."""
    result = json.loads(json.dumps(state))
    edits = result["segments"]
    if position is not None:
        if not 0 <= position < len(edits):
            raise ArrangementError("片段位置无效")
        row = edits[position]
        if row.get("deleted", False):
            raise ArrangementError("已删除片段不能优化")
        block = VTTBlock(row["index"], 0, 1, row["text"])
        row["text"] = await client.optimize_from_japanese(block, row["japanese"])
    else:
        active = [position for position, row in enumerate(edits) if not row.get("deleted", False)]
        optimized = await asyncio.gather(*(optimize_fragment(edits[position], client, "auto") for position in active))
        for position, edit in zip(active, optimized):
            edits[position] = edit
    return result


async def synthesize_edit(work: Path, row: dict, config: dict, kwargs: dict, adapter=None) -> dict:
    """Always create a fresh immutable TTS asset; never overwrite baseline cache."""
    if not row["text"].strip():
        raise ArrangementError("中文字幕为空，无法重新生成")
    adapter = adapter or get_tts_adapter(config.get("ttsBackend", "edge-tts"), kwargs)
    directory = work / "arrangement/audio"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}{adapter.output_extension}"
    try:
        await adapter.synthesize(prepare_text_for_synthesis(adapter, row["text"]),
                                 adapter.get_voice_for_speaker(row.get("speaker")), str(path),
                                 rate=config.get("rate", "+0%"), pitch=config.get("edgePitch", "+0Hz"))
    finally:
        await adapter.aclose()
    if not path.is_file() or path.stat().st_size == 0:
        raise ArrangementError("TTS 未返回有效音频")
    return {"path": path.relative_to(work).as_posix(), "text": row["text"]}


def run_editor_request(payload: dict) -> dict:
    """Sidecar editor entry: network config only arrives through stdin."""
    work = Path(payload["workDir"]).resolve()
    action = payload["action"]
    if action == "load":
        return {"state": load_state(work)}
    if action == "commit-optimization":
        return commit_optimized_fragment(work, payload)
    if action == "optimize-fragment":
        rows, version = read_baseline(work)
        position = payload.get("position")
        fragment = payload.get("fragment")
        if (payload.get("sourceVersion") != version or not isinstance(position, int)
                or not 0 <= position < len(rows) or not isinstance(fragment, dict)
                or fragment.get("index") != rows[position]["index"]):
            raise ArrangementError("来源字幕或片段标识已变化，请重新载入")
        if not isinstance(fragment.get("text"), str) or not isinstance(fragment.get("japanese"), str):
            raise ArrangementError("字幕草稿格式无效")
        audit = work / f"arrangement/audit/{uuid4().hex}"
        save_json(str(audit.with_suffix(".before.json")), fragment)
        try:
            edit = asyncio.run(optimize_fragment(fragment, create_optimizer(payload["config"]), payload["mode"]))
        except Exception as error:
            from .error import unexpected_error
            save_json(str(audit.with_suffix(".error.json")), unexpected_error(error).to_payload())
            raise
        save_json(str(audit.with_suffix(".after.json")), edit)
        # A separate commit merges only target text into the latest saved draft.
        return {"edit": edit}
    state = validate_state(work, payload["state"])
    if action == "save":
        with draft_lock(work) as repository:
            return {"state": save_state(work, state, repository)}
    if action in ("optimize", "optimize-all"):
        save_json(str(work / f"arrangement/audit/{uuid4().hex}-before.json"), state)
        state = asyncio.run(optimize_subtitles(state, create_optimizer(payload["config"]),
                                              payload.get("position") if action == "optimize" else None))
        save_json(str(work / f"arrangement/audit/{uuid4().hex}-after.json"), state)
        return {"state": save_state(work, state)}
    if action == "regenerate":
        position = payload["position"]
        if not isinstance(position, int) or not 0 <= position < len(state["segments"]):
            raise ArrangementError("片段位置无效")
        rows, _ = read_baseline(work)
        edit = state["segments"][position]
        replacement = asyncio.run(synthesize_edit(work, {**rows[position], **edit},
                                                  payload["config"], payload["ttsKwargs"]))
        with draft_lock(work) as repository:
            latest = load_state(work, repository)
            if latest["segments"][position]["text"] == edit["text"]:
                latest["segments"][position]["replacement"] = replacement
                latest["segments"][position]["ttsSample"] = None
            return {"state": save_state(work, latest, repository)}
    raise ArrangementError("未知编排操作")


def prepare_arrangement_plan(work: Path, source_audio: Path, state: ArrangementState, config: dict, kwargs: dict) -> ArrangementPlan:
    """Resolve cached and regenerated assets into an explicit, immutable audio plan."""
    from pydub import AudioSegment
    validate_state(work, state)
    rows, _ = read_baseline(work)
    pipeline_rates = pipeline_tts_rates(work)
    max_rate = tts_max_rate(config)
    voice_volume = tts_volume(config)
    clips = []
    reused = replaced = 0
    previous_end = -math.inf
    for position, (baseline, edit) in enumerate(zip(rows, state["segments"])):
        if edit.get("deleted", False):
            continue
        replacement = edit.get("replacement")
        if replacement and replacement["text"] == edit["text"]:
            path = resolve_replacement(work, replacement["path"])
            replaced += 1
        elif edit["text"] != baseline["text"]:
            replacement = asyncio.run(synthesize_edit(work, {**baseline, **edit}, config, kwargs))
            path = resolve_replacement(work, replacement["path"])
            replaced += 1
        else:
            path = cached_tts(work, position)
            if path is None:
                # Original-only interjections have no generated TTS cache.
                if baseline.get("skip_tts", len(re.findall(r"[\u4e00-\u9fff]", baseline["text"])) <= 3):
                    continue
                raise ArrangementError(f"片段 {baseline['index']} 的 TTS 缓存不存在")
            reused += 1
        sample = edit.get("ttsSample")
        interval = edit.get("interval")
        audio_length = len(AudioSegment.from_file(path)) if sample is not None or interval is None else None
        sample_start = round(sample["start"]) if sample else 0
        sample_end = round(sample["end"]) if sample else audio_length
        if sample is not None and (sample_start < 0 or sample_end > audio_length or sample_end <= sample_start):
            raise ArrangementError(f"片段 {baseline['index']} 的 TTS 取样超出音频范围")
        if interval is None:
            start = baseline["start_ms"]
            next_sentence_start = next(
                (
                    round(later["interval"]["start"] if later.get("interval") is not None else later_baseline["start_ms"])
                    for later_baseline, later in zip(rows[position + 1:], state["segments"][position + 1:])
                    if not later.get("deleted", False)
                ),
                None,
            )
            end, kept = fit_default_placement(start, sample_end - sample_start, next_sentence_start,
                                              pipeline_rates.get(baseline["index"], 1), max_rate,
                                              landing_hint_ms(edit["text"], baseline["start_ms"], baseline["end_ms"]))
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
    deleted = sum(bool(edit.get("deleted", False)) for edit in state["segments"])
    print(f"[arrangement] 共 {len(rows)} 段，删除 {deleted} 段，复用 TTS 缓存 {reused} 段，替换 {replaced} 段")
    return ArrangementPlan(source_audio, tuple(clips), False)
