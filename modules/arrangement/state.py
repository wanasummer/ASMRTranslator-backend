"""Arrangement draft state: baseline rows, validation, persistence and the published subtitle base."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .assets import resolve_replacement
from .contracts import ArrangementError, ArrangementState
from .repository import ArrangementRepository, create_arrangement_repository
from ..storage import save_json, read_bytes, read_json


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


def inserted_row(edit: dict) -> dict:
    """Baseline-shaped row for a user-added fragment; it has no subtitle or cached audio."""
    inserted = edit["inserted"]
    row = {"index": edit["index"], "start_ms": inserted["startMs"], "end_ms": inserted["endMs"],
           "text": "", "source_text": ""}
    if inserted.get("sourceBlockId"):
        row["source_block_id"] = inserted["sourceBlockId"]
    if inserted.get("speaker"):
        row["speaker"] = inserted["speaker"]
    return row


def aligned_rows(rows: list[dict], edits: list[dict]) -> list[tuple[int | None, dict]]:
    """Pair each draft position with its baseline position (None when inserted) and row."""
    result, source = [], 0
    for edit in edits:
        if edit.get("inserted"):
            result.append((None, inserted_row(edit)))
        else:
            result.append((source, rows[source]))
            source += 1
    return result


def fragment_tts(edit: dict) -> dict | None:
    """The fragment's own TTS profile, or None when it follows the generation profile."""
    return edit.get("tts") or None


def replacement_matches(edit: dict) -> bool:
    """Whether the stored replacement was synthesized from the fragment's current text and TTS profile."""
    replacement = edit.get("replacement")
    return bool(replacement) and replacement["text"] == edit["text"] and fragment_tts(replacement) == fragment_tts(edit)


def _valid_tts_profile(value) -> bool:
    return value is None or (isinstance(value, dict) and isinstance(value.get("ttsBackend"), str) and bool(value["ttsBackend"]))


def block_id(edit: dict, row: dict) -> str:
    """Effective block of a draft fragment; ``row`` is its ``aligned_rows`` row."""
    return edit.get("blockId") or row.get("source_block_id") or f"row-{row['index']}"


def _validate_inserted(edit: dict) -> None:
    inserted = edit["inserted"]
    if not isinstance(inserted, dict) or isinstance(edit.get("index"), bool) or not isinstance(edit.get("index"), int):
        raise ArrangementError("新增片段格式无效")
    start, end = inserted.get("startMs"), inserted.get("endMs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
        raise ArrangementError("新增片段时间范围无效")
    if any(inserted.get(key) is not None and not isinstance(inserted[key], str) for key in ("sourceBlockId", "speaker")):
        raise ArrangementError("新增片段归属格式无效")


def validate_state(work: Path, state: ArrangementState) -> ArrangementState:
    rows, version = read_baseline(work)
    if state.get("schemaVersion") != 1 or state.get("sourceVersion") != version:
        raise ArrangementError("来源字幕已变化，请重新载入后编排")
    edits = state.get("segments", [])
    baseline_edits = [edit for edit in edits if not edit.get("inserted")]
    if len(baseline_edits) != len(rows) or [r.get("index") for r in baseline_edits] != [r["index"] for r in rows]:
        raise ArrangementError("编排片段标识或顺序与来源不一致")
    for edit in edits:
        if edit.get("inserted"):
            _validate_inserted(edit)
    if len({edit["index"] for edit in edits}) != len(edits):
        raise ArrangementError("编排片段标识重复")
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
        if edit.get("blockId") is not None and (not isinstance(edit["blockId"], str) or not edit["blockId"]):
            raise ArrangementError("片段分组标识无效")
        if not _valid_tts_profile(edit.get("tts")):
            raise ArrangementError("片段单独配音配置无效")
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
            if not isinstance(replacement.get("text"), str) or not _valid_tts_profile(replacement.get("tts")):
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


SUBTITLE_BASE = "arrangement/subtitle_base.json"


def publish_subtitle_base(work: Path, state: ArrangementState) -> None:
    """Replace the single subtitle base with the texts of a successfully rendered arrangement."""
    save_json(str(work / SUBTITLE_BASE), {"sourceVersion": state["sourceVersion"],
                                          "texts": {str(edit["index"]): edit["text"] for edit in state["segments"]}})


def subtitle_base(work: Path) -> dict[int, str]:
    """Per-fragment text of the audio currently treated as the source.

    The last published arrangement for the current baseline, else the original pipeline subtitles.
    Fragments absent from the base (later insertions) have no entry.
    """
    rows, version = read_baseline(work)
    try:
        published = read_json(work / SUBTITLE_BASE)
    except (OSError, ValueError):
        published = None
    if isinstance(published, dict) and published.get("sourceVersion") == version:
        return {int(index): text for index, text in published.get("texts", {}).items()}
    return {row["index"]: row["text"] for row in rows}
