"""LLM subtitle optimization of arrangement drafts: per fragment, whole draft, and batch restructuring."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from .contracts import ArrangementError, ArrangementState
from .restructure import apply_restructure_plan, restructure_rows
from .state import aligned_rows, block_id, draft_lock, load_state, read_baseline, save_state, validate_state
from ..llm import LLMConfig, is_configured
from ..llm.contracts import RestructurePlan, TimedSourceRow
from ..storage import save_json
from ..subtitle_processing.editor import create_subtitle_editor
from ..vtt_parser import VTTBlock


class SubtitleOptimizer(Protocol):
    async def optimize_from_japanese(self, block: VTTBlock, japanese: str) -> str: ...
    async def optimize_japanese_text(self, block: VTTBlock, japanese: str) -> str: ...
    async def optimize_chinese_text(self, block: VTTBlock, chinese: str) -> str: ...
    async def summarize(self, rows: Sequence[TimedSourceRow]) -> str: ...
    async def restructure(self, rows: Sequence[TimedSourceRow], summary: str, instruction: str) -> RestructurePlan: ...


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


def optimize_draft(work: Path, state: ArrangementState, config: dict, position: int | None) -> ArrangementState:
    """Optimize one position (or every active fragment when ``None``) and save the audited result."""
    save_json(str(work / f"arrangement/audit/{uuid4().hex}-before.json"), state)
    state = asyncio.run(optimize_subtitles(state, create_optimizer(config), position))
    save_json(str(work / f"arrangement/audit/{uuid4().hex}-after.json"), state)
    return save_state(work, state)


def optimize_submitted_fragment(work: Path, payload: dict) -> dict:
    """Optimize a fragment snapshot without saving; ``commit_optimized_fragment`` merges it later."""
    rows, version = read_baseline(work)
    aligned = aligned_rows(rows, load_state(work)["segments"])
    position = payload.get("position")
    fragment = payload.get("fragment")
    if (payload.get("sourceVersion") != version or not isinstance(position, int)
            or not 0 <= position < len(aligned) or not isinstance(fragment, dict)
            or fragment.get("index") != aligned[position][1]["index"]):
        raise ArrangementError("来源字幕或片段标识已变化，请重新载入")
    if not isinstance(fragment.get("text"), str) or not isinstance(fragment.get("japanese"), str):
        raise ArrangementError("字幕草稿格式无效")
    audit = work / f"arrangement/audit/{uuid4().hex}"
    save_json(str(audit.with_suffix(".before.json")), fragment)
    try:
        edit = asyncio.run(optimize_fragment(fragment, create_optimizer(payload["config"]), payload["mode"]))
    except Exception as error:
        from ..error import unexpected_error
        save_json(str(audit.with_suffix(".error.json")), unexpected_error(error).to_payload())
        raise
    save_json(str(audit.with_suffix(".after.json")), edit)
    return {"edit": edit}


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


def _active_fragments(work: Path, snapshot: ArrangementState):
    rows, _ = read_baseline(work)
    return restructure_rows(snapshot, aligned_rows(rows, snapshot["segments"]))


def summarize_arrangement(work: Path, payload: dict, optimizer: SubtitleOptimizer | None = None) -> dict:
    """Whole-plot summary for later batches; like the pipeline stage, failure degrades to no summary."""
    fragments = _active_fragments(work, validate_state(work, payload["state"]))
    if not fragments:
        raise ArrangementError("没有可优化的片段")
    client = optimizer or create_optimizer(payload["config"])
    try:
        return {"summary": asyncio.run(client.summarize(fragments)), "summaryError": None}
    except Exception as error:
        from ..error import unexpected_error
        print(f"[arrangement] 剧情摘要不可用，直接分析片段：{error}")
        return {"summary": "", "summaryError": unexpected_error(error).to_payload()}


def restructure_arrangement(work: Path, payload: dict, optimizer: SubtitleOptimizer | None = None) -> dict:
    """Plan one batch (``indices``) of the submitted draft, then merge it into the latest saved draft."""
    instruction, summary, indices = payload.get("instruction"), payload.get("summary") or "", payload.get("indices")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ArrangementError("请填写全局字幕优化要求")
    if not isinstance(summary, str) or not isinstance(indices, list) or not indices:
        raise ArrangementError("全局字幕优化批次无效")
    snapshot = validate_state(work, payload["state"])
    wanted = set(indices)
    fragments = [row for row in _active_fragments(work, snapshot) if row.index in wanted]
    if len(fragments) != len(wanted):
        raise ArrangementError("优化批次包含不存在或已删除的片段")
    audit = work / f"arrangement/audit/{uuid4().hex}-restructure"
    save_json(str(audit.with_suffix(".before.json")),
              {"instruction": instruction, "summary": summary, "indices": indices, "state": snapshot})
    try:
        plan = asyncio.run((optimizer or create_optimizer(payload["config"])).restructure(fragments, summary, instruction))
    except Exception as error:
        from ..error import unexpected_error
        save_json(str(audit.with_suffix(".error.json")), unexpected_error(error).to_payload())
        raise
    with draft_lock(work) as repository:
        latest = load_state(work, repository)
        if latest["sourceVersion"] != snapshot["sourceVersion"]:
            raise ArrangementError("来源字幕已变化，请重新载入")
        rows, _ = read_baseline(work)
        blocks = {edit["index"]: block_id(edit, row)
                  for edit, (_, row) in zip(latest["segments"], aligned_rows(rows, latest["segments"]))}
        state, report, changes = apply_restructure_plan(latest, plan, snapshot, blocks)
        save_state(work, state, repository)
    save_json(str(audit.with_suffix(".after.json")), {"plan": asdict(plan), "report": report, "changes": changes})
    print(f"[arrangement] 全局字幕优化批次 {len(fragments)} 段：修改 {report['modified']}，删除 {report['deleted']}，"
          f"合并 {report['merged']} 组，跳过 {len(report['skipped'])}")
    return {"changes": changes, "report": report}
