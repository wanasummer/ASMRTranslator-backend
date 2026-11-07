"""Restructure response parsing without I/O."""
import json
import re
from typing import Sequence
from ....text_normalization import _sanitize_symbols, _normalize_tags
from ...contracts import RestructureOperation, RestructurePlan


def _extract_object(raw: str):
    text = re.sub(r"^\s*```(?:json)?\s*\n?|\n?```\s*$", "", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("响应中没有 JSON 对象")
    return json.loads(text[start:end + 1])


def _clean_text(value) -> str:
    return _sanitize_symbols(_normalize_tags(value.strip())) if isinstance(value, str) else ""


def _is_index(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_restructure_plan(raw: str, order: Sequence[int]) -> RestructurePlan:
    """Raise ``ValueError`` for an unusable response; drop and report individually invalid operations.

    ``order`` lists the batch's row identifiers in draft order; merges must cover a contiguous run of it.
    """
    value = _extract_object(raw)
    items = value.get("operations") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise ValueError("响应缺少 operations 列表")
    positions = {index: position for position, index in enumerate(order)}
    claimed: set[int] = set()
    operations, rejected = [], []
    for item in items:
        if not isinstance(item, dict):
            rejected.append(f"无效操作：{item!r}"[:200])
            continue
        kind = item.get("type")
        indices = item.get("indices") if kind == "merge" else [item.get("index")]
        reason = item.get("reason") if isinstance(item.get("reason"), str) else ""
        text = _clean_text(item.get("text"))
        problem = ""
        if kind not in ("modify", "delete", "merge"):
            problem = "未知操作类型"
        elif not isinstance(indices, list) or not all(_is_index(i) and i in positions for i in indices):
            problem = "片段序号不在本批中"
        elif claimed.intersection(indices) or len(set(indices)) != len(indices):
            problem = "片段被多个操作引用"
        elif kind == "merge" and (len(indices) < 2 or [positions[i] for i in indices]
                                  != list(range(positions[indices[0]], positions[indices[0]] + len(indices)))):
            problem = "合并的片段不相邻"
        elif kind != "delete" and not text:
            problem = "缺少中文字幕"
        if problem:
            rejected.append(f"{problem}：{json.dumps(item, ensure_ascii=False)}"[:300])
            continue
        claimed.update(indices)
        operations.append(RestructureOperation(kind, tuple(indices), "" if kind == "delete" else text, reason.strip()))
    return RestructurePlan(tuple(operations), tuple(rejected))
