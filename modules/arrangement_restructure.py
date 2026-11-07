"""Pure translation between arrangement drafts and subtitle-block restructure plans."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .arrangement_contracts import ArrangementState
from .llm.contracts import RestructurePlan


@dataclass(frozen=True)
class FragmentRow:
    index: int
    text: str
    source_text: str
    speaker: str | None
    start_ms: int
    end_ms: int


def restructure_rows(state: ArrangementState, aligned: list[tuple[int | None, dict]]) -> list[FragmentRow]:
    """Active fragments in draft order; ``aligned`` is ``aligned_rows`` for the same draft."""
    return [FragmentRow(edit["index"], edit["text"], edit["japanese"], row.get("speaker"),
                        round(row["start_ms"]), round(row["end_ms"]))
            for edit, (_, row) in zip(state["segments"], aligned) if not edit.get("deleted", False)]


_TRACKED = ("text", "japanese", "deleted")


def apply_restructure_plan(state: ArrangementState, plan: RestructurePlan,
                           expected: ArrangementState | None = None) -> tuple[ArrangementState, dict, list[dict]]:
    """Return ``(new draft, report, changed fragments)``; audio intervals and samples stay untouched.

    A merge keeps the first fragment (new text, joined Japanese) and deletes the rest.
    Operations touching fragments that changed since ``expected`` are skipped, never partially applied.
    Each change carries ``index`` plus the tracked fields after applying.
    """
    result: ArrangementState = json.loads(json.dumps(state))
    edits = result["segments"]
    position = {edit["index"]: i for i, edit in enumerate(edits)}
    before = {edit["index"]: edit for edit in (expected or state)["segments"]}
    active = [edit["index"] for edit in edits if not edit.get("deleted", False)]
    active_position = {index: i for i, index in enumerate(active)}
    report = {"modified": 0, "deleted": 0, "merged": 0, "mergedFragments": 0, "skipped": list(plan.rejected)}
    touched: list[int] = []
    for operation in plan.operations:
        indices = operation.indices
        if any(index not in active_position or index not in before
               or any(edits[position[index]].get(key, False) != before[index].get(key, False) for key in _TRACKED)
               for index in indices):
            report["skipped"].append(f"片段 {list(indices)} 已变化或已删除，跳过 {operation.kind}")
            continue
        if operation.kind == "merge" and [active_position[i] for i in indices] != list(
                range(active_position[indices[0]], active_position[indices[0]] + len(indices))):
            report["skipped"].append(f"片段 {list(indices)} 不再相邻，跳过合并")
            continue
        targets = [edits[position[index]] for index in indices]
        touched.extend(indices)
        if operation.kind == "modify":
            targets[0]["text"] = operation.text
            report["modified"] += 1
        elif operation.kind == "delete":
            targets[0]["deleted"] = True
            report["deleted"] += 1
        else:
            targets[0]["text"] = operation.text
            targets[0]["japanese"] = "".join(edit["japanese"].strip() for edit in targets)
            for edit in targets[1:]:
                edit["deleted"] = True
            report["merged"] += 1
            report["mergedFragments"] += len(targets)
    changes = [{"index": index, **{key: edits[position[index]].get(key, False) for key in _TRACKED}}
               for index in sorted(touched, key=position.__getitem__)]
    return result, report, changes
