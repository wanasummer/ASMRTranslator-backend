"""Adopt clips synthesized by a successful render back into the editable draft.

The studio may still hold, and save, a draft from before the render. Each adoption is recorded with the
audio fields it replaced; a save that still carries exactly those fields is stale and keeps the adopted
take. The record is cleared when the studio loads the draft and sees the adopted takes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .contracts import ArrangementState, RenderedTake
from .state import draft_lock, load_state, save_state
from ..storage import contains, delete_document, read_json, save_json

ADOPTED_TAKES = "arrangement/adopted_takes.json"
_AUDIO_FIELDS = ("replacement", "interval", "ttsSample")


def adopt_rendered_takes(work: Path, takes: Sequence[RenderedTake]) -> int:
    """Attach each take to its fragment while the latest draft still has that text and no matching audio."""
    if not takes:
        return 0
    ledger = {}
    with draft_lock(work) as repository:
        latest = load_state(work, repository)
        by_index = {edit["index"]: edit for edit in latest["segments"]}
        for take in takes:
            edit = by_index.get(take.index)
            replacement = edit and edit.get("replacement")
            if (edit is None or edit.get("deleted", False) or edit["text"] != take.text
                    or (replacement and replacement["text"] == edit["text"])):
                continue
            before = {field: edit.get(field) for field in _AUDIO_FIELDS}
            edit.update(replacement={"path": take.path, "text": take.text},
                        interval={"start": take.start_ms, "end": take.end_ms},
                        ttsSample={"start": 0, "end": take.sample_end_ms})
            ledger[str(take.index)] = {"text": take.text, "before": before,
                                       "after": {field: edit[field] for field in _AUDIO_FIELDS}}
        if ledger:
            save_state(work, latest, repository)
    if ledger:
        save_json(str(work / ADOPTED_TAKES), {**_read_ledger(work), **ledger})
    return len(ledger)


def _read_ledger(work: Path) -> dict:
    path = work / ADOPTED_TAKES
    return read_json(path) if contains(path) else {}


def keep_adopted_takes(work: Path, state: ArrangementState) -> ArrangementState:
    """Return ``state`` with adopted takes restored on fragments the caller has not edited since adoption."""
    ledger = _read_ledger(work)
    if not ledger:
        return state
    result: ArrangementState = json.loads(json.dumps(state))
    for edit in result["segments"]:
        entry = ledger.get(str(edit["index"]))
        if (entry and not edit.get("deleted", False) and edit["text"] == entry["text"]
                and {field: edit.get(field) for field in _AUDIO_FIELDS} == entry["before"]):
            edit.update(entry["after"])
    return result


def forget_adopted_takes(work: Path) -> None:
    path = work / ADOPTED_TAKES
    if contains(path):
        delete_document(path)
