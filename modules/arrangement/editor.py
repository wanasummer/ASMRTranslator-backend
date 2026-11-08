"""Sidecar arrangement editor protocol: dispatch one stdin request to the owning submodule."""
from __future__ import annotations

from pathlib import Path

from .contracts import ArrangementError
from .optimization import (commit_optimized_fragment, optimize_draft, optimize_submitted_fragment,
                           restructure_arrangement, summarize_arrangement)
from .state import draft_lock, load_state, save_state, subtitle_base, validate_state
from .synthesis import regenerate_fragment
from .takes import forget_adopted_takes, keep_adopted_takes


def run_editor_request(payload: dict) -> dict:
    """Sidecar editor entry: network config only arrives through stdin."""
    work = Path(payload["workDir"]).resolve()
    action = payload["action"]
    if action == "load":
        state = load_state(work)
        forget_adopted_takes(work)
        return {"state": state}
    if action == "commit-optimization":
        return commit_optimized_fragment(work, payload)
    if action == "subtitle-base":
        return {"texts": {str(index): text for index, text in subtitle_base(work).items()}}
    if action == "summarize-arrangement":
        return summarize_arrangement(work, payload)
    if action == "optimize-arrangement":
        return restructure_arrangement(work, payload)
    if action == "optimize-fragment":
        return optimize_submitted_fragment(work, payload)
    state = keep_adopted_takes(work, validate_state(work, payload["state"]))
    if action == "save":
        with draft_lock(work) as repository:
            return {"state": save_state(work, state, repository)}
    if action in ("optimize", "optimize-all"):
        return {"state": optimize_draft(work, state, payload["config"],
                                        payload.get("position") if action == "optimize" else None)}
    if action == "regenerate":
        return {"state": regenerate_fragment(work, state, payload["position"],
                                             payload["config"], payload["ttsKwargs"])}
    raise ArrangementError("未知编排操作")
