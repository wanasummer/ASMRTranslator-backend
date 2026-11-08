"""Fresh TTS assets for edited fragments; baseline caches are never overwritten."""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from .contracts import ArrangementError, ArrangementState
from .state import aligned_rows, draft_lock, load_state, read_baseline, save_state
from ..tts_engine import get_tts_adapter, prepare_text_for_synthesis


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


def regenerate_fragment(work: Path, state: ArrangementState, position: int, config: dict, kwargs: dict) -> ArrangementState:
    """Synthesize the submitted fragment, then attach it only if the latest draft still has that text."""
    if not isinstance(position, int) or not 0 <= position < len(state["segments"]):
        raise ArrangementError("片段位置无效")
    rows, _ = read_baseline(work)
    edit = state["segments"][position]
    _, row = aligned_rows(rows, state["segments"])[position]
    replacement = asyncio.run(synthesize_edit(work, {**row, **edit}, config, kwargs))
    with draft_lock(work) as repository:
        latest = load_state(work, repository)
        target = latest["segments"][position] if position < len(latest["segments"]) else None
        if target is not None and target["index"] == edit["index"] and target["text"] == edit["text"]:
            latest["segments"][position]["replacement"] = replacement
            latest["segments"][position]["ttsSample"] = None
        return save_state(work, latest, repository)
