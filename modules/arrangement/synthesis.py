"""Fresh TTS assets for edited fragments; baseline caches are never overwritten."""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from .contracts import ArrangementError, ArrangementState, TtsVoice
from .state import aligned_rows, draft_lock, fragment_tts, load_state, read_baseline, save_state
from ..tts_engine import get_tts_adapter, prepare_text_for_synthesis


def resolve_voice(edit: dict, config: dict, kwargs: dict, overrides: dict | None = None) -> TtsVoice:
    """A fragment with its own TTS profile uses that profile's resolved settings; others use the generation profile.

    ``config``/``kwargs`` already describe the generation profile; ``overrides`` maps fragment index to the
    resolved ``{ttsBackend, rate, edgePitch, kwargs}`` of each fragment profile.
    """
    if fragment_tts(edit) is None:
        return TtsVoice(config.get("ttsBackend", "edge-tts"), config.get("rate", "+0%"),
                        config.get("edgePitch", "+0Hz"), kwargs)
    override = (overrides or {}).get(str(edit["index"]))
    if not isinstance(override, dict) or not override.get("ttsBackend"):
        raise ArrangementError(f"片段 {edit['index']} 的单独配音配置未能解析")
    return TtsVoice(override["ttsBackend"], override.get("rate", "+0%"),
                    override.get("edgePitch", "+0Hz"), override.get("kwargs") or {})


async def synthesize_edit(work: Path, row: dict, voice: TtsVoice, adapter=None) -> dict:
    """Always create a fresh immutable TTS asset; never overwrite baseline cache."""
    if not row["text"].strip():
        raise ArrangementError("中文字幕为空，无法重新生成")
    adapter = adapter or get_tts_adapter(voice.backend, voice.kwargs)
    from ..tts.references import arrangement_segment
    segment = arrangement_segment(work, row)
    if callable(getattr(type(adapter), "bind_segment", None)):
        adapter = adapter.bind_segment(segment)
    if callable(getattr(type(adapter), "for_segment", None)):
        from ..tts.models import TTSSegmentContext
        adapter = adapter.for_segment(TTSSegmentContext.from_segment(segment))
    directory = work / "arrangement/audio"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}{adapter.output_extension}"
    try:
        await adapter.synthesize(prepare_text_for_synthesis(adapter, row["text"]),
                                 adapter.get_voice_for_speaker(row.get("speaker")), str(path),
                                 rate=voice.rate, pitch=voice.pitch)
    finally:
        await adapter.aclose()
    if not path.is_file() or path.stat().st_size == 0:
        raise ArrangementError("TTS 未返回有效音频")
    replacement = {"path": path.relative_to(work).as_posix(), "text": row["text"]}
    if fragment_tts(row) is not None:
        replacement["tts"] = row["tts"]
    return replacement


def regenerate_fragment(work: Path, state: ArrangementState, position: int, config: dict, kwargs: dict,
                        overrides: dict | None = None) -> ArrangementState:
    """Synthesize the submitted fragment. Its TTS profile is stored only after the audio is attached."""
    if not isinstance(position, int) or not 0 <= position < len(state["segments"]):
        raise ArrangementError("片段位置无效")
    rows, _ = read_baseline(work)
    edit = state["segments"][position]
    _, row = aligned_rows(rows, state["segments"])[position]
    replacement = asyncio.run(synthesize_edit(work, {**row, **edit}, resolve_voice(edit, config, kwargs, overrides)))
    with draft_lock(work) as repository:
        latest = load_state(work, repository)
        target = latest["segments"][position] if position < len(latest["segments"]) else None
        if target is not None and target["index"] == edit["index"] and target["text"] == edit["text"]:
            latest["segments"][position]["replacement"] = replacement
            latest["segments"][position]["ttsSample"] = None
            latest["segments"][position]["tts"] = fragment_tts(edit)
        return save_state(work, latest, repository)
