"""Stable parent-ASR reference identity, independent of subtitle placement."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def supports_parent_reference(backend: str, settings: dict | None) -> bool:
    """Prepare optional assets only for manifests advertising this reference scope."""
    if backend != "common-tts":
        return False
    settings = settings or {}
    schema = settings.get("manifest", {}).get("parameters", {}).get("properties", {})
    return (settings.get("parameters", {}).get("reference_scope") == "asr_block"
            or "asr_block" in schema.get("reference_scope", {}).get("enum", ()))


def block_reference_path(directory: str | Path, identity: str, start_ms: int, end_ms: int) -> Path:
    key = sha256(f"{identity}:{start_ms}:{end_ms}".encode("utf-8")).hexdigest()[:24]
    return Path(directory) / f"asr_block_{key}.wav"


def parent_span(block) -> tuple[str, int, int]:
    """Prefer the detected block's ASR parent; an unsplit block is its own parent."""
    aligned = getattr(block, "aligned_sentence", None)
    start = getattr(aligned, "source_start_ms", None)
    end = getattr(aligned, "source_end_ms", None)
    identity = getattr(aligned, "source_block_id", None)
    if start is None or end is None:
        start, end = block.start_ms, block.end_ms
        identity = str(block.index)
    return str(identity), int(start), int(end)


def arrangement_segment(work: Path, row: dict) -> dict:
    """Rebind immutable generation assets for a regenerated studio fragment."""
    segment = dict(row)
    directory = work / "chunk_0/01_segments"
    index = row["index"]
    for key, prefix in (("speech_file", "speech"), ("segment_file", "seg")):
        path = directory / f"{prefix}_{index:04d}.wav"
        if path.is_file():
            segment[key] = str(path)
    if all(row.get(key) is not None for key in ("source_block_id", "source_start_ms", "source_end_ms")):
        path = block_reference_path(directory, row["source_block_id"], row["source_start_ms"], row["source_end_ms"])
        if path.is_file():
            segment["asr_block_file"] = str(path)
            segment["tts_context"] = {key: row[key] for key in ("source_block_id", "source_start_ms", "source_end_ms")}
    return segment
