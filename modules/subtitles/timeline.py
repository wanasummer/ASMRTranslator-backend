"""Map subtitle blocks from the source timeline onto the joined-output timeline.

Why this module exists
----------------------
``modules/audio_joiner.join_audio`` concatenates one audio file per segment. In
``dual`` (retain original) mode every segment already contains
``original + gap + TTS``, so the joined file is longer than the source audio, and
with ``crossfade_ms > 0`` every join is also shortened by the fade. The subtitle
export (``llm_subtitles/final.vtt``) is built from blocks that still carry the
*source* timeline, so its cue times drift away from the exported audio — by up to
minutes on long projects.

``align_blocks_to_output`` closes that gap without decoding or correlating audio:
the joiner already knows exactly where each segment landed, so the mapping is
exact::

    output_ms = placement_start_ms + (source_ms - segment_start_ms)

Contract
--------
``placement`` is the list returned through ``join_audio(..., placement=...)``:
one ``{"index", "start_ms", "end_ms"}`` dict per **surviving** segment, in output
order. Segments that were dropped by the joiner (decoded audio shorter than the
minimum) have no audio in the export, so their blocks are dropped too — keeping
them would shift every later cue.
"""

from dataclasses import replace
from typing import Iterable, Sequence

from ..vtt_parser import VTTBlock

Placement = dict[str, int]


def _index_of(item: dict) -> int | None:
    """Return the segment index of a placement/segment dict, if it has one."""
    index = item.get("index")
    return int(index) if index is not None else None


def align_blocks_to_output(
    blocks: Iterable[VTTBlock],
    segments: Sequence[dict],
    placement: Sequence[Placement],
) -> tuple[list[VTTBlock], list[int]]:
    """Return ``(aligned_blocks, dropped_indexes)`` without mutating the inputs.

    ``aligned_blocks`` keeps the input order and every other block field
    (``text``/``speaker``/``skip_tts``/...); only ``start_ms``/``end_ms`` move.
    Blocks whose segment never made it into the joined audio are returned in
    ``dropped_indexes`` instead of the block list.
    """
    placed: dict[int, int] = {}
    for item in placement:
        index = _index_of(item)
        if index is not None:
            placed.setdefault(index, int(item.get("start_ms", 0)))

    segment_start: dict[int, int] = {}
    for segment in segments:
        index = _index_of(segment)
        if index is not None and "start_ms" in segment:
            segment_start.setdefault(index, int(segment["start_ms"]))

    aligned: list[VTTBlock] = []
    dropped: list[int] = []
    cursor = 0
    for block in blocks:
        index = _index_of({"index": getattr(block, "index", None)})
        if index is None or index not in placed:
            dropped.append(index if index is not None else -1)
            continue
        delta = placed[index] - segment_start.get(index, int(block.start_ms))
        start_ms = max(0, int(block.start_ms) + delta)
        end_ms = max(start_ms, int(block.end_ms) + delta)
        # Keep the export monotonic even if a segment's placement is unusual.
        if start_ms < cursor:
            shift = cursor - start_ms
            start_ms += shift
            end_ms += shift
        cursor = end_ms
        aligned.append(replace(block, start_ms=start_ms, end_ms=end_ms))
    return aligned, dropped
