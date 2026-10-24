"""Pink-segment timeline model — continuous seamless slices over the whole audio.

The original audio (the "black" timeline, 0 → total duration) is divided into
contiguous, seamless pink segments driven by the speech subtitles (the "orange"
timeline). Each pink segment holds exactly one subtitle block plus the SE that
surrounds it:

    Time:
    Black (audio):     |=================================================|  (0 .. total)
    Orange (speech):        [----O1----]           [----O2----]
    Pink (segments):   |-------P1-------|--------------P2----------------|

The translated TTS is triggered at the block's orange start within its pink
segment, so it must fit within ``max_allowed_tts = P_end - O_start`` (falling
back to the pipeline's speed-up / padding strategies when it does not).

This is the single source of truth for the audio slicer in every path — with a
subtitle file the times come straight from the VTT; on the ASR path they replace
the old ``asr_engine.align_asr_times`` re-spanning. All fields are in
milliseconds to match the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PinkSegment:
    """One continuous slice of the original audio around a subtitle block.

    Field mapping to the spec's seconds-based naming:
        pink_start_ms            → P_start
        pink_end_ms              → P_end
        orange_start_ms          → O_start
        orange_end_ms            → O_end
        tts_delay_ms             → O_start - P_start  (TTS trigger offset)
        max_allowed_tts_ms       → P_end - O_start    (TTS time budget)
    """

    segment_id: int
    pink_start_ms: int
    pink_end_ms: int
    orange_start_ms: int
    orange_end_ms: int
    tts_delay_ms: int
    max_allowed_tts_ms: int
    subtitle_data: Any


def _clamp_cut(a_end_ms: int, b_start_ms: int) -> int:
    """Cut point between two consecutive blocks: midpoint, clamped to keep each
    segment covering its own speech (degenerate to ``a_end`` on overlap)."""
    mid = (a_end_ms + b_start_ms) // 2
    return max(a_end_ms, min(mid, b_start_ms))


def calculate_pink_segments(subtitles, total_duration_ms: int) -> list[PinkSegment]:
    """Split the whole audio into continuous pink segments, one per subtitle.

    ``subtitles`` is an iterable of objects exposing ``start_ms``/``end_ms``
    (VTTBlock satisfies this). ``total_duration_ms`` is the black-timeline audio
    length. Returns an empty list for empty input. The first segment starts at 0,
    the last ends at ``total_duration_ms``, and consecutive segments share a cut
    point so the timeline is covered seamlessly.
    """
    ordered = sorted(subtitles, key=lambda s: s.start_ms)
    if not ordered:
        return []
    total = max(0, int(total_duration_ms))

    cuts = [
        _clamp_cut(ordered[i].end_ms, ordered[i + 1].start_ms)
        for i in range(len(ordered) - 1)
    ]

    out: list[PinkSegment] = []
    for i, sub in enumerate(ordered):
        pink_start = 0 if i == 0 else cuts[i - 1]
        pink_end = total if i == len(ordered) - 1 else cuts[i]
        out.append(PinkSegment(
            segment_id=i,
            pink_start_ms=pink_start,
            pink_end_ms=pink_end,
            orange_start_ms=sub.start_ms,
            orange_end_ms=sub.end_ms,
            tts_delay_ms=max(0, sub.start_ms - pink_start),
            max_allowed_tts_ms=max(0, pink_end - sub.start_ms),
            subtitle_data=sub,
        ))
    return out


def attach_pink_geometry(blocks, total_duration_ms: int) -> list[PinkSegment]:
    """Compute the pink timeline for ``blocks`` and write the geometry onto each.

    Each block gains ``pink_start_ms``/``pink_end_ms``/``tts_delay_ms``/
    ``max_allowed_tts_ms`` so the audio slicer can cut the pink slices after the
    blocks have been split across parallel chunks. Returns the pink segments.
    """
    segments = calculate_pink_segments(blocks, total_duration_ms)
    for p in segments:
        b = p.subtitle_data
        b.pink_start_ms = p.pink_start_ms
        b.pink_end_ms = p.pink_end_ms
        b.tts_delay_ms = p.tts_delay_ms
        b.max_allowed_tts_ms = p.max_allowed_tts_ms
    return segments
