"""ASR slice shaping for FSMN-VAD output — pure functions, no ML dependency.

FSMN-VAD already performs native sentence segmentation, so the raw regions
returned by :class:`~modules.vad.engine.FSMNVAD` are sentence-level. These
functions turn those raw regions into ASR transcription slices:

* ``optimize_asmr_segments`` groups short fragments into the target-duration
  golden band (absorbing nearby short lines, keeping isolated sub-second ASMR
  lines), and
* ``split_long_segment`` force-splits any span that still exceeds the max
  length into overlapping windows.

Both are pure and trivially testable offline.
"""

from .models import SpeechRegion


def split_long_segment(
    region: SpeechRegion,
    window_sec: float = 10.0,
    overlap_sec: float = 1.0,
    max_sec: float = 15.0,
) -> list[SpeechRegion]:
    """Split ``region`` into <= ``max_sec`` windows when it is too long.

    Windows advance by ``window_sec - overlap_sec`` so neighbours overlap by
    ``overlap_sec``; the downstream ``dedupe_overlaps`` deduplicates that overlap.
    A region already within ``max_sec`` is returned unchanged.
    """
    dur = region.duration
    if dur <= max_sec:
        return [region]
    step = max(1.0, window_sec - overlap_sec)
    out: list[SpeechRegion] = []
    t = 0.0
    while t < dur:
        e = min(t + window_sec, dur)
        out.append(SpeechRegion(region.start + t, region.start + e))
        if e >= dur:
            break
        t += step
    return out


def optimize_asmr_segments(regions, **kwargs) -> list[SpeechRegion]:
    """Compatibility entry point returning only the optimized extraction ranges."""
    return optimize_asmr_segments_with_lineage(regions, **kwargs)[0]


def optimize_asmr_segments_with_lineage(
    regions,
    *,
    min_sec: float = 0.3,
    target_min_sec: float = 5.0,
    target_max_sec: float = 25.0,
    max_sec: float = 25.0,
    max_merge_gap_sec: float = 2.0,
    split_window_sec: float = 20.0,
    split_overlap_sec: float = 1.0,
    padding_sec: float = 0.4,
    audio_duration_sec: float | None = None,
) -> tuple[list[SpeechRegion], tuple[tuple[SpeechRegion, tuple[int, ...], bool], ...]]:
    """Smooth raw FSMN-VAD regions into ASR slices.

    FSMN-VAD splits at > ``max_end_silence_time`` pauses, so raw regions are
    already sentence-level; this pass groups nearby regions into ASR-sized
    slices approaching ``target_max_sec``. Absorb the next region when its
    gap <= ``max_merge_gap_sec`` and either the accumulated slice is shorter
    than ``target_min_sec`` or the merged span stays <= ``target_max_sec``.
    Short fragment rescue may exceed the target; ``max_sec`` still controls
    splitting after merging.
    Fragments still under ``min_sec`` after merging are dropped (the default
    0.3s keeps genuine sub-second ASMR lines). Any span longer than ``max_sec``
    is split via :func:`split_long_segment`. Each resulting slice is then
    widened by ``padding_sec`` on both sides (clamped to the source audio
    boundaries when ``audio_duration_sec`` is provided) so the ASR transcribes
    a small buffer around the speech and boundary words are not clipped.
    """
    ordered = sorted(regions, key=lambda r: r.start)
    merged: list[SpeechRegion] = []
    owners: list[tuple[int, ...]] = []
    for ordinal, r in enumerate(ordered):
        if not merged:
            merged.append(r)
            owners.append((ordinal,))
            continue
        prev = merged[-1]
        if (
            r.start - prev.end <= max_merge_gap_sec
            and (
                prev.duration < target_min_sec
                or (r.end - prev.start) <= target_max_sec
            )
        ):
            merged[-1] = SpeechRegion(prev.start, max(prev.end, r.end))
            owners[-1] += (ordinal,)
        else:
            merged.append(r)
            owners.append((ordinal,))

    lineage = []
    out: list[SpeechRegion] = []
    for r, ids in zip(merged, owners):
        if r.duration < min_sec:
            continue
        windows = split_long_segment(r, split_window_sec, split_overlap_sec, max_sec)
        out.extend(windows)
        lineage.extend((window, ids, len(windows) > 1) for window in windows)
    if padding_sec > 0:
        audio_end = audio_duration_sec if audio_duration_sec is not None else float("inf")
        out = [
            SpeechRegion(
                max(0.0, r.start - padding_sec),
                min(r.end + padding_sec, audio_end),
                speech_start_ms=round(r.start * 1000),
                speech_end_ms=round(r.end * 1000),
            )
            for r in out
        ]
    return out, tuple(lineage)
