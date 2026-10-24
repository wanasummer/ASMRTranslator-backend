"""Produce lineage during optimization, never infer it in later stages."""

from .contracts import AsrSlice, VadAnalysis, VadRegion, VadAnalysisError
from .segments import optimize_asmr_segments_with_lineage


def identify_regions(regions, threshold_ms=1500):
    """Also accepts the legacy JSON region rows without IDs."""
    rows = [(r['start'], r['end']) if isinstance(r, dict) else (r.start, r.end)
            for r in regions]
    return tuple(VadRegion(f"initial-{round(s * 1000)}-{round(e * 1000)}",
                           round(s * 1000), round(e * 1000), threshold_ms=threshold_ms)
                 for s, e in sorted(rows))


def analyze_regions(raw_regions, config):
    initial = identify_regions(raw_regions, config.get("max_end_silence_time", 1500))
    previous = -1
    for r in initial:
        if r.speech_start_ms < previous or r.speech_end_ms <= r.speech_start_ms:
            raise VadAnalysisError("VAD_LINEAGE_INVALID", "Invalid initial VAD timeline")
        previous = r.speech_end_ms
    keys = ("min_sec", "target_min_sec", "target_max_sec", "max_sec",
            "max_merge_gap_sec", "split_window_sec", "split_overlap_sec", "padding_sec",
            "audio_duration_sec")
    extractions, lineage = optimize_asmr_segments_with_lineage(
        raw_regions, **{k: config[k] for k in keys if k in config})
    slices, owned = [], []
    previous_end = 0
    for extraction, (window, ordinals, forced) in zip(extractions, lineage):
        # Same ownership rule as ASR's overlapping extraction windows. Padding
        # is decode-only; when it overlaps speech, ownership stays with speech.
        # Native VAD can report a final speech end beyond the decoded audio
        # duration. The extraction is already clamped to the real file, so clip
        # ownership to that extraction as well and keep the downstream lineage
        # contract internally consistent.
        extraction_start = round(extraction.start * 1000)
        extraction_end = round(extraction.end * 1000)
        start = max(round(window.start * 1000), previous_end, extraction_start)
        end = min(round(window.end * 1000), extraction_end)
        if end <= start:
            continue
        slice_id = f"slice-{start}-{end}"
        regions = []
        for ordinal in ordinals:
            original = initial[ordinal]
            left = max(start, original.speech_start_ms)
            right = min(end, original.speech_end_ms)
            if left >= right:
                continue
            clipped = (left, right) != (original.speech_start_ms, original.speech_end_ms)
            regions.append(VadRegion(
                f"{original.id}@{slice_id}" if clipped else original.id,
                left, right, threshold_ms=original.threshold_ms,
                parent_region_id=original.id if clipped else None))
        if not regions:
            raise VadAnalysisError("VAD_LINEAGE_INVALID", "Slice has no speech lineage")
        owned.extend(regions)
        slices.append(AsrSlice(slice_id, extraction_start,
                               extraction_end, regions[0].speech_start_ms,
                               regions[-1].speech_end_ms, tuple(r.id for r in regions), forced))
        previous_end = end
    return extractions, VadAnalysis(tuple(owned), tuple(slices), initial)
