"""VAD analysis stage: detect speech, optimize ASR slices, preserve lineage."""

import asyncio
from dataclasses import replace

from ..audio_splitter import get_audio_duration
from .analysis import analyze_regions, identify_regions
from .artifacts import write_analysis, write_initial_regions
from .contracts import VadAnalysisError
from .engine import create_vad


def run_analysis_stage(audio_path, vad_config, output_dir):
    engine = create_vad({**vad_config, "strict": True})
    if not engine.available:
        raise VadAnalysisError("VAD_ANALYSIS_FAILED", "Initial VAD unavailable")
    raw = asyncio.run(engine.detect(audio_path))
    write_initial_regions(
        output_dir, identify_regions(raw, vad_config.get("max_end_silence_time", 1500)))
    extractions, analysis = analyze_regions(
        raw,
        {**vad_config, "audio_duration_sec": get_audio_duration(audio_path)},
    )
    write_analysis(output_dir, analysis)
    return [replace(region, asr_slice=slice_contract)
            for region, slice_contract in zip(extractions, analysis.slices)], analysis
