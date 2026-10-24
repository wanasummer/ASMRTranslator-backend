"""Persistence only: versioned recovery inputs and alignment audit."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from ..io import write_json
from .contracts import VadAnalysisError


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def write_artifact(output_dir, name, payload):
    try:
        return write_json(str(Path(output_dir) / "vad_output" / name),
                          {"schemaVersion": 1, **_json_value(payload)})
    except (OSError, TypeError, ValueError) as exc:
        raise VadAnalysisError("ARTIFACT_WRITE_FAILED", f"Cannot write {name}") from exc


def write_initial_regions(output_dir, regions):
    return write_artifact(output_dir, "initial_raw_regions.json", {"regions": regions})


def write_analysis(output_dir, analysis):
    return write_artifact(output_dir, "asr_slices.json", {
        "slices": analysis.slices, "regions": analysis.regions,
        "note": "Forced windows use owned fragments with parent_region_id; original regions are retained separately."})
