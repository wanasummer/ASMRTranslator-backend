"""Persistence for detect-stage recovery and audit artifacts."""

from dataclasses import asdict, is_dataclass
from pathlib import Path

from ..io import write_json
from .contracts import DetectError


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def write_detect_artifact(output_dir, name, payload):
    try:
        return write_json(str(Path(output_dir) / "detect_output" / name),
                          {"schemaVersion": 1, **_json_value(payload)})
    except (OSError, TypeError, ValueError) as exc:
        raise DetectError("DETECT_ARTIFACT_WRITE_FAILED", f"Cannot write {name}") from exc


def write_alignment_audit(output_dir, config, summary, records):
    try:
        write_detect_artifact(output_dir, "sentence_alignment_performance.json", {
            "slices": [{"sliceId": record[0], "attempts": record[3].timings}
                       for record in records]})
    except DetectError:
        print("       [warn] DETECT_ARTIFACT_WRITE_FAILED: optional timing report unavailable")
    return write_detect_artifact(output_dir, "sentence_alignment.json", {
        "enabled": True, "config": config, "summary": summary,
        "slices": [{"sliceId": slice_id, "detection": detection,
                    "request": request, "result": result}
                   for slice_id, detection, request, result in records]})
