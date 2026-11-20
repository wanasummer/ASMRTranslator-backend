"""Persist optional native document responses before subtitle cleanup."""
import os

from ..storage import save_json


def write_document_artifacts(result, output_dir: str) -> None:
    from ..storage import initialize_project
    initialize_project(output_dir)
    payload = getattr(result, "raw_response", None)
    if payload is not None:
        save_json(os.path.join(os.path.join(output_dir, "asr_subtitles"), "raw_response.json"), payload)
