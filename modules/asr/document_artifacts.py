"""Persist optional native document responses before subtitle cleanup."""
import os

from ..io import intermediate_dir, write_json


def write_document_artifacts(result, output_dir: str) -> None:
    payload = getattr(result, "raw_response", None)
    if payload is not None:
        write_json(os.path.join(intermediate_dir(output_dir, "asr_subtitles"), "raw_response.json"), payload)
