"""Persist LLM inputs and raw responses separately from transformations."""

from dataclasses import asdict
from pathlib import Path
from ..storage import save_json
from ..task_cache import public_settings


class RequestArtifactWriter:
    def __init__(self, directory: str | None, *, secret: str = ""):
        self.directory = Path(directory) / "llm_results" / "requests" if directory else None
        self.secret = secret

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, "[REDACTED]") if self.secret else value
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def _write(self, key: str, suffix: str, value: dict) -> None:
        if self.directory is not None:
            save_json(str(self.directory / f"{key}.{suffix}.json"), self._redact(value))

    def request(self, key: str, config, system: str, user: str, max_tokens: int | None) -> None:
        self._write(key, "input", {"schemaVersion": 1, "settings": public_settings(asdict(config)),
                                  "system": system, "user": user, "max_tokens": max_tokens})

    def response(self, key: str, raw) -> None:
        self._write(key, "response", {"schemaVersion": 1, "raw": raw})
