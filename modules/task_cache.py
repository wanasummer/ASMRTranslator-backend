"""Atomic, content-addressed stage results and failure audit records.

Only successful records are reusable. Failed units are retried on the next run.
Credentials never appear in serialized configuration or audit messages.
"""

import hashlib
import json
import os
from typing import Any, Protocol
from uuid import uuid4

from .error import safe_detail
from .io import write_json


def public_settings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: public_settings(item) for key, item in value.items()
                if key not in ("transport", "_transport", "client", "adapter", "proxy_provider",
                               "timeout", "max_retries", "batch_size", "min_interval_sec")
                if not any(word in key.lower() for word in
                           ("key", "token", "secret", "password", "authorization"))}
    if isinstance(value, (list, tuple)):
        return [public_settings(item) for item in value]
    if isinstance(value, str) and os.path.isfile(value):
        return {"path": os.path.abspath(value), "sha256": file_digest(value)}
    return value


def cache_key(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResultCache(Protocol):
    def load(self, key: str) -> Any: ...
    def save(self, key: str, result: Any) -> None: ...
    def fail(self, key: str, error: Exception) -> None: ...


class JsonResultCache:
    """One atomic UTF-8 record per unit; safe across independent worker writes."""

    def __init__(self, directory: str):
        self.directory = directory

    def load(self, key: str) -> Any:
        try:
            with open(os.path.join(self.directory, key + ".json"), encoding="utf-8") as source:
                record = json.load(source)
            if record.get("version") == 1 and record.get("status") == "succeeded":
                return record["result"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        return None

    def _write(self, key: str, record: dict) -> None:
        path = os.path.join(self.directory, key + ".json")
        temporary = path + "." + uuid4().hex + ".tmp"
        try:
            write_json(temporary, {"version": 1, **record})
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def save(self, key: str, result: Any) -> None:
        self._write(key, {"status": "succeeded", "result": result})

    def fail(self, key: str, error: Exception) -> None:
        descriptor = getattr(error, "descriptor", None)
        self._write(key, {"status": "failed", "code": getattr(descriptor, "code", ""),
                          "detail": safe_detail(error),
                          "context": public_settings(getattr(error, "context", {}))})
