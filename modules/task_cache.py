"""Atomic, content-addressed stage results and failure audit records.

Only successful records are reusable. Failed units are retried on the next run.
Credentials never appear in serialized configuration or audit messages.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

from .error import safe_detail


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


class SqliteResultCache:
    """Content-addressed records, committed atomically in the project database."""

    def __init__(self, work_dir: str, namespace: str = "default"):
        from .storage import address, initialize_project
        initialize_project(work_dir)
        target = address(Path(work_dir) / "cache")
        self.store, self.scope = target.store, target.scope
        prefix = target.key.rsplit("/", 1)[0] + "/" if "/" in target.key else ""
        self.namespace = prefix + namespace

    def load(self, key: str) -> Any:
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT payload FROM cache_entries WHERE scope=? AND namespace=? AND cache_key=? AND status='succeeded'",
                (self.scope, self.namespace, key)).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, key: str, status: str, value: Any) -> None:
        from .storage import encode_json
        import time
        with self.store.connect() as connection:
            connection.execute("""INSERT INTO cache_entries(scope,namespace,cache_key,status,payload,updated_at)
                VALUES(?,?,?,?,?,?) ON CONFLICT(scope,namespace,cache_key) DO UPDATE SET
                status=excluded.status,payload=excluded.payload,updated_at=excluded.updated_at""",
                (self.scope, self.namespace, key, status, encode_json(value).decode('utf-8'), time.time_ns() // 1_000_000))

    def save(self, key: str, result: Any) -> None:
        self._write(key, "succeeded", result)

    def fail(self, key: str, error: Exception) -> None:
        descriptor = getattr(error, "descriptor", None)
        self._write(key, "failed", {"code": getattr(descriptor, "code", ""), "detail": safe_detail(error),
                                  "context": public_settings(getattr(error, "context", {}))})
