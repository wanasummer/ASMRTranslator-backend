"""Project-local persistence. Audio is filesystem data; documents are SQLite data.

Addresses are explicit logical names under a generated directory, never files to
be materialized. Both desktop and CLI use the same schema and UTF-8 contracts.
Every operation owns a short connection/transaction; no connection crosses threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Protocol
from urllib.parse import quote, unquote

GENERATED_DIRECTORY = "generated"
DATABASE_NAME = "project.sqlite3"
SCHEMA_VERSION = 1


class StorageError(RuntimeError):
    """A project database cannot be read or committed."""


class DocumentStore(Protocol):
    def get(self, scope: str, key: str) -> bytes | None: ...
    def put(self, scope: str, key: str, payload: bytes, media_type: str) -> None: ...
    def delete(self, scope: str, key: str) -> None: ...


def encode_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class SqliteProjectStore:
    def __init__(self, root: Path):
        if root.is_symlink() or (root / DATABASE_NAME).is_symlink():
            raise StorageError("Project database cannot use filesystem links")
        self.root = root.resolve()
        self.path = self.root / DATABASE_NAME

    @contextmanager
    def connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=15)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='schema_info'").fetchone():
                connection.executescript(Path(__file__).with_name("storage_schema.sql").read_text(encoding="utf-8"))
            version = connection.execute("SELECT version FROM schema_info").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise StorageError(f"Unsupported project database version: {version}")
            with connection:
                yield connection
        except sqlite3.Error as error:
            raise StorageError(f"Project database operation failed: {self.path}: {error}") from error
        finally:
            if connection is not None:
                connection.close()

    def get(self, scope: str, key: str) -> bytes | None:
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM documents WHERE scope=? AND key=?", (scope, key)).fetchone()
        return bytes(row[0]) if row else None

    def put(self, scope: str, key: str, payload: bytes, media_type: str) -> None:
        with self.connect() as connection:
            connection.execute("""INSERT INTO documents(scope,key,media_type,payload,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(scope,key) DO UPDATE SET media_type=excluded.media_type,payload=excluded.payload,
                revision=documents.revision+1,updated_at=excluded.updated_at""",
                (scope, key, media_type, payload, time.time_ns() // 1_000_000))

    def delete(self, scope: str, key: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM documents WHERE scope=? AND key=?", (scope, key))

    def append_log(self, scope: str, stream: str, content: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO log_chunks(scope,stream,content,created_at) VALUES(?,?,?,?)",
                               (scope, stream, content, time.time_ns() // 1_000_000))

    def read_log(self, scope: str, stream: str) -> str:
        with self.connect() as connection:
            return "".join(row[0] for row in connection.execute(
                "SELECT content FROM log_chunks WHERE scope=? AND stream=? ORDER BY id", (scope, stream)))

    def keys(self, scope: str, prefix: str = "") -> list[str]:
        with self.connect() as connection:
            return [row[0] for row in connection.execute(
                "SELECT key FROM documents WHERE scope=? AND substr(key,1,?)=? ORDER BY key",
                (scope, len(prefix), prefix))]


@dataclass(frozen=True)
class DocumentAddress:
    store: SqliteProjectStore
    scope: str
    key: str

    @property
    def reference(self) -> str:
        return "sqlite:" + quote(str(self.store.path), safe="/") + "#" + quote(self.scope, safe="") + "/" + quote(self.key, safe="")


def address(name: str | Path) -> DocumentAddress:
    raw = str(name)
    if raw.startswith("sqlite:"):
        database, fragment = raw[7:].split("#", 1)
        scope, key = fragment.split("/", 1)
        return DocumentAddress(SqliteProjectStore(Path(unquote(database)).parent), unquote(scope), unquote(key))
    path = Path(name).resolve()
    for parent in path.parents:
        if parent.name == GENERATED_DIRECTORY or (parent / DATABASE_NAME).is_file():
            relative = path.relative_to(parent).parts
            # Root documents have a single component; asset data starts at assets/<id>.
            if len(relative) >= 3 and relative[0] == "assets":
                return DocumentAddress(SqliteProjectStore(parent), relative[1], Path(*relative[2:]).as_posix())
            return DocumentAddress(SqliteProjectStore(parent), "", Path(*relative).as_posix())
    # A standalone CLI output directory is itself a project storage root.
    return DocumentAddress(SqliteProjectStore(path.parent), "", path.name)


def initialize_project(work_dir: str | Path) -> SqliteProjectStore:
    """Open a standalone output root or resolve an existing desktop project root."""
    store = address(Path(work_dir) / "storage").store
    with store.connect():
        pass
    return store


def save_bytes(name: str | Path, value: bytes, media_type: str = "application/octet-stream") -> str:
    target = address(name)
    target.store.put(target.scope, target.key, value, media_type)
    return target.reference


def save_json(name: str | Path, value, indent: int = 2) -> str:
    return save_bytes(name, encode_json(value), "application/json")


def save_text(name: str | Path, value: str, encoding: str = "utf-8") -> str:
    return save_bytes(name, value.encode("utf-8"), "text/plain; charset=utf-8")


def read_bytes(name: str | Path) -> bytes:
    target = address(name)
    value = target.store.get(target.scope, target.key)
    if value is None:
        raise FileNotFoundError(f"Project document missing: {target.reference}")
    return value


def read_json(name: str | Path):
    return json.loads(read_bytes(name))


def read_text(name: str | Path) -> str:
    return read_bytes(name).decode("utf-8")


def contains(name: str | Path) -> bool:
    target = address(name)
    return target.store.get(target.scope, target.key) is not None


def document_digest(name: str | Path) -> str:
    return hashlib.sha256(read_bytes(name)).hexdigest()


def delete_document(name: str | Path) -> None:
    target = address(name)
    target.store.delete(target.scope, target.key)


def move_document(source: str | Path, destination: str | Path) -> None:
    origin, target = address(source), address(destination)
    if origin.store.path != target.store.path:
        raise StorageError("Cannot move a document across project databases")
    with origin.store.connect() as connection:
        row = connection.execute("SELECT payload,media_type FROM documents WHERE scope=? AND key=?", (origin.scope, origin.key)).fetchone()
        if row:
            connection.execute("INSERT OR REPLACE INTO documents(scope,key,payload,media_type,updated_at) VALUES(?,?,?,?,?)",
                               (target.scope, target.key, row[0], row[1], time.time_ns() // 1_000_000))
        else:
            connection.execute("DELETE FROM documents WHERE scope=? AND key=?", (target.scope, target.key))
        connection.execute("DELETE FROM documents WHERE scope=? AND key=?", (origin.scope, origin.key))


class DatabaseLog:
    def __init__(self, name: str | Path, encoding: str = "utf-8"):
        self.target = address(name)
        self.name = self.target.reference
        self.closed = False
        self._pending: list[str] = []

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("Log is closed")
        self._pending.append(text)
        if "\n" in text:
            self.flush()
        return len(text)

    def flush(self):
        if self._pending:
            self.target.store.append_log(self.target.scope, self.target.key, "".join(self._pending))
            self._pending.clear()

    def close(self):
        self.flush()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_log(reference: str) -> str:
    target = address(reference)
    return target.store.read_log(target.scope, target.key)
