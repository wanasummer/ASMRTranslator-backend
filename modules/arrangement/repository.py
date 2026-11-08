"""Transactional persistence for editable drafts and immutable draft history."""
from contextlib import contextmanager, AbstractContextManager
import json
from pathlib import Path
import time
from typing import Protocol
from uuid import uuid4

from .contracts import ArrangementState
from ..storage import address, encode_json


class ArrangementRepository(Protocol):
    def load(self) -> ArrangementState | None: ...
    def save(self, state: ArrangementState) -> None: ...
    def transaction(self) -> AbstractContextManager["ArrangementRepository"]: ...


class SqliteArrangementRepository:
    def __init__(self, work: Path, *, _connection=None):
        self.work = work
        self.target = address(work / "arrangement/draft.json")
        self._connection = _connection

    @contextmanager
    def transaction(self):
        with self.target.store.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            yield SqliteArrangementRepository(self.work, _connection=connection)

    def load(self) -> ArrangementState | None:
        if self._connection is None:
            with self.target.store.connect() as active:
                return SqliteArrangementRepository(self.work, _connection=active).load()
        row = self._connection.execute("SELECT payload FROM documents WHERE scope=? AND key=?",
                                 (self.target.scope, self.target.key)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, state: ArrangementState) -> None:
        if self._connection is None:
            with self.transaction() as active:
                active.save(state)
            return
        connection = self._connection
        previous = connection.execute("SELECT payload FROM documents WHERE scope=? AND key=?",
                                      (self.target.scope, self.target.key)).fetchone()
        now = time.time_ns() // 1_000_000
        if previous:
            connection.execute("INSERT INTO documents(scope,key,media_type,payload,updated_at) VALUES(?,?,?,?,?)",
                (self.target.scope, "arrangement/history/" + uuid4().hex, "application/json", previous[0], now))
        connection.execute("""INSERT INTO documents(scope,key,media_type,payload,updated_at) VALUES(?,?,?,?,?)
            ON CONFLICT(scope,key) DO UPDATE SET payload=excluded.payload,revision=documents.revision+1,updated_at=excluded.updated_at""",
            (self.target.scope, self.target.key, "application/json", encode_json(state), now))


_REPOSITORIES = {"sqlite": SqliteArrangementRepository}


def create_arrangement_repository(work: Path, implementation: str = "sqlite") -> ArrangementRepository:
    return _REPOSITORIES[implementation](work)
