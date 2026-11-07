"""Stable ports for reusable text generation and domain-specific capabilities."""

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence
from ..task_cache import ResultCache


class TextRow(Protocol):
    index: int
    text: str
    speaker: str | None


class TimedSourceRow(TextRow, Protocol):
    source_text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class RestructureOperation:
    """One subtitle-block edit. ``indices`` are stable row identifiers in draft order."""
    kind: Literal["modify", "delete", "merge"]
    indices: tuple[int, ...]
    text: str = ""
    reason: str = ""


@dataclass(frozen=True)
class RestructurePlan:
    operations: tuple[RestructureOperation, ...]
    rejected: tuple[str, ...] = field(default=())


class TextGenerator(Protocol):
    cache: ResultCache | None
    def request_key(self, system: str, user: str, max_tokens: int | None) -> str: ...
    async def complete(self, system: str, user: str, max_tokens: int | None = None) -> str: ...


class Summary(Protocol):
    async def summarize(self, blocks: Sequence[TextRow]) -> str: ...


class Translation(Protocol):
    async def translate(self, blocks: Sequence[TextRow], summary: str) -> dict[int, list[str]]: ...
    async def optimize_from_japanese(self, block: TextRow, japanese: str) -> str: ...


class Polish(Protocol):
    async def polish(self, blocks: Sequence[TextRow], summary: str) -> dict[int, str]: ...
    async def optimize_chinese_text(self, block: TextRow, chinese: str) -> str: ...
    async def optimize_japanese_text(self, block: TextRow, japanese: str) -> str: ...


class Restructure(Protocol):
    async def plan(self, rows: Sequence[TimedSourceRow], summary: str, instruction: str) -> RestructurePlan: ...


class Provider(Protocol):
    def endpoint(self) -> str: ...
    def headers(self) -> dict: ...
    def payload(self, system: str, user: str, max_tokens: int | None = None) -> dict: ...
    def parse(self, value: object) -> str: ...
    def is_direct_gemini(self) -> bool: ...
