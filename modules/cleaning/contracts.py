"""Stable data contracts and interfaces for subtitle cleaning capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class MorphToken:
    """Dictionary-independent token emitted by a morphological tokenizer."""

    surface: str
    pos_large: str
    pos_detail: str


class MorphTokenizer(Protocol):
    """Small adapter contract that keeps the cleaner independent of MeCab."""

    @property
    def name(self) -> str: ...

    def tokenize(self, text: str) -> list[MorphToken]: ...


@dataclass(frozen=True)
class CleaningRecord:
    """Auditable result for one Japanese subtitle block."""

    index: int
    original: str
    cleaned: str
    action: str
    reason: str
    removed_tokens: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "original": self.original,
            "cleaned": self.cleaned,
            "action": self.action,
            "reason": self.reason,
            "removed_tokens": list(self.removed_tokens),
        }


@dataclass
class JapaneseCleaningResult:
    """Output of capJapaneseClean and input to the next pipeline stage."""

    blocks: list[Any]
    records: list[CleaningRecord]
    cleaner: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kept_count(self) -> int:
        return sum(record.action == "translate" for record in self.records)

    @property
    def skipped_count(self) -> int:
        return sum(record.action == "skip_empty_keep_original" for record in self.records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cleaner": self.cleaner,
            "total": len(self.records),
            "kept": self.kept_count,
            "skipped": self.skipped_count,
            "metadata": self.metadata,
            "records": [record.as_dict() for record in self.records],
        }


class JapaneseCleanerUnavailableError(RuntimeError):
    """Raised when a configured cleaner cannot initialize its runtime data."""


class JapaneseCleaner(ABC):
    """Transform ASR subtitle blocks without knowing later LLM/TTS stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable implementation identifier used in diagnostics."""

    @abstractmethod
    def clean(self, blocks: Sequence[Any]) -> JapaneseCleaningResult:
        """Return new blocks plus an audit report; never mutate caller blocks."""


@dataclass(frozen=True)
class NonJapaneseFilterRecord:
    """Auditable result for one ASR block checked for Japanese content."""

    index: int
    text: str
    kept: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "kept": self.kept,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NonJapaneseFilterResult:
    """Output of capNonJapaneseFilter; removed blocks keep no timeline entry.

    Kept blocks are copies that preserve their original ``index`` so later
    stages can still trace them to raw ASR output.
    """

    blocks: tuple[Any, ...]
    records: tuple[NonJapaneseFilterRecord, ...]
    enabled: bool

    @property
    def removed_count(self) -> int:
        return sum(not record.kept for record in self.records)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total": len(self.records),
            "kept": len(self.records) - self.removed_count,
            "removed": self.removed_count,
            "records": [record.as_dict() for record in self.records],
        }


@dataclass(frozen=True)
class InterjectionRecord:
    """Auditable result for one translated block checked for short-interjection skip."""

    index: int
    text: str
    cjk_count: int
    skipped: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "cjk_count": self.cjk_count,
            "skipped": self.skipped,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InterjectionSkipResult:
    """Immutable result of capInterjectionSkip."""

    blocks: tuple[Any, ...]
    records: tuple[InterjectionRecord, ...]
    flagged_count: int
    enabled: bool
    max_chars: int
