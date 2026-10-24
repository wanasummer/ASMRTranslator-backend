"""Data contracts shared by Japanese-cleaning implementations and consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MorphToken:
    """Dictionary-independent token emitted by a morphological tokenizer."""

    surface: str
    pos_large: str
    pos_detail: str


@dataclass(frozen=True)
class CleaningRecord:
    """Auditable result for one subtitle block."""

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
    """Output of one cleaner and input to the next pipeline stage."""

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
