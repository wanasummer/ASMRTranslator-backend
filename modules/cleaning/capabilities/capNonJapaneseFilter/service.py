"""Toggleable post-ASR filter that drops blocks without any Japanese text (capNonJapaneseFilter)."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import re
from typing import Any

from ...config import NonJapaneseFilterConfig
from ...contracts import NonJapaneseFilterRecord, NonJapaneseFilterResult


POLICY_VERSION = "kana-required-v2"

# Kana only (excluding the bare prolonged-sound mark and middle dot). Kanji
# alone does not count: misrecognized interjections often come back as
# kanji-only fragments.
_KANA_RE = re.compile(
    r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\u31f0-\u31ff\uff66-\uff6f\uff71-\uff9d]"
)


def contains_japanese(text: str | None) -> bool:
    """Return True when ``text`` contains at least one hiragana or katakana character."""
    return _KANA_RE.search(text or "") is not None


class NonJapaneseFilterCapability:
    """Drop ASR blocks whose text has no kana at all.

    Targets ASR hallucinations such as English credits, music notes, symbols,
    kanji-only misrecognized interjections, or empty slices. Removed blocks
    leave the timeline; neighbouring blocks keep the original audio around them.
    """

    def __init__(self, config: NonJapaneseFilterConfig | None = None):
        self.cfg = config or NonJapaneseFilterConfig()

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def filter(self, blocks: Sequence[Any]) -> NonJapaneseFilterResult:
        """Return copies of the kept blocks plus an audit record per input block."""
        kept: list[Any] = []
        records: list[NonJapaneseFilterRecord] = []
        for position, block in enumerate(blocks):
            text = (getattr(block, "text", "") or "").strip()
            index = getattr(block, "index", position)
            if not self.cfg.enabled:
                keep, reason = True, "disabled"
            elif contains_japanese(text):
                keep, reason = True, "contains_kana"
            else:
                keep, reason = False, "empty" if not text else "no_kana"
            if keep:
                kept.append(copy.copy(block))
            records.append(NonJapaneseFilterRecord(index=index, text=text, kept=keep, reason=reason))
        return NonJapaneseFilterResult(
            blocks=tuple(kept),
            records=tuple(records),
            enabled=self.cfg.enabled,
        )
