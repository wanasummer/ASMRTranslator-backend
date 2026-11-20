"""Toggleable post-translation Chinese short-interjection skip capability (capInterjectionSkip)."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import re
from typing import Any

from ...config import InterjectionSkipConfig
from ...contracts import InterjectionRecord, InterjectionSkipResult


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_cjk_chars(text: str | None) -> int:
    """Count Chinese characters in ``text`` (CJK Unified Ideographs only).

    Punctuation, whitespace, kana, Latin and digits are all excluded, so the
    result is exactly "the number of Chinese characters after removing
    punctuation" — the rule used to detect interjection fragments.
    """
    return len(_CJK_RE.findall(text or ""))


def mark_interjection_skips(
    blocks: Sequence[Any],
    max_chars: int = 3,
    *,
    enabled: bool = True,
) -> int:
    """Flag blocks whose translated text is too short to dub."""
    capability = InterjectionSkipCapability(
        InterjectionSkipConfig(
            enabled=enabled,
            max_chars=max_chars,
        )
    )
    return capability.mark_in_place(blocks)


class InterjectionSkipCapability:
    """Self-contained post-translation short-interjection filter with frontend on/off switch."""

    def __init__(self, config: InterjectionSkipConfig | None = None):
        self.cfg = config or InterjectionSkipConfig()

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @staticmethod
    def count_chars(text: str | None) -> int:
        return count_cjk_chars(text)

    def filter(self, blocks: Sequence[Any]) -> InterjectionSkipResult:
        """Return copied blocks with short Chinese translations marked skip_tts."""
        copied_blocks = [copy.copy(block) for block in blocks]
        records: list[InterjectionRecord] = []
        flagged = 0

        for idx, block in enumerate(copied_blocks):
            text = getattr(block, "text", "") or ""
            cjk_count = count_cjk_chars(text)
            block_index = getattr(block, "index", idx)
            already_skipped = bool(getattr(block, "skip_tts", False))

            if not self.cfg.enabled:
                records.append(InterjectionRecord(
                    index=block_index,
                    text=text,
                    cjk_count=cjk_count,
                    skipped=already_skipped,
                    reason="disabled",
                ))
                continue

            if already_skipped:
                records.append(InterjectionRecord(
                    index=block_index,
                    text=text,
                    cjk_count=cjk_count,
                    skipped=True,
                    reason="upstream_skip_tts",
                ))
                continue

            if cjk_count <= self.cfg.max_chars:
                block.skip_tts = True
                flagged += 1
                records.append(InterjectionRecord(
                    index=block_index,
                    text=text,
                    cjk_count=cjk_count,
                    skipped=True,
                    reason="short_cjk_interjection",
                ))
            else:
                records.append(InterjectionRecord(
                    index=block_index,
                    text=text,
                    cjk_count=cjk_count,
                    skipped=False,
                    reason="kept_translation",
                ))

        return InterjectionSkipResult(
            blocks=tuple(copied_blocks),
            records=tuple(records),
            flagged_count=flagged,
            enabled=self.cfg.enabled,
            max_chars=self.cfg.max_chars,
        )

    def mark_in_place(self, blocks: Sequence[Any]) -> int:
        """Mutate caller-owned blocks in place for legacy pipeline compatibility."""
        if not self.cfg.enabled:
            return 0
        flagged = 0
        for block in blocks:
            if not block.skip_tts and count_cjk_chars(block.text) <= self.cfg.max_chars:
                block.skip_tts = True
                flagged += 1
        return flagged
