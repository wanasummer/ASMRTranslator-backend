"""Toggleable pre-translation Japanese ASR cleaning capability (capJapaneseClean)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
from typing import Any

from ...config import JapaneseCleanConfig
from ...contracts import CleaningRecord, JapaneseCleaner, JapaneseCleaningResult
from .mecab_regex import MecabRegexJapaneseCleaner, RegexJapaneseCleaner


_CLEANERS: dict[str, Callable[[], JapaneseCleaner]] = {
    "mecab-regex": MecabRegexJapaneseCleaner,
    "regex-only": RegexJapaneseCleaner,
}


def register_japanese_cleaner(name: str, factory: Callable[[], JapaneseCleaner]) -> None:
    """Register a cleaner backend without changing pipeline orchestration code."""
    _CLEANERS[name] = factory


def create_japanese_cleaner(name: str = "mecab-regex") -> JapaneseCleaner:
    try:
        return _CLEANERS[name]()
    except KeyError as error:
        available = ", ".join(sorted(_CLEANERS))
        raise ValueError(f"未知日语清洗器 {name!r}；可用模块：{available}") from error


class JapaneseCleanCapability(JapaneseCleaner):
    """Self-contained Japanese cleaning capability with frontend on/off bypass.

    When ``config.is_enabled`` is False, MeCab/UniDic is never initialized and
    blocks pass through copied and uncleaned while still emitting a valid
    ``JapaneseCleaningResult`` for downstream stages and audit artifacts.
    """

    def __init__(
        self,
        config: JapaneseCleanConfig | None = None,
        *,
        cleaner: JapaneseCleaner | None = None,
        cleaner_factory: Callable[[str], JapaneseCleaner] = create_japanese_cleaner,
    ):
        self.cfg = config or JapaneseCleanConfig()
        self._cleaner = cleaner
        self._cleaner_factory = cleaner_factory
        if self.cfg.is_enabled and self._cleaner is None:
            # Fail fast on initialization when enabled so unavailable runtime
            # dependencies surface immediately at stage setup.
            self._cleaner = self._cleaner_factory(self.cfg.cleaner)

    @property
    def enabled(self) -> bool:
        return self.cfg.is_enabled

    @property
    def name(self) -> str:
        if not self.enabled:
            return "disabled"
        return self._cleaner.name if self._cleaner is not None else self.cfg.cleaner

    def clean_text(self, text: str) -> tuple[str, tuple[str, ...]]:
        if not self.enabled:
            return text or "", ()
        clean_fn = getattr(self._cleaner, "clean_text", None)
        if callable(clean_fn):
            return clean_fn(text)
        raise AttributeError(f"Cleaner {self.name!r} does not implement clean_text")

    def clean(self, blocks: Sequence[Any]) -> JapaneseCleaningResult:
        if not self.enabled:
            return self._passthrough(blocks)
        return self._cleaner.clean(blocks)

    @staticmethod
    def _passthrough(blocks: Sequence[Any]) -> JapaneseCleaningResult:
        copied_blocks = []
        records: list[CleaningRecord] = []
        for block in blocks:
            original = (getattr(block, "text", "") or "").strip()
            output = copy.copy(block)
            output.source_text = getattr(block, "source_text", "") or original
            output.cleaned_source_text = original
            if original:
                action = "translate"
                reason = "日语清洗已关闭，保留原始识别内容"
            else:
                output.skip_tts = True
                action = "skip_empty_keep_original"
                reason = "原始识别内容为空"
            copied_blocks.append(output)
            records.append(CleaningRecord(
                index=block.index,
                original=original,
                cleaned=original,
                action=action,
                reason=reason,
                removed_tokens=(),
            ))
        return JapaneseCleaningResult(
            blocks=copied_blocks,
            records=records,
            cleaner="disabled",
            metadata={"enabled": False},
        )
