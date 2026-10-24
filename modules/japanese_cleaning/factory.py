"""Factory/registry for interchangeable Japanese-cleaning implementations."""

from collections.abc import Callable

from .base import JapaneseCleaner
from .mecab_regex import MecabRegexJapaneseCleaner, RegexJapaneseCleaner


_CLEANERS: dict[str, Callable[[], JapaneseCleaner]] = {
    "mecab-regex": MecabRegexJapaneseCleaner,
    "regex-only": RegexJapaneseCleaner,
}


def register_japanese_cleaner(name: str, factory: Callable[[], JapaneseCleaner]) -> None:
    """Register a cleaner without changing pipeline orchestration code."""
    _CLEANERS[name] = factory


def create_japanese_cleaner(name: str = "mecab-regex") -> JapaneseCleaner:
    try:
        return _CLEANERS[name]()
    except KeyError as error:
        available = ", ".join(sorted(_CLEANERS))
        raise ValueError(f"未知日语清洗器 {name!r}；可用模块：{available}") from error
