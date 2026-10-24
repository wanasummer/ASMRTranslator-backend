"""MeCab POS filtering with deterministic ASMR-noise regex safeguards."""

from __future__ import annotations

import copy
import os
import re
import unicodedata
from typing import Protocol

from .base import JapaneseCleaner, JapaneseCleanerUnavailableError
from .models import CleaningRecord, JapaneseCleaningResult, MorphToken


DROP_POS_LARGE = frozenset({"感動詞", "フィラー"})
DROP_POS_DETAIL = frozenset({("助詞", "終助詞"), ("助詞", "間投助詞")})

# Fish ASR occasionally emits Chinese graphemes for non-lexical breathing sounds
# even when language=ja. MeCab cannot classify these with a Japanese dictionary.
_CHINESE_VOCALIZATION_RE = re.compile(r"[嗯啊阿呃唔哦噢呀哼欸诶]+")
_LONG_MARK_RUN_RE = re.compile(r"[ー~〜]{2,}")
_BREATH_RUN_RE = re.compile(
    r"(?:[ぁあァアぅうゥウぇえェエぉおォオんンっッ][ー~〜っッ]*){2,}"
)
_PURE_VOCALIZATION_RE = re.compile(
    r"[ぁあァアぅうゥウぇえェエぉおォオんンっッー~〜…・.。、,!！?？\s]+"
)
_EFFECTIVE_CONTENT_RE = re.compile(
    r"[一-龯々〆ヵヶぁ-ゖァ-ヺA-Za-z0-9]"
)


class MorphTokenizer(Protocol):
    """Small adapter contract that keeps the cleaner independent of MeCab."""

    @property
    def name(self) -> str: ...

    def tokenize(self, text: str) -> list[MorphToken]: ...


class MecabTokenizer:
    """MeCab tokenizer pinned to the bundled unidic-lite dictionary."""

    def __init__(self):
        try:
            import MeCab
            import unidic_lite

            dictionary = os.fspath(unidic_lite.DICDIR)
            required = ("dicrc", "sys.dic", "matrix.bin", "char.bin", "unk.dic")
            missing = [
                filename for filename in required
                if not os.path.isfile(os.path.join(dictionary, filename))
                or os.path.getsize(os.path.join(dictionary, filename)) == 0
            ]
            if missing:
                raise FileNotFoundError(
                    f"UniDic 运行文件缺失或为空：{', '.join(missing)}"
                )

            # Explicit dictionary selection makes dev and packaged behavior identical.
            args = f'-r "{os.devnull}" -d "{dictionary}"'
            self._tagger = MeCab.Tagger(args)
            self._tagger.parse("")
        except (ImportError, OSError, RuntimeError) as error:
            raise JapaneseCleanerUnavailableError(
                f"MeCab/UniDic 初始化失败：{error}"
            ) from error

    @property
    def name(self) -> str:
        return "mecab-unidic-lite"

    def tokenize(self, text: str) -> list[MorphToken]:
        tokens: list[MorphToken] = []
        node = self._tagger.parseToNode(text)
        while node is not None:
            surface = node.surface or ""
            if surface:
                features = (node.feature or "").split(",")
                tokens.append(MorphToken(
                    surface=surface,
                    pos_large=features[0] if features else "",
                    pos_detail=features[1] if len(features) > 1 else "",
                ))
            node = node.next
        return tokens


class MecabRegexJapaneseCleaner(JapaneseCleaner):
    """Clean Japanese ASR text while preserving the existing block timeline."""

    def __init__(self, tokenizer: MorphTokenizer | None = None):
        self._tokenizer = tokenizer or MecabTokenizer()

    @property
    def name(self) -> str:
        return "mecab-regex"

    @staticmethod
    def _drop_token(token: MorphToken) -> bool:
        return (
            token.pos_large in DROP_POS_LARGE
            or (token.pos_large, token.pos_detail) in DROP_POS_DETAIL
            or bool(_PURE_VOCALIZATION_RE.fullmatch(token.surface))
        )

    def clean_text(self, text: str) -> tuple[str, tuple[str, ...]]:
        normalized = unicodedata.normalize("NFKC", text or "").strip()
        normalized = _CHINESE_VOCALIZATION_RE.sub("", normalized)
        if not normalized:
            return "", ()

        kept: list[str] = []
        removed: list[str] = []
        for token in self._tokenizer.tokenize(normalized):
            if self._drop_token(token):
                removed.append(token.surface)
            else:
                kept.append(token.surface)

        cleaned = "".join(kept)
        cleaned = _CHINESE_VOCALIZATION_RE.sub("", cleaned)
        cleaned = _BREATH_RUN_RE.sub("", cleaned)
        cleaned = _LONG_MARK_RUN_RE.sub("", cleaned).strip()
        if not _EFFECTIVE_CONTENT_RE.search(cleaned):
            cleaned = ""
        return cleaned, tuple(removed)

    def clean(self, blocks) -> JapaneseCleaningResult:
        cleaned_blocks = []
        records: list[CleaningRecord] = []
        for block in blocks:
            original = block.text or ""
            cleaned, removed = self.clean_text(original)
            output = copy.copy(block)
            output.text = cleaned
            output.source_text = getattr(block, "source_text", "") or original
            output.cleaned_source_text = cleaned
            if cleaned:
                action = "translate"
                reason = "清洗后仍有有效日语内容"
            else:
                output.skip_tts = True
                action = "skip_empty_keep_original"
                reason = "日语语气词清洗后为空或只剩标点"
            cleaned_blocks.append(output)
            records.append(CleaningRecord(
                index=block.index,
                original=original,
                cleaned=cleaned,
                action=action,
                reason=reason,
                removed_tokens=removed,
            ))
        return JapaneseCleaningResult(
            blocks=cleaned_blocks,
            records=records,
            cleaner=self.name,
            metadata={"tokenizer": self._tokenizer.name},
        )


class RegexTokenizer:
    """Conservative tokenizer used when the optional MeCab runtime is unavailable."""

    @property
    def name(self) -> str:
        return "regex-only"

    def tokenize(self, text: str) -> list[MorphToken]:
        if not text:
            return []
        return [MorphToken(surface=text, pos_large="", pos_detail="")]


class RegexJapaneseCleaner(MecabRegexJapaneseCleaner):
    """Keep deterministic noise filtering available without MeCab POS tagging."""

    def __init__(self):
        super().__init__(tokenizer=RegexTokenizer())

    @property
    def name(self) -> str:
        return "regex-only"
