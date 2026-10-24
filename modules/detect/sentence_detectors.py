"""Local Japanese morphology and sentence-boundary evidence."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Protocol

from .contracts import (
    DetectError,
    JapaneseToken,
    SentenceBoundary,
    SentenceDetection,
    SentenceDetectionInput,
    SentenceDetector,
)
from .duration import count_japanese_mora


_TERMINAL_PUNCTUATION = frozenset("。！？!?…")
_SOFT_PUNCTUATION = frozenset("、，,：:；;")
_CLOSING_QUOTES = frozenset("」』）)]】〉》")
_TERMINAL_PARTICLES = frozenset({
    "ね", "よ", "な", "わ", "ぞ", "ぜ", "さ", "か", "の", "かな", "かしら",
    "よね", "だね", "ですね", "ますね", "ですよ", "でしょう", "だろう",
})
_TERMINAL_AUXILIARIES = frozenset({
    "だ", "です", "ます", "だった", "でした", "ません", "ない", "たい",
    "らしい", "みたい", "でしょう", "だろう", "ください",
})
_CLAUSE_ENDINGS = frozenset({
    "けど", "けれど", "けれども", "が", "から", "ので", "のに", "し", "って",
})
_CONTINUING_PARTICLES = frozenset({
    "を", "に", "へ", "で", "と", "の", "も", "は", "や", "から", "まで",
    "より", "って", "という", "ながら", "たり", "て", "でしょ",
})
_NEW_SENTENCE_CONNECTORS = frozenset({
    "でも", "ただ", "じゃあ", "では", "さて", "ところで", "それで", "そして",
    "しかし", "けど", "あと", "次", "ほら", "ねえ",
})
_NEW_CLAUSE_STARTERS = frozenset({
    "どこ", "だれ", "誰", "なに", "何", "いつ", "どう", "なぜ", "なんで",
    "ここ", "そこ", "あそこ", "私", "わたし", "僕", "ぼく", "俺", "あなた",
})
_NEW_CLAUSE_PREFIXES = frozenset({"なんだか", "なんとなく", "どうやら"})
_REQUEST_ENDINGS = frozenset({"ください", "なさい", "て", "で"})
_QUESTION_ENDINGS = frozenset({"か", "かな", "かしら", "の", "でしょう"})


class JapaneseTokenizer(Protocol):
    @property
    def name(self) -> str: ...

    def tokenize(self, text: str) -> tuple[JapaneseToken, ...]: ...


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).strip()


def _make_token(
    surface: str,
    start: int,
    *,
    pos: str = "",
    pos_detail: str = "",
    conjugation_type: str = "",
    conjugation_form: str = "",
    reading: str = "",
    unknown: bool = False,
) -> JapaneseToken:
    return JapaneseToken(
        surface=surface,
        start_char=start,
        end_char=start + len(surface),
        pos=pos,
        pos_detail=pos_detail,
        conjugation_type=conjugation_type,
        conjugation_form=conjugation_form,
        reading=reading if reading != "*" else "",
        mora=count_japanese_mora(reading, surface),
        unknown=unknown,
    )


class MecabJapaneseTokenizer:
    """UniDic-backed tokenizer with offsets and readings for alignment."""

    def __init__(self):
        try:
            import MeCab
            import unidic_lite

            dictionary = os.fspath(unidic_lite.DICDIR)
            required = ("dicrc", "sys.dic", "matrix.bin", "char.bin", "unk.dic")
            if any(
                not os.path.isfile(os.path.join(dictionary, name))
                or os.path.getsize(os.path.join(dictionary, name)) == 0
                for name in required
            ):
                raise FileNotFoundError("UniDic runtime data is incomplete")
            args = f'-r "{os.devnull}" -d "{dictionary}"'
            self._tagger = MeCab.Tagger(args)
            self._tagger.parse("")
        except (ImportError, OSError, RuntimeError) as exc:
            raise DetectError(
                "DETECT_ANALYZER_UNAVAILABLE",
                f"MeCab/UniDic unavailable: {exc}",
            ) from exc

    @property
    def name(self) -> str:
        return "mecab-unidic-lite"

    def tokenize(self, text: str) -> tuple[JapaneseToken, ...]:
        tokens: list[JapaneseToken] = []
        cursor = 0
        node = self._tagger.parseToNode(text)
        while node is not None:
            surface = node.surface or ""
            if surface:
                start = text.find(surface, cursor)
                if start < 0:
                    start = cursor
                features = (node.feature or "").split(",")
                unknown = getattr(node, "stat", 0) == 1
                token = _make_token(
                    surface,
                    start,
                    pos=features[0] if features else "",
                    pos_detail=features[1] if len(features) > 1 else "",
                    conjugation_type=features[4] if len(features) > 4 else "",
                    conjugation_form=features[5] if len(features) > 5 else "",
                    reading=features[9] if len(features) > 9 else "",
                    unknown=unknown,
                )
                # A long unknown ASR span would otherwise offer no possible text
                # boundary. Split it into conservative characters while retaining
                # its low-confidence status.
                if unknown and len(surface) > 1:
                    for offset, char in enumerate(surface):
                        tokens.append(_make_token(char, start + offset, unknown=True))
                else:
                    tokens.append(token)
                cursor = start + len(surface)
            node = node.next
        return tuple(tokens)


class ConservativeJapaneseTokenizer:
    """Fallback that permits acoustic alignment without inventing morphology."""

    @property
    def name(self) -> str:
        return "conservative-character"

    def tokenize(self, text: str) -> tuple[JapaneseToken, ...]:
        return tuple(
            _make_token(char, index, unknown=char not in _TERMINAL_PUNCTUATION)
            for index, char in enumerate(text)
        )


def _last_content(tokens, end: int) -> JapaneseToken | None:
    for token in reversed(tokens[:end]):
        if not all(
            char in _TERMINAL_PUNCTUATION | _SOFT_PUNCTUATION | _CLOSING_QUOTES
            for char in token.surface
        ):
            return token
    return None


def _is_terminal_particle(token: JapaneseToken) -> bool:
    if token.pos == "助詞":
        return token.pos_detail in ("終助詞", "間投助詞")
    return (
        (token.unknown or not token.pos)
        and token.surface in _TERMINAL_PARTICLES
        and token.surface != "の"
    )


def _is_nominal_no_ending(tokens, end: int) -> bool:
    """Recognize sentence-final ``なの`` when punctuation is absent.

    UniDic labels ``の`` as a nominalizing particle in both ``朝なの`` and a
    continuing nominal phrase.  Only promote it when it follows the copular
    auxiliary ``な`` and the next token can plausibly begin a new utterance.
    The acoustic scorer still decides whether an actual cut exists.
    """
    if end < 2 or tokens[end - 1].surface != "の":
        return False
    previous = tokens[end - 2]
    following = tokens[end] if end < len(tokens) else None
    return (
        previous.surface == "な"
        and previous.pos == "助動詞"
        and (
            following is None
            or following.pos not in ("助詞", "助動詞", "接尾辞")
        )
    )


def _sentence_type(tokens, end: int) -> str:
    surfaces = "".join(token.surface for token in tokens[:end])
    last = _last_content(tokens, end)
    if not last:
        return "fragment"
    prior = _last_content(tokens, end - 1) if end > 1 else None
    question_ending = (
        last.surface in _QUESTION_ENDINGS
        and (_is_terminal_particle(last) or last.surface == "でしょう")
    )
    vocative_after_question = (
        last.pos_detail == "固有名詞"
        and prior is not None
        and prior.surface in _QUESTION_ENDINGS
    )
    if (
        any(char in surfaces[-3:] for char in "？?")
        or question_ending
        or _is_nominal_no_ending(tokens, end)
        or vocative_after_question
    ):
        return "question"
    if last.surface in _REQUEST_ENDINGS or "命令形" in last.conjugation_form:
        return "request"
    if last.pos in ("感動詞", "フィラー"):
        return "interjection"
    if last.surface in _CLAUSE_ENDINGS or "連用形" in last.conjugation_form:
        return "continuation"
    if last.pos in ("動詞", "形容詞", "助動詞") or last.surface in _TERMINAL_PARTICLES:
        return "statement"
    return "fragment"


def _boundary_evidence(tokens, end: int, parse_confidence: float):
    current = tokens[end - 1]
    following = tokens[end] if end < len(tokens) else None
    surface = current.surface
    last = _last_content(tokens, end)
    reasons: list[str] = []
    score = 0.12

    if any(char in _TERMINAL_PUNCTUATION for char in surface):
        score = 1.0
        reasons.append("terminal_punctuation")
    elif surface in _CLOSING_QUOTES and end > 1 and any(
        char in _TERMINAL_PUNCTUATION for char in tokens[end - 2].surface
    ):
        score = 0.96
        reasons.append("quoted_terminal")
    elif last is not None:
        if _is_terminal_particle(last):
            score = max(score, 0.86)
            reasons.append("terminal_particle")
        if _is_nominal_no_ending(tokens, end):
            score = max(score, 0.68)
            reasons.append("nominal_no_ending")
        if last.surface in _TERMINAL_AUXILIARIES:
            score = max(score, 0.76)
            reasons.append("terminal_auxiliary")
        if (
            last.pos in ("動詞", "形容詞", "助動詞")
            and ("終止形" in last.conjugation_form or "命令形" in last.conjugation_form)
        ):
            score = max(score, 0.62)
            reasons.append("terminal_inflection")
        if last.surface in _CLAUSE_ENDINGS:
            score = max(score, 0.42)
            reasons.append("clause_ending")
        if last.surface in _CONTINUING_PARTICLES and last.surface not in _TERMINAL_PARTICLES:
            score = min(score, 0.16)
            reasons.append("continuing_particle")
        if "連体形" in last.conjugation_form and following and following.pos == "名詞":
            score = min(score, 0.05)
            reasons.append("relative_clause")
        if "連用形" in last.conjugation_form:
            score = min(score, 0.12)
            reasons.append("continuative_inflection")
        if (
            last.pos in ("名詞", "代名詞", "接頭辞")
            and following
            and following.pos in ("助詞", "助動詞", "接尾辞", "形容詞")
        ):
            score = min(score, 0.05)
            reasons.append("nominal_continuation")

    if following and following.surface in _NEW_SENTENCE_CONNECTORS:
        score = max(score, 0.80)
        reasons.append("new_sentence_connector")
    following_phrase = "".join(token.surface for token in tokens[end:end + 4])
    if any(following_phrase.startswith(item) for item in _NEW_SENTENCE_CONNECTORS):
        score = max(score, 0.80)
        reasons.append("new_sentence_connector")
    if (
        last is not None
        and last.pos in ("動詞", "形容詞", "助動詞")
        and any(following_phrase.startswith(item) for item in _NEW_CLAUSE_PREFIXES)
    ):
        score = max(score, 0.68)
        reasons.append("new_clause_starter")
    if (
        following
        and current.surface == "か"
        and current.pos_detail == "副助詞"
        and end >= 2
        and tokens[end - 2].surface == "と"
        and following.pos in ("名詞", "代名詞")
        and "".join(token.surface for token in tokens[max(0, end - 3):end])
        not in ("なんとか", "何とか", "なんだか", "何だか")
    ):
        # ``とか`` can close an informal utterance. Keep it as weak evidence
        # before a new nominal phrase, while excluding fixed adverbs such as
        # なんとか and なんだか.
        score = max(score, 0.52)
        reasons.append("quotative_terminal")
    if (
        following
        and following.surface in _NEW_CLAUSE_STARTERS
        and last is not None
        and last.pos in ("動詞", "形容詞", "助動詞")
    ):
        score = max(score, 0.48)
        reasons.append("new_clause_starter")
    if (
        following
        and following.pos == "助詞"
        and following.pos_detail in ("終助詞", "間投助詞")
    ):
        score = min(score, 0.05)
        reasons.append("terminal_particle_follows")
    if (
        following
        and current.surface == "だ"
        and following.surface == "か"
        and end >= 2
        and tokens[end - 2].surface in ("なん", "何")
    ):
        score = min(score, 0.03)
        reasons.append("fixed_expression_continuation")
    if (
        following
        and current.surface == "だ"
        and following.surface == "けど"
        and end >= 2
        and tokens[end - 2].surface in ("ん", "なん")
    ):
        score = min(score, 0.03)
        reasons.append("fixed_expression_continuation")
    if (
        following
        and end + 1 < len(tokens)
        and current.pos == "動詞"
        and current.conjugation_form.startswith("命令形")
        and following.pos == "助動詞"
        and tokens[end + 1].pos == "接尾辞"
    ):
        # An imperative + auxiliary + suffix chain is morphologically
        # inconsistent and usually means ASR split a lexical word. Keep it
        # intact instead of trusting the apparently sentence-final verb.
        score = min(score, 0.04)
        reasons.append("suspect_asr_word_fragment")
    if (
        following
        and "terminal_particle" in reasons
        and following.pos in ("動詞", "形容詞", "副詞")
        and "terminal_punctuation" not in reasons
    ):
        # ASR often hallucinates a short sentence-final particle inside a word
        # sequence (for example ``どこがよ変わらず``).  Without punctuation,
        # a following predicate is continuation evidence; a genuinely long VAD
        # pause can still overcome this soft penalty in the joint scorer.
        score = min(score, 0.28)
        reasons.append("terminal_particle_predicate_follows")
    if (
        following
        and following.surface in _CONTINUING_PARTICLES
        and "new_sentence_connector" not in reasons
    ):
        score = min(score, 0.06)
        reasons.append("continuing_particle_follows")
    remaining = tokens[end:]
    if (
        "terminal_punctuation" not in reasons
        and len(remaining) == 1
        and remaining[0].pos_detail == "固有名詞"
        and remaining[0].mora <= 4
    ):
        score = min(score, 0.24)
        reasons.append("trailing_vocative")
    if any(char in _SOFT_PUNCTUATION for char in surface):
        score = max(score, 0.30)
        reasons.append("soft_punctuation")

    # Explicit punctuation survives poor ASR tokenization. Morphology-dependent
    # evidence is shrunk when unknown tokens dominate.
    if "terminal_punctuation" not in reasons and "quoted_terminal" not in reasons:
        score *= 0.35 + 0.65 * parse_confidence
    return min(1.0, max(0.0, score)), tuple(dict.fromkeys(reasons))


class MecabJapaneseSentenceDetector:
    """Emit boundary evidence; the audio-aware aligner makes the final cuts."""

    def __init__(self, tokenizer: JapaneseTokenizer | None = None):
        self.tokenizer = tokenizer or MecabJapaneseTokenizer()

    def detect(self, value: SentenceDetectionInput) -> SentenceDetection:
        text = _normalized(value.source_text)
        if not text:
            return SentenceDetection("", (), (), self.tokenizer.name, 0.0, "empty_asr")
        tokenizer_name = self.tokenizer.name
        try:
            tokens = self.tokenizer.tokenize(text)
        except Exception as exc:
            fallback = ConservativeJapaneseTokenizer()
            tokens = fallback.tokenize(text)
            tokenizer_name = f"{fallback.name}:runtime-fallback"
            print(f"       [warn] 日语形态分析失败，detect 使用保守字符分析: {exc}")
        if not tokens:
            return SentenceDetection(text, (), (), tokenizer_name, 0.0, "no_tokens")

        lexical = [token for token in tokens if token.mora > 0]
        lexical_weight = sum(max(token.mora, 1.0) for token in lexical)
        unknown_weight = sum(max(token.mora, 1.0) for token in lexical if token.unknown)
        if not lexical_weight:
            confidence = 0.0
        elif all(not token.pos for token in lexical):
            confidence = 0.15
        else:
            confidence = max(0.05, 1.0 - unknown_weight / lexical_weight)

        boundaries = []
        for end in range(1, len(tokens) + 1):
            if end == len(tokens):
                score, reasons = 1.0, ("end_of_slice",)
            else:
                score, reasons = _boundary_evidence(tokens, end, confidence)
            boundaries.append(SentenceBoundary(
                token_index=end,
                char_index=tokens[end - 1].end_char,
                language_score=round(score, 6),
                sentence_type=_sentence_type(tokens, end),
                reasons=reasons,
            ))
        return SentenceDetection(
            text,
            tokens,
            tuple(boundaries),
            tokenizer_name,
            round(confidence, 6),
        )


SENTENCE_DETECTORS = {"mecab-japanese": MecabJapaneseSentenceDetector}


def create_sentence_detector(name: str) -> SentenceDetector:
    try:
        factory = SENTENCE_DETECTORS[name]
    except KeyError as exc:
        raise DetectError(
            "DETECT_IMPLEMENTATION_UNKNOWN",
            f"Unknown sentence detector: {name}",
        ) from exc
    try:
        return factory()
    except DetectError as exc:
        if exc.code != "DETECT_ANALYZER_UNAVAILABLE":
            raise
        print(f"       [warn] {exc}; detect 降级为保守字符分析")
        return MecabJapaneseSentenceDetector(ConservativeJapaneseTokenizer())
