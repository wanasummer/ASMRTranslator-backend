"""Qwen-only Japanese sentence policy over aligned words.

No FSMN atoms, MeCab boundaries, mora estimates or joint partitioner are used.
Input times are relative to one ASR slice; output indices retain word order.
"""

import re
import unicodedata
from typing import Protocol

from .qwen_config import QwenConfig


_HARD_TERMINAL_PUNCT = frozenset("。！？!?")
_ELLIPSIS_PUNCT = frozenset("….．")
_SOFT_PUNCT = frozenset("、，,：:；;～~")
_BINDING_PARTICLES = frozenset({
    "を", "に", "へ", "で", "と", "の", "も", "は", "が", "や",
    "から", "まで", "より", "って", "お", "ご",
})
_CONTINUATION_FOLLOWER_PREFIXES = (
    "され", "れる", "られる", "せる", "させる", "ちゃ", "じゃ",
    "てる", "でる", "てい", "でい", "ます", "です", "ました", "でした",
    "ない", "たい", "たがる",
)
_CONTINUATION_FOLLOWERS = frozenset({
    "を", "に", "へ", "で", "と", "の", "も", "は", "が", "や", "って",
    "ね", "よ", "な", "わ", "さ", "ぞ", "ぜ", "か", "かな", "かしら",
    "た", "だ", "て", "で",
})
_SENTENCE_FINAL_SUFFIXES = (
    "ますね", "ですね", "ますよ", "ですよ", "ました", "でした",
    "ましょう", "でしょう", "だろう", "ください", "ちょうだい",
    "ます", "です", "よね", "だね", "かな", "かしら",
    "ね", "よ", "わ", "ぞ", "ぜ",
)
_CLAUSE_FINAL_SUFFIXES = (
    "けれども", "けれど", "けど", "から", "ので", "のに", "って", "て", "で", "し",
)
_NEW_SENTENCE_STARTERS = (
    "でも", "ただ", "じゃあ", "では", "さて", "ところで", "それで",
    "そして", "しかし", "ほら", "ねえ", "次",
)


def lexical(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if c.isalnum())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).strip()


def _is_pure_vocalization(text: str, vocalization_chars: str) -> bool:
    lex = lexical(text)
    if not lex:
        return True
    allowed = set(vocalization_chars) | {"っ", "ッ", "ー", "ふ", "フ", "は", "ハ"}
    return all(ch in allowed for ch in lex)


def _qwen_acoustic_score(gap_ms: int) -> float:
    if gap_ms <= 60:
        return 0.0
    if gap_ms <= 160:
        return 0.10 + 0.15 * ((gap_ms - 60) / 100.0)
    if gap_ms < 250:
        return 0.25 + 0.25 * ((gap_ms - 160) / 90.0)
    if gap_ms < 400:
        return 0.50 + 0.30 * ((gap_ms - 250) / 150.0)
    if gap_ms < 600:
        return 0.80 + 0.12 * ((gap_ms - 400) / 200.0)
    return min(1.0, 0.92 + 0.08 * min(1.0, (gap_ms - 600) / 300.0))


def _lexical_span_end(text: str, target_count: int) -> tuple[int, int]:
    """Return (lex_end_idx, char_end_idx_including_trailing_non_lexical)."""
    if target_count <= 0:
        return 0, 0
    seen = 0
    lex_end = len(text)
    for idx, ch in enumerate(text):
        seen += len(lexical(ch))
        if seen >= target_count:
            lex_end = idx + 1
            break
    char_end = lex_end
    while char_end < len(text) and not lexical(text[char_end]):
        char_end += 1
    return lex_end, char_end


def _infer_sentence_type(curr_surface: str, inter_punct: str) -> str:
    if any(ch in "？?" for ch in inter_punct) or curr_surface.endswith(("か", "かな", "かしら", "でしょう")):
        return "question"
    if curr_surface.endswith(("ください", "ちょうだい", "なさい")):
        return "request"
    if any(ch in _HARD_TERMINAL_PUNCT | _ELLIPSIS_PUNCT for ch in inter_punct) or curr_surface.endswith(_SENTENCE_FINAL_SUFFIXES):
        return "statement"
    if any(ch in _SOFT_PUNCT for ch in inter_punct) or curr_surface.endswith(_CLAUSE_FINAL_SUFFIXES):
        return "continuation"
    return "statement"


def _continues_phrase(curr_surface: str, next_surface: str, remaining_text: str,
                      inter_punct: str) -> bool:
    """Recognize attached particles/auxiliaries even when Qwen splits their words.

    ASMR pauses and ellipses can occur inside a predicate. They must not detach
    its suffix; explicit sentence-final punctuation still owns a real boundary.
    """
    if any(ch in _HARD_TERMINAL_PUNCT for ch in inter_punct):
        return False
    return (
        curr_surface in _BINDING_PARTICLES
        or next_surface in _CONTINUATION_FOLLOWERS
        or lexical(remaining_text).startswith(_CONTINUATION_FOLLOWER_PREFIXES)
    )


def _score_qwen_word_boundary(
    curr_surface: str,
    next_surface: str,
    inter_punct: str,
    gap_ms: int,
    config: QwenConfig,
) -> tuple[float, bool, str]:
    acoustic = _qwen_acoustic_score(gap_ms)
    follower_continues = (
        next_surface in _CONTINUATION_FOLLOWERS
        or next_surface.startswith(_CONTINUATION_FOLLOWER_PREFIXES)
    )
    curr_binds = curr_surface in _BINDING_PARTICLES
    curr_vocalization = _is_pure_vocalization(curr_surface, config.vocalization_chars)

    has_hard_punct = any(ch in _HARD_TERMINAL_PUNCT for ch in inter_punct)
    has_ellipsis = any(ch in _ELLIPSIS_PUNCT for ch in inter_punct)
    has_soft_punct = any(ch in _SOFT_PUNCT for ch in inter_punct)

    if has_hard_punct:
        punct_score = 1.0
        strong_punct_cut = True
    elif has_ellipsis:
        if (follower_continues or curr_binds) and gap_ms < 320:
            punct_score = 0.24
            strong_punct_cut = False
        elif curr_vocalization and gap_ms < 320:
            punct_score = 0.35
            strong_punct_cut = False
        else:
            punct_score = 0.92
            strong_punct_cut = True
    elif has_soft_punct:
        punct_score = 0.45
        strong_punct_cut = False
    else:
        punct_score = 0.0
        strong_punct_cut = False

    lexical_score = 0.12
    if curr_surface.endswith(_SENTENCE_FINAL_SUFFIXES):
        lexical_score = max(lexical_score, 0.78)
    elif curr_surface.endswith(_CLAUSE_FINAL_SUFFIXES):
        lexical_score = max(lexical_score, 0.42)
    if next_surface.startswith(_NEW_SENTENCE_STARTERS):
        lexical_score = max(lexical_score, 0.76)

    language = max(punct_score, lexical_score)
    score = min(
        1.5,
        max(
            punct_score,
            config.language_weight * language
            + config.acoustic_weight * acoustic
            + config.boundary_synergy_weight * min(language, acoustic),
        ),
    )

    hard_continuation = (not inter_punct) and (curr_binds or follower_continues)
    if hard_continuation and gap_ms < 450:
        score -= 0.55

    should_cut = (
        strong_punct_cut
        or score >= config.boundary_bias
        or (gap_ms >= 350 and (not hard_continuation or gap_ms >= 550))
        or (gap_ms >= 200 and language >= 0.42 and not hard_continuation)
    )
    sentence_type = _infer_sentence_type(curr_surface, inter_punct)
    return score, should_cut, sentence_type


class QwenWordPartitioner(Protocol):
    def partition(self, words: list[dict], text: str) -> tuple[list[int], list[dict]]: ...


class QwenSentencePartitioner:
    def __init__(self, config: QwenConfig):
        self.config = config

    def partition(
        self,
        words: list[dict],
        text: str,
    ) -> tuple[list[int], list[dict]]:
        lex_counts: list[int] = []
        running = 0
        for row in words:
            running += len(lexical(row["text"]))
            lex_counts.append(running)

        word_meta: list[dict] = []
        for idx, count in enumerate(lex_counts):
            lex_end, char_end = _lexical_span_end(text, count)
            inter_punct = text[lex_end:char_end]
            if idx + 1 < len(words):
                gap_ms = max(0, words[idx + 1]["start_ms"] - words[idx]["end_ms"])
                score, should_cut, sentence_type = _score_qwen_word_boundary(
                    words[idx]["text"],
                    words[idx + 1]["text"],
                    inter_punct,
                    gap_ms,
                    self.config,
                )
                protected = _continues_phrase(
                    words[idx]["text"], words[idx + 1]["text"],
                    text[char_end:], inter_punct,
                )
                if protected:
                    score, should_cut = 0.0, False
            else:
                gap_ms = 0
                score = 1.0
                should_cut = True
                protected = False
                sentence_type = _infer_sentence_type(words[idx]["text"], inter_punct)
            word_meta.append({
                "char_end": len(text) if idx == len(words) - 1 else char_end,
                "gap_ms": gap_ms,
                "score": score,
                "should_cut": should_cut,
                "protected_continuation": protected,
                "sentence_type": sentence_type,
            })

        min_dur = self.config.min_sentence_duration_ms
        max_dur = self.config.max_sentence_duration_ms
        candidate_ends: list[int] = []
        seg_start = 0
        char_cursor = 0
        for idx in range(len(words) - 1):
            meta = word_meta[idx]
            if not meta["should_cut"]:
                continue
            seg_dur = words[idx]["end_ms"] - words[seg_start]["start_ms"]
            rem_dur = words[-1]["end_ms"] - words[idx + 1]["start_ms"]
            if seg_dur < min_dur or rem_dur < min_dur:
                continue
            seg_text = text[char_cursor:meta["char_end"]]
            rem_text = text[meta["char_end"]:]
            # Avoid chopping off a brief initial or trailing breath (< 900ms) as
            # its own tiny segment unless there is a long pause.
            if (
                _is_pure_vocalization(seg_text, self.config.vocalization_chars)
                and (seg_dur < 900 or meta["gap_ms"] < 320)
            ):
                continue
            if (
                _is_pure_vocalization(rem_text, self.config.vocalization_chars)
                and (rem_dur < 900 or meta["gap_ms"] < 320)
            ):
                continue
            candidate_ends.append(idx)
            seg_start = idx + 1
            char_cursor = meta["char_end"]
        candidate_ends.append(len(words) - 1)

        # Prefer the duration limit, but never force a cut through an attached
        # suffix or through adjacent word pieces without an acoustic boundary.
        final_ends: list[int] = []
        seg_start = 0
        for seg_end in candidate_ends:
            queue = [(seg_start, seg_end)]
            while queue:
                s_idx, e_idx = queue.pop(0)
                dur = words[e_idx]["end_ms"] - words[s_idx]["start_ms"]
                if dur <= max_dur or s_idx >= e_idx:
                    final_ends.append(e_idx)
                    continue
                best_cut = None
                best_key = None
                for mid in range(s_idx, e_idx):
                    meta = word_meta[mid]
                    if (meta["protected_continuation"]
                            or (not meta["should_cut"] and meta["gap_ms"] < 160)):
                        continue
                    left_dur = words[mid]["end_ms"] - words[s_idx]["start_ms"]
                    right_dur = words[e_idx]["end_ms"] - words[mid + 1]["start_ms"]
                    if left_dur < min_dur or right_dur < min_dur:
                        continue
                    key = (
                        word_meta[mid]["score"],
                        word_meta[mid]["gap_ms"],
                        -abs(left_dur - dur / 2),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_cut = mid
                if best_cut is None:
                    final_ends.append(e_idx)
                else:
                    queue.insert(0, (best_cut + 1, e_idx))
                    queue.insert(0, (s_idx, best_cut))
            seg_start = seg_end + 1

        # If segments exceed max_sentence_count, merge the weakest internal cuts instead of failing.
        while len(final_ends) > self.config.max_sentence_count and len(final_ends) > 1:
            weakest_pos = min(
                range(len(final_ends) - 1),
                key=lambda pos: (
                    word_meta[final_ends[pos]]["score"],
                    word_meta[final_ends[pos]]["gap_ms"],
                ),
            )
            final_ends.pop(weakest_pos)

        return final_ends, word_meta

