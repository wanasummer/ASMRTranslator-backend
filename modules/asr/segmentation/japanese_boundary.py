"""Japanese grammatical boundary heuristics.

Pure functions used by the sentence segmenter to decide whether two ASR
fragments belong to the same utterance. Japanese sentence structure differs from
English: a clause ending in a particle (は・を・に…) is very likely to continue,
while a clause ending in a predicate (です・ます・だ…) or terminal punctuation is
a strong sentence end. Scores are in ``[0, 1]``: 0 = merge, 1 = cut.

The gap alone only *tends* toward a cut — it never forces one: a sentence-final
``。`` cuts even across a short gap, and a particle-final clause keeps merging
even across a long one.
"""

# Complete utterances that lexically end in a particle-like char (greetings,
# closings) — they are sentence-final, not dangling particles.
_COMPLETE_EXCEPTIONS = (
    "こんにちは", "こんばんは", "それでは", "ではでは", "こちらこそ",
    "おやすみなさい", "ありがとうございます", "ありがとう", "さようなら",
    "おはようございます", "はじめまして", "どういたしまして",
)

# Particles that usually continue a sentence (sorted longest-first so
# compound particles match before their single-char tails).
_PARTICLE_ENDINGS = (
    "には", "では", "とは", "へは",
    "から", "まで", "より", "ほど",
    "は", "が", "を", "に", "へ", "で", "と", "の",
)

# Predicates / copulas that signal a completed clause.
_STRONG_ENDINGS = (
    "でした", "でしょう", "ません", "ましょう",
    "じゃない", "じゃありません", "ではありません",
    "ですね", "ですよ", "ますね",
    "だった", "ました", "ます", "です", "だ",
    "ね", "よ",
)

# Clause-initial conjunctions that usually continue from a previous clause.
_CONJUNCTIONS = (
    "それから", "それでは", "それじゃ", "だけど", "けれど",
    "それで", "だから", "なので", "そして", "けど", "でも",
    "あと", "つぎに", "そのあと",
)

_FULL_HALF = {
    "。": ".",
    "．": ".",
    "｡": ".",
    "？": "?",
    "！": "!",
    "　": " ",
    "“": '"',
    "”": '"',
}


def normalize_japanese_text(text: str) -> str:
    """Unify full/half-width terminal punctuation and whitespace."""
    return "".join(_FULL_HALF.get(ch, ch) for ch in text).strip()


def _tail_text(text: str) -> str:
    """For tail checks: normalize and drop any terminal punctuation/space."""
    return normalize_japanese_text(text).strip(".!?、, ")


def _end_text(text: str) -> str:
    """For final-punctuation checks: normalize, keep the terminal char."""
    return normalize_japanese_text(text).rstrip("、, ")


def ends_with_particle(text: str) -> bool:
    """True when the clause ends in a case/postpositional particle.

    A particle-final clause is incomplete — more of the sentence usually follows.
    """
    t = _tail_text(text)
    if not t:
        return False
    for exc in _COMPLETE_EXCEPTIONS:
        if t.endswith(exc):
            return False
    for p in _PARTICLE_ENDINGS:
        if t.endswith(p):
            return True
    return False


def has_sentence_final_punctuation(text: str) -> bool:
    """True when the clause ends with ``。！？`` (any width)."""
    return _end_text(text).endswith((".", "!", "?"))


def has_strong_ending(text: str) -> bool:
    """True when the clause ends in a copula / polite predicate / final particle."""
    t = _tail_text(text)
    if not t:
        return False
    for e in _STRONG_ENDINGS:
        if t.endswith(e):
            return True
    return False


def starts_with_conjunction(text: str) -> bool:
    """True when the clause begins with a conjunction that continues a sentence."""
    t = normalize_japanese_text(text).lstrip("、, ")
    for c in _CONJUNCTIONS:
        if t.startswith(c):
            return True
    return False


def calculate_boundary_score(
    cur_text: str,
    next_text: str,
    gap_ms: float,
    accum_ms: float = None,
    merge_gap_ms: float = 500,
    candidate_gap_ms: float = 700,
    strong_cut_gap_ms: float = 1200,
    soft_max_sec: float = 12,
) -> float:
    """Score whether a fragment boundary is a sentence cut (0 = merge, 1 = cut).

    The gap contributes a base score that ramps gently: 0 up to ``merge_gap_ms``,
    then to 0.6 at ``candidate_gap_ms``, then to 1.0 at ``strong_cut_gap_ms``.
    Syntax modulates it: a terminal ``。`` (+0.45) or strong predicate ending
    (+0.35) pushes toward a cut — hard enough that a period cuts even across a
    short gap — while a particle ending (−0.3) or a following conjunction (−0.2)
    pulls toward a merge. ``accum_ms`` (running sentence length) adds pressure as
    it approaches ``soft_max_sec`` so long utterances still find a natural
    boundary instead of waiting for the hard ceiling.
    """
    if gap_ms < merge_gap_ms:
        base = 0.0
    elif gap_ms <= candidate_gap_ms:
        base = 0.6 * (gap_ms - merge_gap_ms) / max(1.0, candidate_gap_ms - merge_gap_ms)
    else:
        base = 0.6 + 0.4 * min(
            1.0, (gap_ms - candidate_gap_ms) / max(1.0, strong_cut_gap_ms - candidate_gap_ms)
        )

    if has_sentence_final_punctuation(cur_text):
        base += 0.45
    if has_strong_ending(cur_text):
        base += 0.35
    if ends_with_particle(cur_text):
        base -= 0.3
    if starts_with_conjunction(next_text):
        base -= 0.2

    if accum_ms is not None and soft_max_sec > 0:
        soft_ms = soft_max_sec * 1000.0
        ramp_lo = soft_ms * 0.75
        ramp_hi = max(ramp_lo + 1.0, soft_ms)
        pressure = min(1.0, max(0.0, (accum_ms - ramp_lo) / (ramp_hi - ramp_lo)))
        base += pressure * 0.3

    return max(0.0, min(1.0, base))
