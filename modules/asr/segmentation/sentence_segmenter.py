"""Merge deduplicated ASR fragments into sentence-level utterances.

``merge_into_sentences`` walks the fragments in time order and decides at each
fragment boundary whether the accumulated utterance ends there. The decision uses
``calculate_boundary_score`` with hard merge/cut gap rules plus syntactic
modulation. Overlapping fragments (the same speech re-sliced by ASR, gap ≤ 0) are
always absorbed so a trailing particle never splits into its own block. A clear
sentence end (terminal 。！？ or a strong です/ます/ね… ending) cuts whenever the gap
makes the boundary at least ambiguous — it never defers to the next boundary —
while weak endings defer to the next boundary's context. A sentence never exceeds
``hard_max_sec`` (cut before a fragment would push it past), and ``soft_max_sec``
pressure steers long utterances toward a natural boundary. Fragments are atomic —
cuts always land between words at a real speech gap, never mid-fragment. Short
standalone utterances (はい, ええ) separated by a large gap survive because the
hard gap rule cuts them; the ``min_sec`` floor only gates ambiguous soft cuts so
no sub-second fragments are emitted.
"""

from typing import Optional

from ..models import ASRSegment, SentenceSegment
from .japanese_boundary import (
    calculate_boundary_score,
    has_sentence_final_punctuation,
    has_strong_ending,
    normalize_japanese_text,
)


def _decide(score: float, next_score: Optional[float], strong_signal: bool) -> str:
    """Map a boundary score to merge/cut; ambiguous middle scores defer.

    ``strong_signal`` is true when the current clause already ends in a clear
    sentence end (terminal 。！？ or a strong predicate / final particle). Such a
    boundary cuts once the gap makes it at least ambiguous — it never defers to
    the next boundary. Without that, run-ons like 「…知ってるみたいですね」+
    「…好きで」 stay merged because the next boundary is a weak particle ending.
    """
    if score < 0.3:
        return "merge"
    if score > 0.7:
        return "cut"
    if strong_signal:
        return "cut"
    if next_score is not None:
        if next_score > 0.7:
            return "cut"
        if next_score < 0.3:
            return "merge"
    return "merge"


def _finalize(sent: SentenceSegment, idx: int) -> SentenceSegment:
    sent.id = f"s{idx}"
    return sent


def merge_into_sentences(
    segs: list[ASRSegment],
    boundary_cfg: Optional[dict] = None,
    sentence_cfg: Optional[dict] = None,
) -> list[SentenceSegment]:
    """Merge a deduplicated fragment stream into sentence-level segments.

    ``boundary_cfg`` keys (all optional): merge_gap_ms, candidate_gap_ms,
    strong_cut_gap_ms. ``sentence_cfg`` keys: min_sec, soft_max_sec,
    hard_max_sec.
    """
    bc = {
        "merge_gap_ms": 500,
        "candidate_gap_ms": 700,
        "strong_cut_gap_ms": 1200,
        **(boundary_cfg or {}),
    }
    sc = {
        "min_sec": 0.4,
        "soft_max_sec": 12,
        "hard_max_sec": 15,
        **(sentence_cfg or {}),
    }

    ordered = sorted(segs, key=lambda s: s.start)
    if not ordered:
        return []

    min_ms = sc["min_sec"] * 1000.0
    soft_ms = sc["soft_max_sec"] * 1000.0
    hard_ms = sc["hard_max_sec"] * 1000.0
    strong_cut_ms = bc["strong_cut_gap_ms"]

    sentences: list[SentenceSegment] = []
    cur = SentenceSegment(
        id="", start=ordered[0].start, end=ordered[0].end,
        text=normalize_japanese_text(ordered[0].text),
    )
    n = len(ordered)

    for i in range(1, n):
        gap = (ordered[i].start - ordered[i - 1].end) * 1000.0
        t = normalize_japanese_text(ordered[i].text)

        # Overlapping fragment — the same speech re-sliced by ASR (the fragment
        # starts before the previous one ends). Absorb it regardless of scoring
        # so a trailing particle (ね/か/だ) is never split into its own block;
        # the suffix check keeps an already-present tail from being duplicated.
        if gap <= 0:
            if (ordered[i].end - cur.start) * 1000.0 > hard_ms:
                sentences.append(_finalize(cur, len(sentences)))
                cur = SentenceSegment(
                    id="", start=ordered[i].start, end=ordered[i].end, text=t
                )
            else:
                if t and not cur.text.endswith(t):
                    cur.text += t
                cur.end = max(cur.end, ordered[i].end)
            continue

        accum_ms = (ordered[i - 1].end - cur.start) * 1000.0

        # Hard ceiling: never merge a fragment that would push past hard_max_sec.
        if (ordered[i].end - cur.start) * 1000.0 > hard_ms:
            decision = "cut"
        else:
            score = calculate_boundary_score(
                cur.text, ordered[i].text, gap, accum_ms,
                merge_gap_ms=bc["merge_gap_ms"],
                candidate_gap_ms=bc["candidate_gap_ms"],
                strong_cut_gap_ms=strong_cut_ms,
                soft_max_sec=sc["soft_max_sec"],
            )
            next_score = None
            if i + 1 < n:
                gap2 = (ordered[i + 1].start - ordered[i].end) * 1000.0
                next_score = calculate_boundary_score(
                    ordered[i].text, ordered[i + 1].text, gap2, None,
                    merge_gap_ms=bc["merge_gap_ms"],
                    candidate_gap_ms=bc["candidate_gap_ms"],
                    strong_cut_gap_ms=strong_cut_ms,
                )
            decision = _decide(
                score,
                next_score,
                has_sentence_final_punctuation(cur.text) or has_strong_ending(cur.text),
            )

            # Sub-second utterances only survive as their own sentence when the
            # gap itself forces a cut (short standalone はい / ええ). Otherwise an
            # ambiguous short cut is merged so we don't emit noise fragments.
            if decision == "cut" and accum_ms < min_ms and gap < strong_cut_ms:
                decision = "merge"

        if decision == "cut":
            sentences.append(_finalize(cur, len(sentences)))
            cur = SentenceSegment(
                id="", start=ordered[i].start, end=ordered[i].end, text=t
            )
        else:
            cur.end = max(cur.end, ordered[i].end)
            cur.text += t

    sentences.append(_finalize(cur, len(sentences)))
    return sentences
