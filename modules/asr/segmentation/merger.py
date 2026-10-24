"""Japanese ASR chunk stitching and overlap deduplication.

Designed for:
    - chunk duration: approximately 20~25 seconds
    - fixed temporal overlap: approximately 4 seconds
    - Japanese ASR
    - Kanji / Hiragana / Katakana inconsistency
    - minor ASR recognition errors
    - whisper / ASMR speech where utterance length varies considerably

Core strategy:

    Chunk A
    ├──────────────────────────────┤
                             ┌─────┐
                             │ 4s  │  <- overlap
                             └─────┘
                                   ├──────────────────────────────┤
                                   Chunk B

Only the temporal overlap region is used for fuzzy matching.

The algorithm prefers preserving text over aggressive deletion.
When the overlap cannot be established with sufficient confidence,
both pieces are retained rather than silently losing speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

import pykakasi

from ..models import ASRSegment, AudioChunk, ASRResult
from .japanese_boundary import normalize_japanese_text


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_OVERLAP_SECONDS = 4.0

# Search slightly outside the nominal 4-second overlap because ASR timestamps
# are not exact.
DEFAULT_TIME_MARGIN = 1.0

# Maximum amount of text considered around the temporal overlap.
DEFAULT_MAX_MATCH_CHARS = 80

# Minimum amount of meaningful Japanese text for fuzzy matching.
DEFAULT_MIN_OVERLAP_CHARS = 2


# ============================================================================
# Kakasi
# ============================================================================

kks = pykakasi.kakasi()


# ============================================================================
# Text normalization
# ============================================================================

def remove_japanese_punc(text: str) -> str:
    """Remove punctuation, symbols and whitespace.

    This is used ONLY for matching / stitching.

    We intentionally use Unicode categories instead of blindly deleting
    complete Unicode ranges such as U+3000-U+303F.
    """

    result: list[str] = []

    for ch in text:
        category = unicodedata.category(ch)

        # Punctuation: P*
        if category.startswith("P"):
            continue

        # Symbols: S*
        if category.startswith("S"):
            continue

        # Separators: Z*
        if category.startswith("Z"):
            continue

        if ch.isspace():
            continue

        result.append(ch)

    return "".join(result)


def normalize_for_match(text: str) -> str:
    """Normalize Japanese text for comparison."""

    if not text:
        return ""

    # Existing project-specific normalization.
    text = normalize_japanese_text(text)

    # Unicode compatibility normalization.
    text = unicodedata.normalize("NFKC", text)

    # Remove punctuation only for matching.
    text = remove_japanese_punc(text)

    return text.lower()


def to_romaji(text: str) -> str:
    """Convert Japanese text to normalized Hepburn romaji."""

    text = normalize_for_match(text)

    if not text:
        return ""

    result = kks.convert(text)

    return "".join(
        item["hepburn"]
        for item in result
        if item.get("hepburn")
    ).lower()


# ============================================================================
# Basic similarity
# ============================================================================

def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(
        None,
        a,
        b,
        autojunk=False,
    ).ratio()


def _bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return set()

    return {
        text[i:i + 2]
        for i in range(len(text) - 1)
    }


def _jaccard_similarity(a: str, b: str) -> float:
    ga = _bigrams(a)
    gb = _bigrams(b)

    if not ga or not gb:
        return 0.0

    return len(ga & gb) / len(ga | gb)


def _containment_similarity(a: str, b: str) -> float:
    """Measure how much of the shorter text is contained by the longer."""

    if not a or not b:
        return 0.0

    shorter, longer = (
        (a, b)
        if len(a) <= len(b)
        else (b, a)
    )

    if shorter in longer:
        return len(shorter) / len(longer)

    return 0.0


def _text_similarity(a: str, b: str) -> float:
    """Character-level Japanese similarity."""

    a = normalize_for_match(a)
    b = normalize_for_match(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    seq = _sequence_ratio(a, b)
    jaccard = _jaccard_similarity(a, b)
    containment = _containment_similarity(a, b)

    return max(
        seq,
        jaccard,
        containment,
    )


def _phonetic_similarity(a: str, b: str) -> float:
    """Japanese pronunciation-level similarity."""

    ra = to_romaji(a)
    rb = to_romaji(b)

    if not ra or not rb:
        return 0.0

    return _sequence_ratio(
        ra,
        rb,
    )


def _combined_similarity(a: str, b: str) -> float:
    """Combined text + phonetic similarity.

    Japanese ASR may output:

        天気がいい

    and:

        てんきがいい

    Therefore phonetic similarity gets slightly more weight.
    """

    text_score = _text_similarity(a, b)
    phonetic_score = _phonetic_similarity(a, b)

    return (
        text_score * 0.35
        + phonetic_score * 0.65
    )


# ============================================================================
# Absolute time conversion
# ============================================================================

def to_absolute(
    chunk: AudioChunk,
    result: ASRResult,
) -> list[ASRSegment]:
    """Convert chunk-relative ASR timestamps to absolute timestamps."""

    return [
        ASRSegment(
            text=s.text,
            start=chunk.start + s.start,
            end=chunk.start + s.end,
        )
        for s in result.segments
    ]


# ============================================================================
# Segment normalization
# ============================================================================

def normalize_segments(
    segments: list[ASRSegment],
) -> list[ASRSegment]:
    """Normalize and sort ASR segments."""

    result: list[ASRSegment] = []

    for segment in segments:
        text = normalize_japanese_text(segment.text)

        if not text:
            continue

        if segment.end <= segment.start:
            continue

        result.append(
            ASRSegment(
                text=text,
                start=segment.start,
                end=segment.end,
            )
        )

    result.sort(
        key=lambda x: (
            x.start,
            x.end,
        )
    )

    return result


# ============================================================================
# Time utilities
# ============================================================================

def _overlap_duration(
    a: ASRSegment,
    b: ASRSegment,
) -> float:
    """Return temporal intersection in seconds."""

    return max(
        0.0,
        min(a.end, b.end)
        - max(a.start, b.start),
    )


def _duration(segment: ASRSegment) -> float:
    return max(
        0.0,
        segment.end - segment.start,
    )


# ============================================================================
# Text extraction by time
# ============================================================================

def _segments_in_time_range(
    segments: list[ASRSegment],
    start: float,
    end: float,
) -> list[ASRSegment]:
    """Get segments that intersect a time interval."""

    result: list[ASRSegment] = []

    for segment in segments:
        if segment.end <= start:
            continue

        if segment.start >= end:
            break

        if _overlap_duration(
            segment,
            ASRSegment(
                text="",
                start=start,
                end=end,
            ),
        ) > 0:
            result.append(segment)

    return result


def _join_segment_text(
    segments: list[ASRSegment],
) -> str:
    """Join ASR segment text for matching."""

    return "".join(
        normalize_for_match(segment.text)
        for segment in segments
    )


# ============================================================================
# Chunk overlap candidate
# ============================================================================

@dataclass(frozen=True)
class OverlapCandidate:
    """Best textual overlap candidate."""

    left_chars: int
    right_chars: int

    text_score: float
    phonetic_score: float
    combined_score: float

    length_score: float
    final_score: float

    left_text: str
    right_text: str


# ============================================================================
# Fuzzy suffix / prefix matching
# ============================================================================

def _find_best_suffix_prefix_overlap(
    left: str,
    right: str,
    max_chars: int = DEFAULT_MAX_MATCH_CHARS,
    min_chars: int = DEFAULT_MIN_OVERLAP_CHARS,
) -> OverlapCandidate | None:
    """Find fuzzy suffix/prefix overlap.

    IMPORTANT:

    left_chars != right_chars is allowed.

    Example:

        left:
            天気が良いので

        right:
            天気がいいので外に出ました

    Matching overlap:

        left  = 天気が良いので
        right = 天気がいいので

    The two representations may have different character counts.
    """

    left = normalize_for_match(left)
    right = normalize_for_match(right)

    if not left or not right:
        return None

    left = left[-max_chars:]
    right = right[:max_chars]

    best: OverlapCandidate | None = None

    max_left = len(left)
    max_right = len(right)

    for left_len in range(
        min_chars,
        max_left + 1,
    ):
        left_part = left[-left_len:]

        # Japanese ASR insertion / deletion is usually small.
        #
        # Allow a length difference of up to 6 characters.
        right_min = max(
            min_chars,
            left_len - 6,
        )

        right_max = min(
            max_right,
            left_len + 6,
        )

        for right_len in range(
            right_min,
            right_max + 1,
        ):
            right_part = right[:right_len]

            text_score = _text_similarity(
                left_part,
                right_part,
            )

            phonetic_score = _phonetic_similarity(
                left_part,
                right_part,
            )

            combined_score = (
                text_score * 0.35
                + phonetic_score * 0.65
            )

            # Longer overlap is more convincing.
            length_score = min(
                1.0,
                min(left_len, right_len) / 10.0,
            )

            final_score = (
                combined_score * 0.85
                + length_score * 0.15
            )

            candidate = OverlapCandidate(
                left_chars=left_len,
                right_chars=right_len,
                text_score=text_score,
                phonetic_score=phonetic_score,
                combined_score=combined_score,
                length_score=length_score,
                final_score=final_score,
                left_text=left_part,
                right_text=right_part,
            )

            if (
                best is None
                or candidate.final_score > best.final_score
            ):
                best = candidate

    return best


# ============================================================================
# Candidate validation
# ============================================================================

def _is_reliable_overlap(
    candidate: OverlapCandidate,
) -> bool:
    """Conservative validation of an overlap candidate."""

    overlap_len = min(
        candidate.left_chars,
        candidate.right_chars,
    )

    # Very short Japanese utterances are dangerous.
    #
    # "はい"
    # "ね"
    # "あ"
    #
    # should not be aggressively deduplicated.
    if overlap_len <= 2:
        return (
            candidate.phonetic_score >= 0.95
            and candidate.text_score >= 0.90
        )

    if overlap_len <= 4:
        return (
            candidate.phonetic_score >= 0.88
            and candidate.combined_score >= 0.84
        )

    if overlap_len <= 7:
        return (
            candidate.phonetic_score >= 0.80
            and candidate.combined_score >= 0.76
        )

    return (
        candidate.phonetic_score >= 0.72
        and candidate.combined_score >= 0.70
    )


# ============================================================================
# Stitch two texts
# ============================================================================

def stitch_japanese_asr_overlap(
    text1: str,
    text2: str,
    max_match_chars: int = DEFAULT_MAX_MATCH_CHARS,
) -> str:
    """Stitch two Japanese ASR texts.

    This function does not know the timestamps.
    It should therefore only be used after the caller has established that
    the two pieces actually belong to an overlap region.
    """

    clean1 = normalize_for_match(text1)
    clean2 = normalize_for_match(text2)

    if not clean1:
        return clean2

    if not clean2:
        return clean1

    if clean1 == clean2:
        return clean1

    # Complete containment.
    if clean1 in clean2:
        return clean2

    if clean2 in clean1:
        return clean1

    candidate = _find_best_suffix_prefix_overlap(
        clean1,
        clean2,
        max_chars=max_match_chars,
    )

    if candidate is None:
        return clean1 + clean2

    if not _is_reliable_overlap(candidate):
        # IMPORTANT:
        # Never discard text if the overlap is uncertain.
        return clean1 + clean2

    # Remove the duplicate suffix from the first chunk.
    left_prefix = clean1[
        :-candidate.left_chars
    ]

    return left_prefix + clean2


# ============================================================================
# Determine chunk overlap region
# ============================================================================

def _build_chunk_overlap_text(
    segments: list[ASRSegment],
    overlap_start: float,
    overlap_end: float,
) -> str:
    """Extract ASR text inside the expected chunk overlap."""

    overlap_segments = _segments_in_time_range(
        segments,
        overlap_start,
        overlap_end,
    )

    return _join_segment_text(
        overlap_segments,
    )


# ============================================================================
# Main chunk-level stitching
# ============================================================================

def stitch_chunk_pair(
    chunk1_segments: list[ASRSegment],
    chunk2_segments: list[ASRSegment],
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    time_margin: float = DEFAULT_TIME_MARGIN,
) -> list[ASRSegment]:
    """Stitch two adjacent chunks using their known temporal overlap.

    Example:

        Chunk 1: 0 ~ 24s
        Chunk 2: 20 ~ 44s

    Expected overlap:

        20 ~ 24s

    The function:

        1. determines the expected overlap
        2. extracts ASR text around that region
        3. finds the best fuzzy suffix/prefix match
        4. merges duplicate text
        5. preserves all non-overlapping segments

    NOTE:
    This function assumes chunk1 and chunk2 are adjacent chunks.
    """

    left = normalize_segments(
        chunk1_segments,
    )

    right = normalize_segments(
        chunk2_segments,
    )

    if not left:
        return right

    if not right:
        return left

    chunk1_end = max(
        segment.end
        for segment in left
    )

    chunk2_start = min(
        segment.start
        for segment in right
    )

    # Expected temporal overlap.
    expected_start = max(
        chunk2_start,
        chunk1_end - overlap_seconds,
    )

    expected_end = min(
        chunk1_end,
        chunk2_start + overlap_seconds,
    )

    # Add a small safety margin because ASR timestamps are not exact.
    search_start = max(
        chunk2_start,
        expected_start - time_margin,
    )

    search_end = min(
        chunk1_end,
        expected_end + time_margin,
    )

    left_overlap_text = _build_chunk_overlap_text(
        left,
        search_start,
        search_end,
    )

    right_overlap_text = _build_chunk_overlap_text(
        right,
        search_start,
        search_end,
    )

    # ------------------------------------------------------------------
    # Try to identify the textual overlap.
    # ------------------------------------------------------------------

    candidate = _find_best_suffix_prefix_overlap(
        left_overlap_text,
        right_overlap_text,
        max_chars=DEFAULT_MAX_MATCH_CHARS,
    )

    # If no reliable textual overlap exists, preserve both chunks.
    if (
        candidate is None
        or not _is_reliable_overlap(candidate)
    ):
        return _merge_non_duplicate_segments(
            left,
            right,
        )

    # ------------------------------------------------------------------
    # Determine how much text from chunk2 belongs to the duplicate area.
    #
    # We don't delete arbitrary ASR segments based purely on their text.
    # Instead, only segments whose time lies inside the known overlap are
    # considered.
    # ------------------------------------------------------------------

    overlap_segments_right = _segments_in_time_range(
        right,
        search_start,
        search_end,
    )

    if not overlap_segments_right:
        return _merge_non_duplicate_segments(
            left,
            right,
        )

    # Identify the first right-side segment that participates in the
    # textual overlap.
    #
    # We use text matching rather than assuming that segment boundaries
    # exactly match the 4-second boundary.
    consumed_right_count = 0

    accumulated_right_text = ""

    for index, segment in enumerate(
        overlap_segments_right
    ):
        accumulated_right_text += normalize_for_match(
            segment.text
        )

        accumulated_candidate = _find_best_suffix_prefix_overlap(
            left_overlap_text,
            accumulated_right_text,
            max_chars=DEFAULT_MAX_MATCH_CHARS,
        )

        if (
            accumulated_candidate
            and _is_reliable_overlap(
                accumulated_candidate
            )
        ):
            consumed_right_count = index + 1

    # ------------------------------------------------------------------
    # If we could not confidently map the text to right-side segments,
    # fall back to conservative segment-level deduplication.
    # ------------------------------------------------------------------

    if consumed_right_count <= 0:
        return _merge_non_duplicate_segments(
            left,
            right,
        )

    duplicate_right_segments = (
        overlap_segments_right[
            :consumed_right_count
        ]
    )

    duplicate_ids = {
        id(segment)
        for segment in duplicate_right_segments
    }

    result = list(left)

    # ------------------------------------------------------------------
    # Keep right-side segments that are not part of the duplicate overlap.
    # ------------------------------------------------------------------

    for segment in right:
        if id(segment) in duplicate_ids:
            continue

        result.append(segment)

    result.sort(
        key=lambda x: (
            x.start,
            x.end,
        )
    )

    return normalize_segments(result)


# ============================================================================
# Conservative segment-level fallback
# ============================================================================

def _merge_non_duplicate_segments(
    left: list[ASRSegment],
    right: list[ASRSegment],
) -> list[ASRSegment]:
    """Fallback when chunk-level overlap cannot be confidently matched."""

    result = list(left)

    for current in right:

        duplicate = False

        # Only inspect nearby previous segments.
        for previous in reversed(result):

            if previous.end <= current.start:
                break

            overlap = _overlap_duration(
                previous,
                current,
            )

            if overlap <= 0:
                continue

            prev_text = normalize_for_match(
                previous.text
            )

            curr_text = normalize_for_match(
                current.text
            )

            if not prev_text or not curr_text:
                continue

            if prev_text == curr_text:
                duplicate = True
                break

            similarity = _combined_similarity(
                prev_text,
                curr_text,
            )

            # Conservative fallback threshold.
            if similarity >= 0.90:
                duplicate = True
                break

        if not duplicate:
            result.append(current)

    result.sort(
        key=lambda x: (
            x.start,
            x.end,
        )
    )

    return normalize_segments(result)


# ============================================================================
# Multiple chunk stitching
# ============================================================================

def stitch_chunks(
    chunks: list[AudioChunk],
    results: list[ASRResult],
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[ASRSegment]:
    """Convert multiple ASR chunk results into one absolute ASR stream.

    ``chunks[i]`` must correspond to ``results[i]``.

    Example:

        chunks:
            0 ~ 24
            20 ~ 44
            40 ~ 64
            60 ~ 84

    overlap:
            4s
    """

    if not chunks or not results:
        return []

    if len(chunks) != len(results):
        raise ValueError(
            "chunks and results must have the same length"
        )

    # Convert all chunks to absolute time.
    absolute_chunks = [
        normalize_segments(
            to_absolute(
                chunk,
                result,
            )
        )
        for chunk, result in zip(
            chunks,
            results,
        )
    ]

    if not absolute_chunks:
        return []

    merged = absolute_chunks[0]

    for current in absolute_chunks[1:]:
        merged = stitch_chunk_pair(
            merged,
            current,
            overlap_seconds=overlap_seconds,
        )

    return normalize_segments(merged)


# ============================================================================
# Generic dedupe for already-absolute segments
# ============================================================================

def dedupe_overlaps(
    segments: list[ASRSegment],
    overlap_ratio: float = 0.25,
) -> list[ASRSegment]:
    """Conservative cleanup for already absolute ASR segments.

    This is a fallback utility, not the primary chunk stitching algorithm.

    Prefer ``stitch_chunks()`` when chunk boundaries are available.
    """

    segs = normalize_segments(
        segments
    )

    if not segs:
        return []

    merged: list[ASRSegment] = []

    for current in segs:

        if not merged:
            merged.append(current)
            continue

        duplicate = False

        # Look backwards through only time-overlapping segments.
        for previous in reversed(merged):

            if previous.end <= current.start:
                break

            overlap = _overlap_duration(
                previous,
                current,
            )

            if overlap <= 0:
                continue

            prev_duration = _duration(
                previous
            )

            curr_duration = _duration(
                current
            )

            temporal_ratio = overlap / max(
                1e-6,
                min(
                    prev_duration,
                    curr_duration,
                ),
            )

            prev_text = normalize_for_match(
                previous.text
            )

            curr_text = normalize_for_match(
                current.text
            )

            if prev_text == curr_text:
                duplicate = True

                # Preserve temporal union.
                previous.end = max(
                    previous.end,
                    current.end,
                )

                break

            if temporal_ratio < overlap_ratio:
                continue

            # Complementary observations of one utterance often share only a
            # suffix/prefix (e.g. 「今日は天気」 + 「天気がいい」).  Whole-string
            # similarity is intentionally low in that case, so stitch the
            # reliable textual overlap before applying the strict fuzzy-
            # duplicate fallback below.  The ASR segment covers the union of
            # both observations, not the union of their parent audio chunks.
            candidate = _find_best_suffix_prefix_overlap(
                prev_text,
                curr_text,
                max_chars=DEFAULT_MAX_MATCH_CHARS,
            )
            if candidate is not None and _is_reliable_overlap(candidate):
                previous.text = stitch_japanese_asr_overlap(
                    previous.text,
                    current.text,
                )
                previous.end = max(previous.end, current.end)
                duplicate = True
                break

            similarity = _combined_similarity(
                prev_text,
                curr_text,
            )

            # Generic dedupe is deliberately strict.
            if similarity >= 0.90:
                duplicate = True

                previous.end = max(
                    previous.end,
                    current.end,
                )

                # Keep the more complete text only if containment
                # clearly indicates that one is a superset.
                if (
                    curr_text in prev_text
                    and len(prev_text) >= len(curr_text)
                ):
                    pass
                elif (
                    prev_text in curr_text
                    and len(curr_text) > len(prev_text)
                ):
                    previous.text = current.text

                break

        if not duplicate:
            merged.append(current)

    return normalize_segments(merged)
