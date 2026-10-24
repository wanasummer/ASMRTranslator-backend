"""Sentence-level segmentation pipeline for the ASR path.

extraction: ffmpeg range extraction (single chunk + parallel slices).
japanese_boundary: Japanese grammatical boundary heuristics.
merger: chunk-relative → absolute, overlap dedup.
sentence_segmenter: merge ASR fragments into sentence-level utterances.
"""

from .extraction import extract_chunk, extract_ranges
from .japanese_boundary import (
    calculate_boundary_score,
    ends_with_particle,
    has_sentence_final_punctuation,
    has_strong_ending,
    normalize_japanese_text,
    starts_with_conjunction,
)
from .merger import dedupe_overlaps, normalize_segments, to_absolute
from .sentence_segmenter import merge_into_sentences

__all__ = [
    "extract_chunk",
    "extract_ranges",
    "calculate_boundary_score",
    "ends_with_particle",
    "has_sentence_final_punctuation",
    "has_strong_ending",
    "normalize_japanese_text",
    "starts_with_conjunction",
    "dedupe_overlaps",
    "normalize_segments",
    "to_absolute",
    "merge_into_sentences",
]
