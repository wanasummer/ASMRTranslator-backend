"""Tests for modules.asr.segmentation — boundary scoring, overlap dedup,
and sentence merging.

Run from the project root:  python -m unittest tests.test_asr_segmentation
Fully offline (no audio files touched).
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.asr.models import ASRSegment, AudioChunk, ASRResult
from modules.asr.segmentation.japanese_boundary import (
    calculate_boundary_score,
    ends_with_particle,
    has_sentence_final_punctuation,
    has_strong_ending,
    normalize_japanese_text,
    starts_with_conjunction,
)
from modules.asr.segmentation.merger import dedupe_overlaps, normalize_segments, to_absolute
from modules.asr.segmentation.sentence_segmenter import merge_into_sentences


def _seg(text, start, end):
    return ASRSegment(text=text, start=start, end=end)


class AbsoluteAndNormalizeTest(unittest.TestCase):
    def test_to_absolute_offsets_chunk_start(self):
        chunk = AudioChunk(id="chunk_0001", start=13.0, end=28.0, path="x.wav")
        result = ASRResult(
            text="こんにちは元気",
            duration=3.0,
            segments=[ASRSegment("こんにちは", 0.0, 1.2), ASRSegment("元気", 1.5, 2.5)],
        )
        out = to_absolute(chunk, result)
        self.assertEqual([(s.start, s.end) for s in out], [(13.0, 14.2), (14.5, 15.5)])

    def test_normalize_segments_drops_empty_and_sorts(self):
        segs = [_seg("元気ですか", 5.0, 6.0), _seg("", 0.0, 1.0), _seg("こんにちは。", 0.5, 1.0)]
        out = normalize_segments(segs)
        self.assertEqual([s.text for s in out], ["こんにちは.", "元気ですか"])

    def test_normalize_japanese_text(self):
        self.assertEqual(normalize_japanese_text("こんにちは。元気？"), "こんにちは.元気?")


class BoundaryHeuristicTest(unittest.TestCase):
    def test_ends_with_particle(self):
        self.assertTrue(ends_with_particle("東京へ"))
        self.assertTrue(ends_with_particle("これから"))
        self.assertFalse(ends_with_particle("行きます"))
        self.assertFalse(ends_with_particle("はい"))

    def test_has_sentence_final_punctuation(self):
        self.assertTrue(has_sentence_final_punctuation("はい。"))
        self.assertTrue(has_sentence_final_punctuation("どうぞ！"))
        self.assertFalse(has_sentence_final_punctuation("はい"))

    def test_has_strong_ending(self):
        self.assertTrue(has_strong_ending("行きます"))
        self.assertTrue(has_strong_ending("お願いします。"))
        self.assertTrue(has_strong_ending("そうですね"))
        self.assertFalse(has_strong_ending("東京へ"))

    def test_starts_with_conjunction(self):
        self.assertTrue(starts_with_conjunction("でも、"))
        self.assertTrue(starts_with_conjunction("それから"))
        self.assertFalse(starts_with_conjunction("それ"))

    def test_gap_under_merge_forces_merge(self):
        self.assertEqual(calculate_boundary_score("はい", "おかえり", 300), 0.0)

    def test_long_gap_scores_high(self):
        self.assertEqual(calculate_boundary_score("はい", "おかえり", 1500), 1.0)

    def test_particle_lowers_score(self):
        low = calculate_boundary_score("東京へ", "行きます", 900)
        neutral = calculate_boundary_score("こんにちは", "さようなら", 900)
        self.assertLess(low, neutral)

    def test_period_raises_score_above_cut(self):
        # Short pause + period → cut.
        self.assertGreater(calculate_boundary_score("行きます。", "どうぞ", 300), 0.7)

    def test_particle_long_gap_stays_below_cut(self):
        # Long pause + particle ending → stays ambiguous (not >0.7).
        self.assertLessEqual(calculate_boundary_score("東京へ", "行きます", 2000), 0.7)

    def test_accum_pressure_raises_score(self):
        base = calculate_boundary_score("はい", "おかえり", 800)
        pressured = calculate_boundary_score("はい", "おかえり", 800, accum_ms=11500)
        self.assertGreater(pressured, base)


class DedupeTest(unittest.TestCase):
    def test_overlapping_similar_merged(self):
        segs = [_seg("こんにちは", 0.0, 10.0), _seg("こんにちは元気", 4.0, 14.0)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].start, 0.0)
        self.assertEqual(out[0].end, 14.0)
        self.assertEqual(out[0].text, "こんにちは元気")

    def test_overlapping_dissimilar_kept(self):
        segs = [_seg("こんにちは", 0.0, 10.0), _seg("ありがとう", 4.0, 14.0)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 2)

    def test_overlapping_suffix_prefix_fragments_are_stitched(self):
        # Two overlapping ASR windows may return complementary halves of the
        # same utterance.  Merge both the text overlap and the time union.
        segs = [_seg("今日は天気", 10.0, 14.0),
                _seg("天気がいい", 12.0, 16.0)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "今日は天気がいい")
        self.assertEqual((out[0].start, out[0].end), (10.0, 16.0))

    def test_no_overlap_kept(self):
        segs = [_seg("こんにちは", 0.0, 5.0), _seg("さようなら", 6.0, 11.0)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 2)

    def test_identical_short_overlap_merged(self):
        # Same short utterance caught in the 2s overlap of two chunks. Identical
        # text with any positive overlap is a cross-chunk duplicate even though
        # the IoU (0.33) is below the 0.5 ratio gate.
        segs = [_seg("はい", 13.0, 13.3), _seg("はい", 13.2, 13.5)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].start, out[0].end), (13.0, 13.5))

    def test_identical_no_overlap_kept(self):
        # Genuine repetition (no time overlap) is never merged.
        segs = [_seg("はい", 0.0, 0.5), _seg("はい", 1.5, 2.0)]
        out = dedupe_overlaps(segs)
        self.assertEqual(len(out), 2)


class MergeSentencesTest(unittest.TestCase):
    def test_one_continuous_sentence(self):
        segs = [_seg("こんにちは", 0.0, 1.0), _seg("元気ですか", 1.0, 2.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "こんにちは元気ですか")
        self.assertEqual((out[0].start, out[0].end), (0.0, 2.0))

    def test_500ms_pause_merges(self):
        segs = [_seg("こんにちは", 0.0, 1.0), _seg("元気ですか", 1.5, 2.5)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)

    def test_two_sentences_cut(self):
        segs = [_seg("こんにちは。", 0.0, 1.0), _seg("元気ですか。", 2.0, 3.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "こんにちは.")
        self.assertEqual(out[1].text, "元気ですか.")

    def test_particle_ending_long_pause_no_cut(self):
        segs = [_seg("東京へ", 0.0, 1.0), _seg("行きます", 3.0, 4.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "東京へ行きます")

    def test_period_short_pause_cuts(self):
        segs = [_seg("行きます。", 0.0, 1.0), _seg("どうぞ", 1.3, 2.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)

    def test_ambiguous_defers_to_next_context(self):
        # A|B gap 800 (ambiguous); B ends with a strong "いですね。" so B|C is a
        # strong cut → A is also cut from B (B starts a new sentence with C).
        segs = [
            _seg("あ", 0.0, 1.0),
            _seg("いですね。", 1.8, 2.8),
            _seg("う", 3.1, 4.0),
        ]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 3)
        self.assertEqual([s.text for s in out], ["あ", "いですね.", "う"])

    def test_ambiguous_alone_merges(self):
        # Only A|B at 800ms (no next context) → default merge.
        segs = [_seg("あ", 0.0, 1.0), _seg("い", 1.8, 2.8)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)

    def test_overlap_trailing_particle_absorbed(self):
        # `か` [47.18-47.42] overlaps `…好きなんです` [45.02-47.42] (same speech
        # re-sliced) → absorbed, not split into its own か block.
        segs = [_seg("耳舐めされるの好きなんです", 45.02, 47.42),
                _seg("か", 47.18, 47.42)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "耳舐めされるの好きなんですか")
        self.assertEqual((out[0].start, out[0].end), (45.02, 47.42))

    def test_overlap_duplicate_tail_not_duplicated(self):
        # `だ` [52.26-52.58] overlaps `…そうなんだ` [51.58-52.62]; the tail is
        # already present so no duplicate だ.
        segs = [_seg("そうなそうなんだ", 51.58, 52.62),
                _seg("だ", 52.26, 52.58)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "そうなそうなんだ")

    def test_overlap_particle_then_long_gap_cut(self):
        # ね overlaps じゃあウィンウィンです → absorbed; the 1.68s gap to the
        # next utterance still cuts cleanly.
        segs = [_seg("じゃあウィンウィンです", 54.98, 57.30),
                _seg("ね", 56.66, 57.30),
                _seg("私アイアンのっています", 58.98, 61.86)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "じゃあウィンウィンですね")
        self.assertEqual((out[0].start, out[0].end), (54.98, 57.30))
        self.assertEqual(out[1].text, "私アイアンのっています")

    def test_overlap_particle_ignores_defer_to_next(self):
        # Before the fix, the overlapping ね got cut into its own block because
        # _decide deferred to the NEXT boundary (a long pause → strong cut).
        # Overlap now forces absorption regardless of the next context.
        segs = [_seg("よろしくお願いします", 62.90, 64.66),
                _seg("ね", 64.32, 64.40),
                _seg("また来てください", 70.00, 72.00)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "よろしくお願いしますね")
        self.assertEqual((out[0].start, out[0].end), (62.90, 64.66))
        self.assertEqual(out[1].text, "また来てください")

    def test_strong_ending_cuts_despite_weak_next(self):
        # The real 14s run-on from the log: 「…みたいですね」+「…好きで」+「続けてください」.
        # The ですね (strong ending) boundary must cut even though the NEXT
        # boundary is a weak particle ending (好きで) that would otherwise
        # defer the cut and keep the whole thing one sentence.
        segs = [_seg("何のことか分からなかったらスルーで大丈夫程度知ってるみたいですね", 24.22, 33.50),
                _seg("私男人に耳舐めするのが好きで", 33.52, 35.50),
                _seg("続けてください", 36.00, 38.22)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "何のことか分からなかったらスルーで大丈夫程度知ってるみたいですね")
        # the particle-ending 好きで clause is incomplete → merges forward
        self.assertEqual(out[1].text, "私男人に耳舐めするのが好きで続けてください")

    def test_strong_ending_short_utterance_still_merges(self):
        # Sub-second strong-ending fragment + short gap → the min_sec gate
        # re-merges it instead of emitting a 0.3s standalone だ.
        segs = [_seg("だ", 0.0, 0.3), _seg("おかえり", 0.8, 2.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "だおかえり")

    def test_hard_max_cap(self):
        # Three 6s fragments, tiny gaps, no syntax → merged until a merge would
        # push past 15s, then force-cut.
        segs = [_seg("あ", 0.0, 6.0), _seg("い", 6.0, 12.0), _seg("う", 12.0, 18.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0].start, out[0].end), (0.0, 12.0))
        self.assertEqual((out[1].start, out[1].end), (12.0, 18.0))

    def test_soft_max_finds_natural_boundary(self):
        # Period at the 12s mark → soft pressure cuts there (not at 15s).
        segs = [_seg("あ", 0.0, 6.0), _seg("い。", 6.0, 12.0), _seg("う", 12.0, 18.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "あい.")
        self.assertEqual((out[0].start, out[0].end), (0.0, 12.0))

    def test_short_standalone_kept(self):
        segs = [_seg("はい", 0.0, 0.3), _seg("おかえり", 2.0, 4.0)]
        out = merge_into_sentences(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].text, "はい")

    def test_single_fragment(self):
        out = merge_into_sentences([_seg("はい", 0.0, 0.3)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "はい")

    def test_empty_input(self):
        self.assertEqual(merge_into_sentences([]), [])


if __name__ == "__main__":
    unittest.main()
