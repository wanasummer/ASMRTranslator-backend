"""Tests for modules.vad.segments — ASR slice shaping for FSMN-VAD output.

Run from the project root:  python -m unittest tests.test_vad_segments
Fully offline (no funasr / audio files touched).
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.vad.models import SpeechRegion
from modules.vad.segments import optimize_asmr_segments, split_long_segment


def _region(start, end):
    return SpeechRegion(start=start, end=end)


def _spans(regions):
    return [(r.start, r.end) for r in regions]


class SplitLongSegmentTest(unittest.TestCase):
    def test_under_max_unchanged(self):
        self.assertEqual(_spans(split_long_segment(_region(0, 12))), [(0.0, 12.0)])

    def test_at_exactly_max_unchanged(self):
        self.assertEqual(_spans(split_long_segment(_region(0, 15))), [(0.0, 15.0)])

    def test_offset_under_max_unchanged(self):
        self.assertEqual(_spans(split_long_segment(_region(30, 45))), [(30.0, 45.0)])

    def test_over_max_split_into_overlapping_windows(self):
        # 25 s → 10 s windows stepping 9 s (1 s overlap), last clamped.
        self.assertEqual(
            _spans(split_long_segment(_region(0, 25))),
            [(0.0, 10.0), (9.0, 19.0), (18.0, 25.0)],
        )

    def test_over_max_split_offset(self):
        self.assertEqual(
            _spans(split_long_segment(_region(30, 50))),
            [(30.0, 40.0), (39.0, 49.0), (48.0, 50.0)],
        )


class OptimizeAsmrSegmentsTest(unittest.TestCase):
    def test_small_fragments_merge_to_golden_band(self):
        out = optimize_asmr_segments([_region(0, 3), _region(3, 6)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 6.0)])

    def test_unsorted_input_sorted(self):
        out = optimize_asmr_segments([_region(3, 6), _region(0, 3)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 6.0)])

    def test_gap_over_max_merge_not_merged(self):
        out = optimize_asmr_segments([_region(0, 3), _region(5.5, 8)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 3.0), (5.5, 8.0)])

    def test_isolated_short_fragment_kept_by_default(self):
        # (0,1.5) cannot merge (gap 2.5) but the default 0.3s floor keeps it —
        # ASMR lines are often sub-second, so an isolated short burst is speech.
        out = optimize_asmr_segments([_region(0, 1.5), _region(4, 9)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 1.5), (4.0, 9.0)])

    def test_isolated_fragment_dropped_when_min_sec_raised(self):
        # An explicit high min_sec restores the noise-drop behavior for callers
        # that prefer to discard fragments shorter than that.
        out = optimize_asmr_segments([_region(0, 1.5), _region(4, 9)], min_sec=2.0, padding_sec=0.0)
        self.assertEqual(_spans(out), [(4.0, 9.0)])

    def test_short_fragment_rescued_by_merge(self):
        # (0,1) alone is < 2 s but the gap 0 merge rescues it into (0,4).
        out = optimize_asmr_segments([_region(0, 1), _region(1, 4)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 4.0)])

    def test_short_fragment_merges_beyond_target_max(self):
        # Rescue a short fragment even when the combined span exceeds the target.
        out = optimize_asmr_segments(
            [_region(0, 3), _region(3, 13)], target_max_sec=12.0, padding_sec=0.0,
        )
        self.assertEqual(_spans(out), [(0.0, 13.0)])

    def test_six_second_minimum_merge_policy(self):
        for first_end, combined_end, expected in (
            (4, 27, [(0.0, 27.0)]),
            (8, 18, [(0.0, 18.0)]),
            (8, 27, [(0.0, 8.0), (8.0, 27.0)]),
            (6, 18, [(0.0, 18.0)]),
            (8, 25, [(0.0, 25.0)]),
        ):
            with self.subTest(first_end=first_end, combined_end=combined_end):
                out = optimize_asmr_segments(
                    [_region(0, first_end), _region(first_end, combined_end)],
                    target_min_sec=6.0, target_max_sec=25.0, max_sec=28.0,
                    padding_sec=0.0,
                )
                self.assertEqual(_spans(out), expected)

    def test_rescued_fragment_still_obeys_hard_split_limit(self):
        out = optimize_asmr_segments(
            [_region(0, 4), _region(4, 27)],
            target_min_sec=6.0, target_max_sec=25.0, max_sec=25.0,
            padding_sec=0.0,
        )
        self.assertEqual(_spans(out), [(0.0, 20.0), (19.0, 27.0)])

    def test_greedy_continues_beyond_target_min(self):
        # Reaching the minimum does not stop nearby regions fitting the target.
        out = optimize_asmr_segments([_region(0, 3), _region(3, 6), _region(6, 9)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 9.0)])

    def test_greedy_fills_target_then_starts_next_slice(self):
        out = optimize_asmr_segments(
            [_region(0, 8), _region(8, 18), _region(18, 25), _region(25, 33)],
            padding_sec=0.0,
        )
        self.assertEqual(_spans(out), [(0.0, 25.0), (25.0, 33.0)])

    def test_long_region_split_after_merge(self):
        # A 30 s span exceeds the default 25 s max → split into 20 s windows
        # (1 s overlap → step 19 s).
        out = optimize_asmr_segments([_region(0, 30)], padding_sec=0.0)
        self.assertEqual(
            _spans(out),
            [(0.0, 20.0), (19.0, 30.0)],
        )

    def test_default_max_sec_keeps_25s_span(self):
        # The pipeline's max length (25 s) is respected: a 25 s region is whole.
        out = optimize_asmr_segments([_region(0, 25)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(0.0, 25.0)])

    def test_padding_adds_buffer_both_sides(self):
        # Default 400 ms padding widens each slice on both edges.
        out = optimize_asmr_segments([_region(2, 5)])
        self.assertEqual(_spans(out), [(1.6, 5.4)])

    def test_padding_clamped_at_audio_start(self):
        # A slice near the file start cannot extend before 0.
        out = optimize_asmr_segments([_region(0.1, 3)])
        self.assertEqual(_spans(out), [(0.0, 3.4)])

    def test_padding_clamped_at_audio_end(self):
        out = optimize_asmr_segments(
            [_region(2086.12, 2095.9)],
            audio_duration_sec=2095.915,
        )
        self.assertEqual(_spans(out), [(2085.72, 2095.915)])

    def test_padding_zero_disables_buffer(self):
        out = optimize_asmr_segments([_region(2, 5)], padding_sec=0.0)
        self.assertEqual(_spans(out), [(2.0, 5.0)])

    def test_padding_applies_after_split(self):
        # Each split window carries its own buffer (clamped at 0 on the head).
        out = optimize_asmr_segments([_region(0, 30)], padding_sec=0.4)
        self.assertEqual(
            _spans(out),
            [(0.0, 20.4), (18.6, 30.4)],
        )

    def test_empty_input(self):
        self.assertEqual(optimize_asmr_segments([]), [])


if __name__ == "__main__":
    unittest.main()
