"""Tests for modules.pink_segments (pink-segment timeline model).

Run from the project root:  python -m unittest tests.test_pink_segments
Offline — pure geometry, no audio I/O.
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.pink_segments import (
    PinkSegment,
    attach_pink_geometry,
    calculate_pink_segments,
)
from modules.vtt_parser import VTTBlock


def _block(start, end, text="T", index=0):
    return VTTBlock(index=index, start_ms=start, end_ms=end, text=text)


class CalculatePinkSegmentsTests(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(calculate_pink_segments([], 1000), [])

    def test_continuous_from_zero_to_end(self):
        blocks = [
            _block(5000, 6000, "a", 0),
            _block(12000, 13000, "b", 1),
            _block(20000, 21000, "c", 2),
        ]
        out = calculate_pink_segments(blocks, 30000)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].pink_start_ms, 0)
        self.assertEqual(out[-1].pink_end_ms, 30000)
        for a, b in zip(out, out[1:]):
            self.assertEqual(a.pink_end_ms, b.pink_start_ms, "segments must be contiguous")

    def test_gap_midpoint_boundary(self):
        # Gap 6000..12000 → midpoint 9000 splits it fairly.
        out = calculate_pink_segments(
            [_block(5000, 6000, "a", 0), _block(12000, 13000, "b", 1)], 20000)
        self.assertEqual(out[0].pink_end_ms, 9000)
        self.assertEqual(out[1].pink_start_ms, 9000)

    def test_orange_bounds_kept(self):
        out = calculate_pink_segments([_block(5000, 6000, "a", 0)], 20000)
        self.assertEqual(out[0].orange_start_ms, 5000)
        self.assertEqual(out[0].orange_end_ms, 6000)

    def test_tts_delay_and_budget_formula(self):
        # P1 = [0, 9000], O1 = [5000, 6000]: delay = 5000-0 = 5000,
        # budget = 9000-5000 = 4000. P2 = [9000, 20000], O2 = [12000, 13000]:
        # delay = 12000-9000 = 3000, budget = 20000-12000 = 8000.
        out = calculate_pink_segments(
            [_block(5000, 6000, "a", 0), _block(12000, 13000, "b", 1)], 20000)
        self.assertEqual(out[0].tts_delay_ms, 5000)
        self.assertEqual(out[0].max_allowed_tts_ms, 4000)
        self.assertEqual(out[1].tts_delay_ms, 3000)
        self.assertEqual(out[1].max_allowed_tts_ms, 8000)

    def test_leading_and_trailing_silence_folded_in(self):
        out = calculate_pink_segments([_block(8000, 9000, "a", 0)], 15000)
        self.assertEqual(out[0].pink_start_ms, 0)
        self.assertEqual(out[0].pink_end_ms, 15000)
        self.assertEqual(out[0].tts_delay_ms, 8000)
        self.assertEqual(out[0].max_allowed_tts_ms, 15000 - 8000)

    def test_overlapping_blocks_clamp_to_first_end(self):
        # O1 ends after O2 starts → cut degrades to O1.end so each slice still
        # covers its own speech; O2 gets a clamped (0) trigger delay.
        out = calculate_pink_segments(
            [_block(1000, 5000, "a", 0), _block(3000, 7000, "b", 1)], 10000)
        self.assertEqual(out[0].pink_end_ms, 5000)
        self.assertEqual(out[1].pink_start_ms, 5000)
        self.assertEqual(out[1].tts_delay_ms, 0)
        self.assertEqual(out[1].max_allowed_tts_ms, 7000)

    def test_unsorted_input_sorted_by_start(self):
        out = calculate_pink_segments(
            [_block(20000, 21000, "b", 0), _block(0, 500, "a", 1)], 30000)
        self.assertEqual([p.subtitle_data.text for p in out], ["a", "b"])
        self.assertEqual(out[0].pink_start_ms, 0)
        self.assertEqual(out[1].pink_end_ms, 30000)

    def test_single_block_covers_whole(self):
        out = calculate_pink_segments([_block(8000, 9000, "a", 0)], 15000)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].pink_start_ms, 0)
        self.assertEqual(out[0].pink_end_ms, 15000)

    def test_block_start_beyond_total_clamps_budget(self):
        out = calculate_pink_segments([_block(9000, 9500, "a", 0)], 8000)
        self.assertEqual(out[0].pink_end_ms, 8000)
        self.assertEqual(out[0].max_allowed_tts_ms, 0)


class AttachPinkGeometryTests(unittest.TestCase):
    def test_writes_geometry_onto_blocks(self):
        blocks = [_block(0, 1000, "a", 0), _block(3000, 4000, "b", 1)]
        segments = attach_pink_geometry(blocks, 10000)
        self.assertIsInstance(segments[0], PinkSegment)
        self.assertEqual(blocks[0].pink_start_ms, 0)
        self.assertEqual(blocks[0].pink_end_ms, 2000)
        self.assertEqual(blocks[0].tts_delay_ms, 0)
        self.assertEqual(blocks[0].max_allowed_tts_ms, 2000)
        self.assertEqual(blocks[1].pink_start_ms, 2000)
        self.assertEqual(blocks[1].pink_end_ms, 10000)
        self.assertEqual(blocks[1].tts_delay_ms, 1000)
        self.assertEqual(blocks[1].max_allowed_tts_ms, 7000)


if __name__ == "__main__":
    unittest.main()
