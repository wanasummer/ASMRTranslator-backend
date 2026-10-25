"""Tests for modules.subtitles.timeline — source→output timeline mapping."""

import unittest
from dataclasses import dataclass

from modules.subtitles.timeline import align_blocks_to_output


@dataclass
class Block:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    skip_tts: bool = False


def segment(index: int, start_ms: int, end_ms: int) -> dict:
    return {"index": index, "start_ms": start_ms, "end_ms": end_ms,
            "segment_file": f"seg_{index}.wav", "vc_file": f"dual_{index}.wav"}


def placement(index: int, start_ms: int, end_ms: int) -> dict:
    return {"index": index, "start_ms": start_ms, "end_ms": end_ms}


class AlignBlocksToOutputTests(unittest.TestCase):
    def test_identical_timeline_is_a_noop(self):
        blocks = [Block(0, 0, 1000, "a"), Block(1, 2000, 3000, "b")]
        segments = [segment(0, 0, 1000), segment(1, 2000, 3000)]
        placed = [placement(0, 0, 1000), placement(1, 2000, 3000)]

        aligned, dropped = align_blocks_to_output(blocks, segments, placed)

        self.assertEqual(dropped, [])
        self.assertEqual([(b.start_ms, b.end_ms) for b in aligned],
                         [(0, 1000), (2000, 3000)])

    def test_dual_mode_shifts_every_cue_after_the_first_insertion(self):
        # Segment 0 is 1 s of source audio plus 5 s of Chinese TTS, so it occupies
        # 0–6000 ms in the export; everything after it moves 5 s later.
        blocks = [Block(0, 0, 1000, "a"), Block(1, 2000, 3000, "b"), Block(2, 4000, 5000, "c")]
        segments = [segment(0, 0, 1000), segment(1, 2000, 3000), segment(2, 4000, 5000)]
        placed = [
            placement(0, 0, 6000),
            placement(1, 6000, 7000),
            placement(2, 7000, 8000),
        ]

        aligned, dropped = align_blocks_to_output(blocks, segments, placed)

        self.assertEqual(dropped, [])
        self.assertEqual([(b.start_ms, b.end_ms) for b in aligned],
                         [(0, 1000), (6000, 7000), (7000, 8000)])

    def test_crossfade_shrinks_the_timeline(self):
        # 100 ms crossfade: segment 1 starts 100 ms earlier than in the source.
        blocks = [Block(0, 0, 300, "a"), Block(1, 500, 1500, "b")]
        segments = [segment(0, 0, 500), segment(1, 500, 1500)]
        placed = [placement(0, 0, 500), placement(1, 400, 1400)]

        aligned, _ = align_blocks_to_output(blocks, segments, placed)

        self.assertEqual([(b.start_ms, b.end_ms) for b in aligned],
                         [(0, 300), (400, 1400)])

    def test_blocks_of_dropped_segments_are_reported_not_shifted(self):
        blocks = [Block(0, 0, 1000, "a"), Block(1, 1000, 2000, "short"), Block(2, 2000, 3000, "c")]
        segments = [segment(0, 0, 1000), segment(1, 1000, 2000), segment(2, 2000, 3000)]
        placed = [placement(0, 0, 1000), placement(2, 1000, 2000)]   # segment 1 skipped by joiner

        aligned, dropped = align_blocks_to_output(blocks, segments, placed)

        self.assertEqual(dropped, [1])
        self.assertEqual([b.index for b in aligned], [0, 2])
        self.assertEqual([(b.start_ms, b.end_ms) for b in aligned],
                         [(0, 1000), (2000, 3000)])

    def test_text_and_flags_are_preserved_and_inputs_untouched(self):
        blocks = [Block(0, 0, 1000, "a", speaker="s1"), Block(1, 1000, 2000, "b", skip_tts=True)]
        segments = [segment(0, 0, 1000), segment(1, 1000, 2000)]
        placed = [placement(0, 0, 1000), placement(1, 3000, 4000)]
        snapshot = [(b.index, b.start_ms, b.end_ms, b.text, b.speaker, b.skip_tts) for b in blocks]

        aligned, _ = align_blocks_to_output(blocks, segments, placed)

        self.assertEqual((aligned[0].speaker, aligned[1].skip_tts), ("s1", True))
        self.assertEqual([b.text for b in aligned], ["a", "b"])
        self.assertEqual(snapshot,
                         [(b.index, b.start_ms, b.end_ms, b.text, b.speaker, b.skip_tts) for b in blocks])

    def test_result_stays_monotonic_when_placement_overlaps(self):
        blocks = [Block(0, 0, 1000, "a"), Block(1, 1000, 2000, "b")]
        segments = [segment(0, 0, 1000), segment(1, 1000, 2000)]
        placed = [placement(0, 0, 1000), placement(1, 500, 1500)]   # 500 ms overlap

        aligned, _ = align_blocks_to_output(blocks, segments, placed)

        self.assertGreaterEqual(aligned[1].start_ms, aligned[0].end_ms)
        self.assertEqual((aligned[1].start_ms, aligned[1].end_ms), (1000, 2000))


if __name__ == "__main__":
    unittest.main()
