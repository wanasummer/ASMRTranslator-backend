import os
import tempfile
import unittest

from modules.subtitles.final_export import (
    FinalSubtitleError,
    build_final_subtitles,
    write_final_vtt,
)
from modules.vtt_parser import VTTBlock


class FinalSubtitleExportTest(unittest.TestCase):
    def test_subtitles_only_keeps_skipped_short_cues_and_filters_invalid_text(self):
        blocks = [self.block(0, 10, 500, "嗯", True), self.block(1, 800, 1200, "", True)]
        cues = build_final_subtitles(blocks, include_skipped=True)
        self.assertEqual([(cue.start_ms, cue.end_ms, cue.text) for cue in cues], [(10, 500, "嗯")])
        self.assertTrue(blocks[0].skip_tts)

    def block(self, index, start, end, text, skipped=False, speaker=None):
        block = VTTBlock(index, start, end, text, speaker=speaker)
        block.skip_tts = skipped
        return block

    def test_filters_skips_and_invalid_cues_without_changing_timestamps(self):
        blocks = [
            self.block(0, 0, 300, "嗯", True),
            self.block(1, 1000, 1800, "第一句", speaker="遥"),
            self.block(2, 2000, 2400, ""),
            self.block(3, 3000, 3800, "第二句"),
            self.block(4, 4000, 3900, "坏区间"),
        ]
        cues = build_final_subtitles(blocks)
        self.assertEqual([cue.index for cue in cues], [1, 2])
        self.assertEqual([(cue.start_ms, cue.end_ms) for cue in cues], [(1000, 1800), (3000, 3800)])

        with tempfile.TemporaryDirectory() as output_dir:
            path = write_final_vtt(cues, output_dir)
            with open(path, encoding="utf-8", newline="") as file:
                content = file.read()
        self.assertIn("遥：第一句", content)
        self.assertNotIn("嗯", content)
        self.assertNotIn("\r", content)

    def test_all_skipped_raises_domain_error(self):
        with self.assertRaises(FinalSubtitleError) as caught:
            build_final_subtitles([self.block(0, 0, 500, "嗯", True)])
        self.assertEqual(caught.exception.code, "SUBTITLE_FINAL_EMPTY")


if __name__ == "__main__":
    unittest.main()
