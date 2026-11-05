import unittest

from modules.subtitle_timeline import calibrate_final_subtitle_end, SubtitleTimelineError
from modules.vtt_parser import VTTBlock


class FinalSubtitleCalibrationTests(unittest.TestCase):
    def test_only_last_end_changes_and_inputs_and_metadata_are_preserved(self):
        blocks = [VTTBlock(7, 100, 700, "前句", speaker="S01"),
                  VTTBlock(8, 600, 1800, "末句", speaker="S02", source_text="原文")]
        result = calibrate_final_subtitle_end(blocks, 1000)
        self.assertEqual([(b.start_ms, b.end_ms) for b in result.blocks], [(100, 700), (600, 1000)])
        self.assertEqual([(b.index, b.text, b.speaker) for b in result.blocks],
                         [(7, "前句", "S01"), (8, "末句", "S02")])
        self.assertEqual(result.blocks[-1].source_text, "原文")
        self.assertEqual(blocks[-1].end_ms, 1800)
        self.assertIsNot(result.blocks[0], blocks[0])
        self.assertEqual((result.adjustments[0].end_ms, result.adjustments[0].restored_end_ms), (1800, 1000))

    def test_valid_timeline_and_empty_result_need_no_calibration(self):
        for blocks in ([], [VTTBlock(0, 0, 1000, "末句")]):
            result = calibrate_final_subtitle_end(blocks, 1000)
            self.assertEqual(result.adjustments, ())
            self.assertEqual(list(result.blocks), blocks)

    def test_nonfinal_overflow_and_invalid_last_start_are_never_repaired(self):
        for blocks in (
            [VTTBlock(0, 0, 1100, "前句"), VTTBlock(1, 500, 1200, "末句")],
            [VTTBlock(0, 1000, 1200, "静音内")],
            [VTTBlock(0, 200, 200, "零时长")],
            [VTTBlock(0, 500, 700, "前句"), VTTBlock(1, 100, 1200, "倒序")],
        ):
            with self.assertRaises(SubtitleTimelineError):
                calibrate_final_subtitle_end(blocks, 1000)
