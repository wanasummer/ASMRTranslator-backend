"""Tests for pipeline's optimized-subtitle WebVTT export.

Run from the project root:  python -m unittest tests.test_subtitle_export
Fully offline — no network, no TTS; only the pure serialization helpers.
"""

import os
import sys
import tempfile
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.pipeline import _format_vtt_timestamp, _write_optimized_subtitles
from modules.vtt_parser import VTTBlock


def _blocks(*texts):
    return [VTTBlock(index=i, start_ms=i * 1000, end_ms=i * 1000 + 500,
                     text=t, speaker=("遥" if i % 2 == 0 else None))
            for i, t in enumerate(texts)]


class TimestampTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_format_vtt_timestamp(0), "00:00:00.000")
        self.assertEqual(_format_vtt_timestamp(1500), "00:00:01.500")
        self.assertEqual(_format_vtt_timestamp(61000), "00:01:01.000")
        self.assertEqual(_format_vtt_timestamp(3600000 + 1250), "01:00:01.250")

    def test_negative_clamped(self):
        self.assertEqual(_format_vtt_timestamp(-5), "00:00:00.000")


class ExportTest(unittest.TestCase):
    def test_writes_optimized_vtt(self):
        blocks = _blocks("[whispering] 你好世界。", "好的，我知道了。")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_optimized_subtitles(tmp, blocks)
            self.assertTrue(path.endswith(os.path.join("llm_subtitles", "optimized.vtt")))
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

        self.assertTrue(content.startswith("WEBVTT"))
        # Block 1 (index 0): speaker prefix + emotion tag preserved
        self.assertIn("1\n00:00:00.000 --> 00:00:00.500\n遥：[whispering] 你好世界。", content)
        # Block 2 (index 1): no speaker, raw text
        self.assertIn("2\n00:00:01.000 --> 00:00:01.500\n好的，我知道了。", content)

    def test_empty_text_ok(self):
        blocks = _blocks("", "只有第二行有字")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_optimized_subtitles(tmp, blocks)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        self.assertIn("00:00:00.000 --> 00:00:00.500\n遥：", content)
        self.assertIn("只有第二行有字", content)

    def test_no_blocks_writes_header_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_optimized_subtitles(tmp, [])
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        self.assertEqual(content, "WEBVTT\n")

    def test_skipped_interjections_are_marked(self):
        blocks = _blocks("嗯", "好的，我知道了。", "啊！")
        blocks[0].skip_tts = True
        blocks[2].skip_tts = True
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_optimized_subtitles(tmp, blocks)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        # Skipped blocks carry the marker; normal blocks are untouched.
        self.assertIn("遥：嗯【已跳过翻译】", content)
        self.assertIn("啊！【已跳过翻译】", content)
        self.assertIn("好的，我知道了。", content)
        self.assertNotIn("好的，我知道了。【已跳过翻译】", content)


if __name__ == "__main__":
    unittest.main()
