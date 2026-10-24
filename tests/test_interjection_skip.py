"""Tests for the interjection-skip feature.

Interjection blocks (≤3 Chinese characters after removing punctuation, e.g.
"嗯", "啊，呀！") have their translation TTS skipped: no API call, no TTS file,
and the pipeline keeps the original audio for the span. This file covers the
counting/marking helpers, the generate_tts skip, and the split_audio
carry-through of the ``skip_tts`` flag.

Run from the project root:  python -m unittest tests.test_interjection_skip
"""

import os
import sys
import tempfile
import unittest
import wave
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.audio_splitter import split_audio
from modules.pipeline import count_cjk_chars, mark_interjection_skips
from modules.tts_engine import generate_tts
from modules.vtt_parser import VTTBlock


class CountCjkCharsTest(unittest.TestCase):
    def test_counts_ideographs_only(self):
        self.assertEqual(count_cjk_chars("你好世界"), 4)
        self.assertEqual(count_cjk_chars("嗯"), 1)
        self.assertEqual(count_cjk_chars("嗯嗯嗯"), 3)
        self.assertEqual(count_cjk_chars("嗯嗯嗯嗯"), 4)
        self.assertEqual(count_cjk_chars("好的，我知道了。"), 6)  # 好 的 我 知 道 了
        self.assertEqual(count_cjk_chars("啊，呀！"), 2)
        self.assertEqual(count_cjk_chars("好　好"), 2)  # full-width space ignored

    def test_tags_do_not_inflate_count(self):
        # English emotion tags are never counted as CJK.
        self.assertEqual(count_cjk_chars("[whispering]嗯"), 1)
        self.assertEqual(count_cjk_chars("[whispering] 好的，我知道了。"), 6)

    def test_non_cjk_is_zero(self):
        for text in ("", "Hello 123", "12345", "……！？", "mm~"):
            self.assertEqual(count_cjk_chars(text), 0)
        # 待 is itself a CJK ideograph, so a Japanese phrase containing it counts it.
        self.assertEqual(count_cjk_chars("ちょっと待って"), 1)

    def test_none_text_is_zero(self):
        self.assertEqual(count_cjk_chars(None), 0)


class MarkInterjectionSkipsTest(unittest.TestCase):
    def test_flags_short_interjections_only(self):
        blocks = [
            VTTBlock(0, 0, 1000, "嗯"),
            VTTBlock(1, 1000, 2000, "啊，呀！"),
            VTTBlock(2, 2000, 3000, "好的，我知道了。"),
            VTTBlock(3, 3000, 4000, "嗯嗯嗯"),
            VTTBlock(4, 4000, 5000, ""),
            VTTBlock(5, 5000, 6000, "[whispering]嗯"),
        ]
        flagged = mark_interjection_skips(blocks)
        self.assertEqual(flagged, 5)
        self.assertTrue(blocks[0].skip_tts)
        self.assertTrue(blocks[1].skip_tts)
        self.assertFalse(blocks[2].skip_tts)  # 5 chars — real sentence, kept
        self.assertTrue(blocks[3].skip_tts)
        self.assertTrue(blocks[4].skip_tts)
        self.assertTrue(blocks[5].skip_tts)

    def test_max_chars_threshold(self):
        blocks = [VTTBlock(0, 0, 1000, "嗯嗯嗯")]
        self.assertEqual(mark_interjection_skips(blocks, max_chars=2), 0)
        self.assertFalse(blocks[0].skip_tts)
        self.assertEqual(mark_interjection_skips(blocks), 1)
        self.assertTrue(blocks[0].skip_tts)

    def test_upstream_skipped_block_is_not_counted_as_chinese_short_text(self):
        blocks = [VTTBlock(0, 0, 1000, "")]
        blocks[0].skip_tts = True
        self.assertEqual(mark_interjection_skips(blocks), 0)
        self.assertTrue(blocks[0].skip_tts)


class GenerateTtsSkipTest(unittest.TestCase):
    def test_interjection_segments_not_synthesized(self):
        adapter = _FakeTTSAdapter()
        segs = [
            {"index": 0, "text": "你好世界", "speaker": None, "skip_tts": False},
            {"index": 1, "text": "嗯", "speaker": None, "skip_tts": True},
            {"index": 2, "text": "再见了", "speaker": None, "skip_tts": False},
        ]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("modules.tts_engine._get_adapter", return_value=adapter):
            out = generate_tts(segs, tmp, max_concurrent=2)

        texts = [t for t, _, _ in adapter.synthesized]
        self.assertIn("你好世界", texts)
        self.assertIn("再见了", texts)
        self.assertNotIn("嗯", texts)
        # Skipped seg keeps no tts_file; the others were synthesized.
        self.assertIsNone(out[1].get("tts_file"))
        self.assertTrue(out[0].get("tts_file"))
        self.assertTrue(out[2].get("tts_file"))

    def test_skip_flag_defaults_to_off(self):
        adapter = _FakeTTSAdapter()
        segs = [{"index": 0, "text": "你好", "speaker": None}]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("modules.tts_engine._get_adapter", return_value=adapter):
            out = generate_tts(segs, tmp)
        self.assertTrue(out[0].get("tts_file"))


class SplitAudioSkipTest(unittest.TestCase):
    def test_skip_flag_carried_to_segment(self):
        blocks = [
            VTTBlock(0, 0, 1000, "你好世界"),
            VTTBlock(1, 1000, 2000, "嗯"),
        ]
        blocks[1].skip_tts = True
        def fake_normalize(_source, target):
            with wave.open(target, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00" * 24000 * 2 * 2 * 5)
            return target

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("modules.audio_splitter.get_audio_duration", return_value=5.0), \
             mock.patch("modules.audio_splitter._normalize_audio", side_effect=fake_normalize):
            segs = split_audio("dummy.mp3", blocks, os.path.join(tmp, "segs"))

        segs.sort(key=lambda s: s["index"])
        self.assertIs(segs[0]["skip_tts"], False)
        self.assertIs(segs[1]["skip_tts"], True)


class _FakeTTSAdapter:
    """Stand-in for the TTS adapter: records synthesize calls, writes a stub file."""

    def __init__(self):
        self.synthesized = []

    def get_voice_for_speaker(self, speaker):
        return "zh-CN-Test"

    async def synthesize(self, text, voice, path, rate="+0%", pitch="+0Hz"):
        self.synthesized.append((text, voice, path))
        with open(path, "wb") as f:
            f.write(b"\x00" * 200)


if __name__ == "__main__":
    unittest.main()
