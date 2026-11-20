import unittest

from modules.dict import SpeechRate
from modules.speech_pace import chinese_char_count, classify_speech_rate, estimate_tts_ms, landing_hint_ms


class SpeechPaceTests(unittest.TestCase):
    def test_counts_only_chinese_characters(self):
        self.assertEqual(chinese_char_count("你好，ASMR。慢慢来"), 5)

    def test_classifies_the_nearest_pace_tier(self):
        self.assertIs(classify_speech_rate(5, 2000), SpeechRate.NORMAL)
        self.assertIs(classify_speech_rate(10, 2000), SpeechRate.FAST)
        self.assertIs(classify_speech_rate(2, 2000), SpeechRate.SLOW)
        self.assertIsNone(classify_speech_rate(0, 2000))
        self.assertIsNone(classify_speech_rate(5, 0))

    def test_estimates_voice_duration_at_the_tier(self):
        self.assertEqual(estimate_tts_ms("一二三四五六七八九十", 2000), 3125)
        self.assertIsNone(estimate_tts_ms("hello", 2000))

    def test_landing_covers_the_original_speech(self):
        self.assertEqual(landing_hint_ms("一二三四五六七八九十", 1000, 3000), 4125)
        self.assertEqual(landing_hint_ms("慢慢", 1000, 3000), 3000)
        self.assertIsNone(landing_hint_ms("", 1000, 3000))


if __name__ == "__main__":
    unittest.main()
