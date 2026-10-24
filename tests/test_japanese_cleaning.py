"""Tests for the pluggable ASR Japanese-cleaning stage."""

import json
import os
import tempfile
import unittest
from unittest import mock

from modules.japanese_cleaning import (
    JapaneseCleanerUnavailableError,
    MecabRegexJapaneseCleaner,
    MecabTokenizer,
    RegexJapaneseCleaner,
    write_cleaning_artifacts,
)
from modules.pipeline import _create_japanese_cleaner_with_fallback
from modules.vtt_parser import VTTBlock


class MecabRegexCleanerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleaner = MecabRegexJapaneseCleaner()

    def test_documented_mecab_and_regex_example(self):
        cleaned, _ = self.cleaner.clean_text(
            "あのう...んっ...今日はね、耳かきをするよ...うん"
        )
        self.assertEqual(cleaned, "今日は耳かきをする")

    def test_chinese_asr_vocalizations_are_removed_before_mecab(self):
        cleaned, _ = self.cleaner.clean_text("啊嗯嗯嗯啊嗯好き")
        self.assertEqual(cleaned, "好き")

    def test_pure_chinese_vocalization_becomes_empty(self):
        cleaned, _ = self.cleaner.clean_text("嗯嗯嗯嗯嗯嗯嗯")
        self.assertEqual(cleaned, "")

    def test_filler_pos_is_removed_but_content_survives(self):
        cleaned, _ = self.cleaner.clean_text("えっとやっぱり私もえっとえっと")
        self.assertEqual(cleaned, "やっぱり私も")

    def test_no_japanese_length_threshold_is_applied(self):
        cleaned, _ = self.cleaner.clean_text("啊嗯好き嗯")
        self.assertEqual(cleaned, "好き")

    def test_clean_returns_new_blocks_and_marks_empty_as_original_only(self):
        source = [
            VTTBlock(0, 0, 5000, "嗯嗯嗯"),
            VTTBlock(1, 5000, 10000, "啊嗯好き"),
        ]
        result = self.cleaner.clean(source)

        self.assertEqual([block.text for block in source], ["嗯嗯嗯", "啊嗯好き"])
        self.assertEqual([block.text for block in result.blocks], ["", "好き"])
        self.assertEqual([block.skip_tts for block in result.blocks], [True, False])
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(result.kept_count, 1)

    def test_artifacts_contain_cleaned_vtt_and_audit_report(self):
        result = self.cleaner.clean([VTTBlock(0, 480, 21280, "嗯嗯好き")])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_cleaning_artifacts(result, tmp)
            self.assertTrue(os.path.isfile(paths["vtt"]))
            self.assertTrue(os.path.isfile(paths["report"]))
            with open(paths["vtt"], encoding="utf-8") as file:
                self.assertIn("00:00:00.480 --> 00:00:21.280", file.read())
            with open(paths["report"], encoding="utf-8") as file:
                report = json.load(file)
        self.assertEqual(report["kept"], 1)
        self.assertEqual(report["records"][0]["cleaned"], "好き")


class JapaneseCleanerFallbackTest(unittest.TestCase):
    def test_mecab_initialization_error_is_domain_specific(self):
        with mock.patch("MeCab.Tagger", side_effect=RuntimeError("broken dictionary")):
            with self.assertRaisesRegex(
                JapaneseCleanerUnavailableError, "MeCab/UniDic 初始化失败"
            ):
                MecabTokenizer()

    def test_regex_fallback_removes_noise_and_preserves_content(self):
        cleaner = RegexJapaneseCleaner()

        cleaned, _ = cleaner.clean_text("啊嗯好き嗯")

        self.assertEqual(cleaned, "好き")
        self.assertEqual(cleaner.name, "regex-only")

    def test_pipeline_fails_when_configured_cleaner_is_unavailable(self):
        fallback = RegexJapaneseCleaner()
        unavailable = JapaneseCleanerUnavailableError("dictionary unavailable")
        with mock.patch(
            "modules.japanese_cleaning.create_japanese_cleaner",
            side_effect=unavailable,
        ) as create:
            from modules.error import PipelineError
            with self.assertRaises(PipelineError):
                _create_japanese_cleaner_with_fallback("mecab-regex")

        create.assert_called_once_with("mecab-regex")


if __name__ == "__main__":
    unittest.main()
