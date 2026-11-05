"""Unit and architecture tests for toggleable capabilities in modules.cleaning."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

from modules.cleaning import (
    InterjectionSkipCapability,
    InterjectionSkipConfig,
    JapaneseCleanCapability,
    JapaneseCleanConfig,
    MecabRegexJapaneseCleaner,
    MorphToken,
)
from modules.pipeline import _create_japanese_cleaner_with_fallback
from modules.vtt_parser import VTTBlock


class CleaningArchitectureTests(unittest.TestCase):
    def test_invalid_configs_are_rejected_at_boundary(self):
        for kwargs in ({"enabled": "yes"}, {"cleaner": ""}, {"cleaner": "   "}):
            with self.subTest(config="JapaneseCleanConfig", kwargs=kwargs):
                with self.assertRaises(ValueError):
                    JapaneseCleanConfig(**kwargs)

        for kwargs in ({"enabled": "no"}, {"max_chars": -1}, {"max_chars": True}):
            with self.subTest(config="InterjectionSkipConfig", kwargs=kwargs):
                with self.assertRaises(ValueError):
                    InterjectionSkipConfig(**kwargs)

    def test_cleaning_package_does_not_import_orchestrator_or_downstream_domains(self):
        directory = Path(__file__).resolve().parents[1] / "modules" / "cleaning"
        forbidden = {
            "pipeline",
            "subtitle_processing",
            "llm",
            "detect",
            "asr",
            "tts_engine",
            "arrangement",
            "vtt_parser",
        }
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        forbidden.intersection(node.module.split(".")),
                        f"Forbidden import in {path}: {node.module}",
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            forbidden.intersection(alias.name.split(".")),
                            f"Forbidden import in {path}: {alias.name}",
                        )


class CapJapaneseCleanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capability = JapaneseCleanCapability(JapaneseCleanConfig(enabled=True, cleaner="mecab-regex"))

    def test_disabled_capability_bypasses_mecab_initialization_and_preserves_blocks(self):
        factory = Mock(side_effect=AssertionError("disabled capability must not initialize cleaner"))
        for cfg in (
            JapaneseCleanConfig(enabled=False, cleaner="mecab-regex"),
            JapaneseCleanConfig(enabled=True, cleaner="none"),
            JapaneseCleanConfig(enabled=True, cleaner="disabled"),
        ):
            with self.subTest(cfg=cfg):
                cap = JapaneseCleanCapability(cfg, cleaner_factory=factory)
                self.assertFalse(cap.enabled)
                self.assertEqual(cap.name, "disabled")
                self.assertEqual(cap.clean_text("はぁ...はぁ..."), ("はぁ...はぁ...", ()))

                source = [
                    VTTBlock(0, 10500, 12820, "はぁ...はぁ..."),
                    VTTBlock(1, 16420, 17140, ""),
                ]
                result = cap.clean(source)
                self.assertIsNot(result.blocks[0], source[0])
                self.assertEqual(result.cleaner, "disabled")
                self.assertEqual([b.text for b in result.blocks], ["はぁ...はぁ...", ""])
                self.assertEqual([b.skip_tts for b in result.blocks], [False, True])
                self.assertEqual(result.kept_count, 1)
                self.assertEqual(result.skipped_count, 1)
        factory.assert_not_called()

    def test_pipeline_helper_supports_disabled_switch(self):
        cap = _create_japanese_cleaner_with_fallback("mecab-regex", enabled=False)
        self.assertFalse(cap.enabled)
        self.assertEqual(cap.name, "disabled")

    def test_greetings_are_preserved_while_breathing_interjections_are_dropped(self):
        self.assertEqual(self.capability.clean_text("はぁ...はぁ...")[0], "")
        self.assertEqual(self.capability.clean_text("こんばんは...")[0], "こんばんは")
        self.assertEqual(self.capability.clean_text("おはようございます")[0], "おはようございます")
        self.assertEqual(self.capability.clean_text("ありがとうございます")[0], "ありがとうございます")
        self.assertEqual(self.capability.clean_text("こんにちは")[0], "こんにちは")
        self.assertEqual(self.capability.clean_text("さようなら")[0], "さようなら")

    def test_greeting_pos_detail_is_preserved_for_custom_dictionaries(self):
        class GreetingTokenizer:
            name = "custom-greeting"

            def tokenize(self, text: str) -> list[MorphToken]:
                return [MorphToken(surface=text, pos_large="感動詞", pos_detail="挨拶")]

        cleaner = MecabRegexJapaneseCleaner(tokenizer=GreetingTokenizer())
        cleaned, removed = cleaner.clean_text("ごきげんよう")
        self.assertEqual(cleaned, "ごきげんよう")
        self.assertEqual(removed, ())

    def test_first_coarse_slice_keeps_konbanwa_and_skips_only_pure_breathing(self):
        blocks = [
            VTTBlock(0, 10500, 12820, "はぁ...はぁ..."),
            VTTBlock(1, 16420, 17140, "こんばんは..."),
            VTTBlock(2, 17860, 19540, "お布団"),
        ]
        result = self.capability.clean(blocks)
        self.assertEqual([b.text for b in result.blocks], ["", "こんばんは", "布団"])
        self.assertEqual([b.skip_tts for b in result.blocks], [True, False, False])
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(result.kept_count, 2)


class CapInterjectionSkipTests(unittest.TestCase):
    def test_filter_returns_new_blocks_and_audit_records_without_mutating_inputs(self):
        cap = InterjectionSkipCapability(InterjectionSkipConfig(enabled=True, max_chars=3))
        upstream_skipped = VTTBlock(0, 0, 1000, "")
        upstream_skipped.skip_tts = True
        short_block = VTTBlock(1, 1000, 2000, "嗯，啊！")
        long_block = VTTBlock(2, 2000, 3000, "晚上好，欢迎回来。")
        source = [upstream_skipped, short_block, long_block]

        result = cap.filter(source)

        # Input blocks remain untouched
        self.assertFalse(short_block.skip_tts)
        self.assertFalse(long_block.skip_tts)
        self.assertIsNot(result.blocks[1], short_block)

        # Returned blocks reflect policy
        self.assertTrue(result.enabled)
        self.assertEqual(result.flagged_count, 1)
        self.assertEqual([b.skip_tts for b in result.blocks], [True, True, False])
        self.assertEqual(
            [r.reason for r in result.records],
            ["upstream_skip_tts", "short_cjk_interjection", "kept_translation"],
        )

    def test_disabled_capability_preserves_short_blocks_and_upstream_skips(self):
        cap = InterjectionSkipCapability(InterjectionSkipConfig(enabled=False, max_chars=3))
        upstream_skipped = VTTBlock(0, 0, 1000, "")
        upstream_skipped.skip_tts = True
        short_block = VTTBlock(1, 1000, 2000, "嗯")
        result = cap.filter([upstream_skipped, short_block])

        self.assertFalse(result.enabled)
        self.assertEqual(result.flagged_count, 0)
        self.assertEqual([b.skip_tts for b in result.blocks], [True, False])
        self.assertEqual([r.reason for r in result.records], ["disabled", "disabled"])


if __name__ == "__main__":
    unittest.main()
