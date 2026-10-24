"""Artifact-directory behavior for repeat pipeline runs."""

import os
import tempfile
import unittest

from modules.pipeline import _clean_previous_run, _load_cached_asr
from modules.task_cache import JsonResultCache, file_digest


class PipelineArtifactCleanupTest(unittest.TestCase):
    def test_whole_asr_cache_requires_matching_input_and_intact_artifact(self):
        with tempfile.TemporaryDirectory() as output_dir:
            directory = os.path.join(output_dir, "asr_subtitles")
            os.makedirs(directory)
            path = os.path.join(directory, "cleaned.vtt")
            with open(path, "w", encoding="utf-8") as target:
                target.write("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nテスト\n")
            cache = JsonResultCache(os.path.join(output_dir, "asr_stage_results"))
            cache.save("matching", {"sha256": file_digest(path)})
            self.assertIsNotNone(_load_cached_asr(output_dir, {}, "matching"))
            self.assertIsNone(_load_cached_asr(output_dir, {}, "changed-settings"))
            with open(path, "a", encoding="utf-8") as target:
                target.write("modified")
            self.assertIsNone(_load_cached_asr(output_dir, {}, "matching"))
    def test_repeat_run_removes_previous_pipeline_log(self):
        with tempfile.TemporaryDirectory() as output_dir:
            old_log = os.path.join(output_dir, "pipeline_20260830_120000.log")
            unrelated = os.path.join(output_dir, "notes.log")
            for path in (old_log, unrelated):
                with open(path, "w", encoding="utf-8") as file:
                    file.write("test")

            removed = _clean_previous_run(output_dir)

            self.assertEqual(removed, 1)
            self.assertFalse(os.path.exists(old_log))
            self.assertTrue(os.path.exists(unrelated))

    def test_repeat_run_preserves_asr_and_tts_cache(self):
        with tempfile.TemporaryDirectory() as output_dir:
            asr_dir = os.path.join(output_dir, "asr_subtitles")
            tts_dir = os.path.join(output_dir, "chunk_0", "02_tts")
            os.makedirs(asr_dir)
            os.makedirs(tts_dir)
            asr_file = os.path.join(asr_dir, "cleaned.vtt")
            tts_file = os.path.join(tts_dir, "tts_0000.mp3")
            final_file = os.path.join(output_dir, "output.mp3")
            for path in (asr_file, tts_file, final_file):
                with open(path, "wb") as file:
                    file.write(b"cache")

            _clean_previous_run(output_dir)

            self.assertTrue(os.path.exists(asr_file))
            self.assertTrue(os.path.exists(tts_file))
            self.assertFalse(os.path.exists(final_file))

    def test_loads_completed_japanese_asr_cache(self):
        with tempfile.TemporaryDirectory() as output_dir:
            asr_dir = os.path.join(output_dir, "asr_subtitles")
            os.makedirs(asr_dir)
            cached = os.path.join(asr_dir, "cleaned.vtt")
            with open(cached, "w", encoding="utf-8") as file:
                file.write(
                    "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nテスト\n"
                )

            result = _load_cached_asr(output_dir, {"output_language": "ja"})

            self.assertIsNotNone(result)
            blocks, language, path = result
            self.assertEqual(language, "ja")
            self.assertEqual(path, cached)
            self.assertEqual(blocks[0].text, "テスト")

    def test_loads_completed_chinese_asr_cache(self):
        with tempfile.TemporaryDirectory() as output_dir:
            asr_dir = os.path.join(output_dir, "asr_subtitles")
            os.makedirs(asr_dir)
            cached = os.path.join(asr_dir, "raw.vtt")
            with open(cached, "w", encoding="utf-8") as file:
                file.write(
                    "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n测试\n"
                )

            result = _load_cached_asr(output_dir, {"output_language": "zh"})

            self.assertIsNotNone(result)
            blocks, language, path = result
            self.assertEqual(language, "zh")
            self.assertEqual(path, cached)
            self.assertEqual(blocks[0].text, "测试")


if __name__ == "__main__":
    unittest.main()
