from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from unittest.mock import Mock, patch

import os
import tempfile
import unittest

from modules.subtitles.final_export import (
    FinalSubtitleError,
    build_aligned_subtitles,
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


class AlignedSubtitleTest(unittest.TestCase):
    def test_actual_overlaps_stable_ids_and_omissions_are_preserved(self):
        from modules.audio_joiner import SegmentPlacement
        blocks = [VTTBlock(10, 5000, 6000, "第一句", speaker="角色"),
                  VTTBlock(20, 6000, 7000, "丢弃"),
                  VTTBlock(30, 7000, 8000, "第三句")]
        placements = (SegmentPlacement("10", 0, 1000, 0, 1000),
                      SegmentPlacement("30", 900, 1900, 900, 1900))
        cues = build_aligned_subtitles(blocks, placements)
        self.assertEqual([(c.index, c.start_ms, c.end_ms) for c in cues],
                         [(1, 0, 1000), (2, 900, 1900)])
        self.assertEqual(cues[0].speaker, "角色")
        self.assertEqual(blocks[0].start_ms, 5000)

    def test_partial_vtt_write_is_removed(self):
        cues = build_final_subtitles([VTTBlock(0, 0, 100, "字幕")])
        with tempfile.TemporaryDirectory() as d:
            def fail(path, text, **kwargs):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("partial", encoding="utf-8")
                raise OSError("disk full")
            with patch("modules.subtitles.final_export.write_text", side_effect=fail):
                with self.assertRaises(OSError):
                    write_final_vtt(cues, d)
            self.assertFalse((Path(d) / "llm_subtitles/final.vtt").exists())

    def test_pipeline_exports_returned_timeline_through_existing_final_vtt_contract(self):
        from modules.pipeline import run_pipeline
        from modules.audio_joiner import JoinedAudio, SegmentPlacement
        # CLI import loads the user dotenv; keep it out of other tests.
        with patch.dict(os.environ):
            import main
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            source = root / "source.wav"
            source.write_bytes(b"source")
            output = root / "work"
            blocks = [VTTBlock(10, 5000, 6000, "这是第一条测试字幕"),
                      VTTBlock(20, 7000, 8000, "这是第二条测试字幕")]
            segments = [{"index": i, "tts_file": "tts.wav"} for i in range(2)]
            chain = Mock()
            chain.run.return_value = segments
            builder = Mock(subdir="built")
            builder.build.return_value = segments
            def join(items, path, **kwargs):
                self.assertTrue(kwargs["return_timeline"])
                Path(path).write_bytes(b"wav")
                Path(path).with_suffix(".mp3").write_bytes(b"mp3")
                return JoinedAudio(path, (SegmentPlacement("10", 0, 800, 0, 800),))
            for target, value in {
                "modules.pipeline.parse_vtt": blocks,
                "modules.pipeline.get_audio_duration": 10,
                "modules.pipeline.split_audio": segments,
                "modules.pipeline.generate_tts": segments,
                "modules.pipeline.build_tts_handler_chain": chain,
                "modules.pipeline.align_tts_to_original": segments,
                "modules.pipeline.enforce_tts_budget": segments,
                "modules.pipeline.create_segment_builder": builder,
                "modules.audio_metadata.create_metadata_reader": Mock(),
                "modules.audio_metadata.write_audio_metadata": "metadata.json",
                "modules.high_quality_export.write_export_recipe": "recipe.json",
            }.items():
                stack.enter_context(patch(target, return_value=value))
            stack.enter_context(patch("modules.pipeline.join_audio", side_effect=join))
            with redirect_stdout(StringIO()):
                result = run_pipeline(str(source), "source.vtt", str(output))
            aligned = Path(result["final_vtt"])
            self.assertIn("00:00:00.000 --> 00:00:00.800", aligned.read_text(encoding="utf-8"))
            self.assertNotIn(blocks[1].text, aligned.read_text(encoding="utf-8"))
            self.assertEqual(blocks[0].start_ms, 5000)
            self.assertEqual(sorted(p.name for p in (output / "llm_subtitles").iterdir()), ["final.vtt"])
            history = json.loads(Path(result["run_result"]).read_text(encoding="utf-8"))
            self.assertEqual(history["runs"]["work"]["files"]["finalVtt"],
                             "llm_subtitles/final.vtt")
            frame = StringIO()
            with redirect_stdout(frame):
                main._emit_result(result)
            self.assertEqual(json.loads(frame.getvalue().removeprefix("[PIPELINE_RESULT]"))
                             ["files"]["finalVtt"], str(aligned))
            from modules.error import PipelineError
            with patch("modules.subtitles.final_export.build_aligned_subtitles",
                       side_effect=FinalSubtitleError("no cues")), redirect_stdout(StringIO()):
                with self.assertRaises(PipelineError):
                    run_pipeline(str(source), "source.vtt", str(output))
            self.assertFalse(aligned.exists())



if __name__ == "__main__":
    unittest.main()
