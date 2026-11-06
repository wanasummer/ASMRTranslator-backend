import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from pydub import AudioSegment

from modules.arrangement import cached_tts
from modules.arrangement_contracts import ArrangementAudioResult
from modules.mode_builders import create_segment_builder
from modules.storage import initialize_project, read_text, save_text
from modules.timeline_render import TimelineRenderConfig, publish_voice_assets, render_initial_timeline
from modules.vtt_parser import VTTBlock


class TimelineRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        initialize_project(self.work)
        rows = [{"index": 7, "start_ms": 0, "end_ms": 2000, "text": "一二三四五六七八九十"},
                {"index": 13, "start_ms": 6000, "end_ms": 7000, "text": "嗯", "skip_tts": True}]
        save_text(self.work / "debug/_parsed_vtt.json", json.dumps(rows, ensure_ascii=False))
        processed = self.work / "chunk_0/02_tts/phantom_dsp_0000.wav"
        processed.parent.mkdir(parents=True)
        AudioSegment.silent(duration=10000, frame_rate=24000).export(processed, format="wav")
        (self.work / "chunk_0/02_tts/voice_0001.wav").write_bytes(b"stale")
        self.segments = [
            {"index": 0, "segment_id": "7", "tts_file": str(processed), "start_ms": 0, "end_ms": 4000,
             "original_start_ms": 0, "original_end_ms": 2000},
            {"index": 1, "segment_id": "13", "skip_tts": True, "start_ms": 4000, "end_ms": 9000,
             "original_start_ms": 6000, "original_end_ms": 7000},
        ]

    def test_only_follow_original_overlay_modes_render_on_the_timeline(self):
        self.assertTrue(create_segment_builder(phantom_mode=True, sync_mode="follow_original").timeline_render)
        self.assertTrue(create_segment_builder(retain_original=True, sync_mode="follow_original").timeline_render)
        self.assertFalse(create_segment_builder(phantom_mode=True, sync_mode="follow_translation").timeline_render)
        self.assertFalse(create_segment_builder(retain_original=True).timeline_render)
        self.assertFalse(create_segment_builder().timeline_render)

    def test_processed_voice_is_what_studio_and_arrangement_read(self):
        published = publish_voice_assets(self.segments, self.work)
        self.assertEqual(published, {0: self.work / "chunk_0/02_tts/voice_0000.wav"})
        self.assertFalse((self.work / "chunk_0/02_tts/voice_0001.wav").exists())
        (self.work / "chunk_0/02_tts/tts_0000.mp3").write_bytes(b"raw")
        self.assertEqual(cached_tts(self.work, 0), published[0])
        self.assertEqual(len(AudioSegment.from_file(published[0])), 10000)

    def test_renders_the_untouched_draft_with_the_arrangement_policy(self):
        adapter, audio = Mock(), Mock()
        adapter.render.side_effect = lambda plan: ArrangementAudioResult(audio, plan.clips, 0)
        result = render_initial_timeline(self.segments, str(self.work), "source.wav",
                                         TimelineRenderConfig(1.2, .85), adapter)
        plan = adapter.render.call_args.args[0]
        self.assertIs(result.plan, plan)
        self.assertEqual(len(plan.clips), 1)
        clip = plan.clips[0]
        self.assertEqual(clip.path, self.work / "chunk_0/02_tts/voice_0000.wav")
        self.assertEqual((clip.start_ms, clip.end_ms, clip.sample_end_ms), (0, 3125, 3750))
        self.assertAlmostEqual(clip.volume, .85)
        self.assertEqual([call.args[0] for call in audio.export.call_args_list],
                         [self.work / "output.wav", self.work / "output.mp3"])
        self.assertEqual([(p.block_id, p.subtitle_start_ms, p.subtitle_end_ms) for p in result.placements],
                         [("7", 0, 2000), ("13", 6000, 7000)])

    def test_pipeline_routes_timeline_modes_through_the_renderer_only(self):
        from modules.pipeline import run_pipeline
        from modules.audio_joiner import SegmentPlacement
        from modules.timeline_render import TimelineRender
        with patch.dict(os.environ):
            import main  # noqa: F401 - CLI import loads the user dotenv; keep it out of other tests.
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            source = root / "source.wav"
            source.write_bytes(b"source")
            output = root / "work"
            blocks = [VTTBlock(10, 5000, 6000, "这是第一条测试字幕")]
            segments = [{"index": 0, "tts_file": "voice.wav"}]
            chain = Mock()
            chain.run.return_value = segments
            builder = Mock(subdir="built", timeline_render=True)

            def render(items, output_dir, audio_path, config):
                self.assertIs(items, segments)
                self.assertEqual(config, TimelineRenderConfig(1.15, .5))
                Path(output_dir, "output.wav").write_bytes(b"wav")
                Path(output_dir, "output.mp3").write_bytes(b"mp3")
                return TimelineRender(str(Path(output_dir, "output.wav")),
                                      (SegmentPlacement("10", 0, 9000, 5000, 6000),), Mock())
            mocks = {}
            for target, value in {
                "modules.pipeline.parse_vtt": blocks,
                "modules.pipeline.get_audio_duration": 10,
                "modules.pipeline.split_audio": segments,
                "modules.pipeline.generate_tts": segments,
                "modules.pipeline.build_tts_handler_chain": chain,
                "modules.pipeline.align_tts_to_original": segments,
                "modules.pipeline.enforce_tts_budget": segments,
                "modules.pipeline.join_audio": None,
                "modules.pipeline.create_segment_builder": builder,
                "modules.audio_metadata.create_metadata_reader": Mock(),
                "modules.audio_metadata.write_audio_metadata": "metadata.json",
                "modules.high_quality_export.write_timeline_recipe": "recipe.json",
            }.items():
                mocks[target] = stack.enter_context(patch(target, return_value=value))
            stack.enter_context(patch("modules.timeline_render.render_initial_timeline", side_effect=render))
            with redirect_stdout(StringIO()):
                result = run_pipeline(str(source), "source.vtt", str(output), tts_volume=.5)
            for skipped in ("align_tts_to_original", "enforce_tts_budget", "join_audio"):
                mocks[f"modules.pipeline.{skipped}"].assert_not_called()
            builder.build.assert_not_called()
            self.assertEqual(result["high_quality_recipe"], "recipe.json")
            self.assertIn("00:00:05.000 --> 00:00:06.000", read_text(result["final_vtt"]))


if __name__ == "__main__":
    unittest.main()
