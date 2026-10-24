"""Subtitle-only runs preserve cues and stop before every audio/TTS stage."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.pipeline import run_pipeline


class SubtitlesOnlyPipelineTest(unittest.TestCase):
    def test_failed_subtitle_run_cannot_leave_a_previous_exportable_subtitle(self):
        from modules.error import PipelineError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, vtt = root / "source.wav", root / "source.vtt"
            source.write_bytes(b"audio")
            vtt.write_text("WEBVTT\n", encoding="utf-8")
            output = root / "work"
            (output / "llm_subtitles").mkdir(parents=True)
            final = output / "llm_subtitles" / "final.vtt"
            final.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n旧字幕\n", encoding="utf-8")
            with redirect_stdout(StringIO()), self.assertRaises(PipelineError) as error:
                run_pipeline(str(source), str(vtt), str(output), tts_backend="subtitles-only")
            self.assertEqual(error.exception.descriptor.code, "SUBTITLE_FINAL_EMPTY")
            self.assertFalse(final.exists())
            self.assertFalse((root / "run-result.json").exists())

    def test_only_subtitle_media_is_produced_and_old_audio_is_removed(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source = root / "source.wav"
            source.write_bytes(b"source audio")
            vtt = root / "source.vtt"
            vtt.write_text("WEBVTT\n\n1\n00:00:00.010 --> 00:00:00.500\n嗯\n\n"
                           "2\n00:00:00.800 --> 00:00:01.200\n好\n", encoding="utf-8")
            output = root / "work"
            output.mkdir()
            (output / "output.mp3").write_bytes(b"old audio")
            (output / "output.wav").write_bytes(b"old audio")
            forbidden = [
                "modules.pipeline.split_audio", "modules.pipeline.generate_tts",
                "modules.pipeline.build_tts_handler_chain", "modules.pipeline.create_segment_builder",
                "modules.pipeline.align_tts_to_original", "modules.pipeline.join_audio",
                "modules.pipeline.get_audio_duration", "modules.pipeline.mark_interjection_skips",
                "modules.audio_metadata.create_metadata_reader", "modules.audio_metadata.write_audio_metadata",
            ]
            mocked = [stack.enter_context(patch(target)) for target in forbidden]
            with redirect_stdout(StringIO()):
                result = run_pipeline(str(source), str(vtt), str(output), tts_backend="subtitles-only",
                                      phantom_mode=True, sync_mode="follow_translation")
            for function in mocked:
                function.assert_not_called()
            self.assertNotIn("output_mp3", result)
            self.assertNotIn("output_wav", result)
            content = Path(result["final_vtt"]).read_text(encoding="utf-8")
            self.assertIn("嗯", content)
            self.assertIn("好", content)
            self.assertIn("00:00:00.010 --> 00:00:00.500", content)
            media = [path.suffix for path in output.rglob("*") if path.is_file()
                     and path.suffix in {".mp3", ".wav", ".vtt"}]
            self.assertEqual(media, [".vtt"])
            record = json.loads(Path(result["run_result"]).read_text(encoding="utf-8"))["runs"]["work"]
            self.assertEqual(record["status"], "succeeded")
            self.assertEqual(record["outputMode"], "subtitles-only")
            self.assertEqual(record["files"], {"finalVtt": "llm_subtitles/final.vtt"})

    def test_cli_accepts_subtitles_only_and_emits_no_audio_paths(self):
        import main
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            source.write_bytes(b"audio")
            subtitle = Path(directory) / "source.vtt"
            subtitle.write_text("WEBVTT\n", encoding="utf-8")
            output = StringIO()
            result = {"final_vtt": "final.vtt", "log_path": "run.log", "run_result": "run-result.json"}
            with patch("sys.argv", ["main.py", str(source), str(subtitle), "--tts-backend", "subtitles-only",
                                     "--tts-backend-kwargs", "invalid-unused-json"]), \
                    patch("main.run_pipeline", return_value=result) as pipeline, redirect_stdout(output):
                main.main()
            self.assertEqual(pipeline.call_args.kwargs["tts_backend"], "subtitles-only")
            frame = next(line for line in output.getvalue().splitlines() if line.startswith("[PIPELINE_RESULT]"))
            files = json.loads(frame[len("[PIPELINE_RESULT]"):])["files"]
            self.assertEqual(files, {"finalVtt": "final.vtt", "log": "run.log"})


if __name__ == "__main__":
    unittest.main()
