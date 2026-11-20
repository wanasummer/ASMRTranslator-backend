from modules.storage import initialize_project, read_text, read_bytes, read_json, save_text, save_json, contains, address
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.high_quality_export import (
    AudioProfile, HighQualityExportError, _append_pcm, load_recipe, probe_audio,
    render_high_quality, write_export_recipe, write_timeline_recipe,
)
from modules.mode_builders import create_segment_builder


class RecipeTests(unittest.TestCase):
    def test_moved_project_rebinds_internal_source_and_keeps_verified_tts(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "before"
            work = project / "generated/assets/track"
            work.mkdir(parents=True)
            source = project / "source.wav"
            tts = work / "voice.wav"
            source.write_bytes(b"source")
            tts.write_bytes(b"voice")
            write_export_recipe(str(source), [{"index":0,"start_ms":0,"end_ms":1000,"tts_file":str(tts)}], {}, str(work), 0, 0)
            moved = Path(directory) / "after"
            shutil.move(str(project),str(moved))
            recipe, restored = load_recipe(str(moved/"generated/assets/track/debug/high_quality_export.json"))
            self.assertEqual(restored, moved/"generated/assets/track")
            self.assertEqual(recipe["source"]["path"], str(moved/"source.wav"))

    def test_recipe_is_independent_snapshot_with_verified_tts_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, tts = root / "original.wav", root / "tts.wav"
            source.write_bytes(b"original")
            tts.write_bytes(b"tts")
            segments = [{"index": 8, "start_ms": 0, "end_ms": 1000, "tts_file": str(tts)}]
            settings = create_segment_builder(retain_original=True).export_settings()
            path = write_export_recipe(str(source), segments, settings, directory, 10, 3000)
            recipe, work = load_recipe(path)
            self.assertEqual(work, root.resolve())
            self.assertEqual(recipe["segments"][0]["tts_file"], "tts.wav")
            self.assertEqual(segments[0]["tts_file"], str(tts))
            tts.write_bytes(b"different")
            with self.assertRaises(HighQualityExportError):
                load_recipe(path)

    def test_source_changes_and_cache_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.wav"
            source.write_bytes(b"original")
            path = write_export_recipe(str(source), [{"index": 0, "start_ms": 0,
                "end_ms": 1000, "skip_tts": True}], {}, directory, 0, 3000)
            source.write_bytes(b"new source")
            with self.assertRaises(HighQualityExportError):
                load_recipe(path)
            source.write_bytes(b"original")
            data = json.loads(read_text(path))
            data["inputs"] = {"../outside.wav": "fake"}
            save_text(path, json.dumps(data))
            with self.assertRaises(HighQualityExportError):
                load_recipe(path)

    def test_all_modes_have_roundtrippable_factory_settings(self):
        for options in ({}, {"retain_original": True},
                        {"retain_original": True, "sync_mode": "follow_translation"},
                        {"phantom_mode": True, "phantom_orig_fade_ms": 250,
                         "phantom_virtual_fade": True, "sync_mode": "follow_translation"}):
            with self.subTest(options=options):
                first = create_segment_builder(**options)
                restored = create_segment_builder(**first.export_settings())
                self.assertEqual(first.name, restored.name)
                self.assertEqual(first.export_settings(), restored.export_settings())


class NativeAudioTests(unittest.TestCase):
    def test_vbr_uses_next_supported_rate_without_reducing_quality(self):
        self.assertEqual(AudioProfile(44100, 2, 16, "mp3", 264123).mp3_bitrate, 320000)
        self.assertEqual(AudioProfile(48000, 2, 16, "mp3", 320000).mp3_bitrate, 320000)
        self.assertEqual(AudioProfile(24000, 2, 16, "mp3", 130000).mp3_bitrate, 144000)

    def test_profile_uses_audio_stream_not_container_bitrate(self):
        with patch("modules.high_quality_export._command", return_value=json.dumps({"streams": [{
            "sample_rate": "48000", "channels": 2, "codec_name": "mp3", "bit_rate": "320000"
        }]}).encode()):
            profile = probe_audio("original.mp3")
        self.assertEqual((profile.sample_rate, profile.channels, profile.bitrate), (48000, 2, 320000))

    def test_streaming_join_preserves_stereo_and_overlap_duration(self):
        output = io.BytesIO()
        first = np.array([[100, -100]] * 6, dtype="<i4")
        second = np.array([[50, -50]] * 4, dtype="<i4")
        _append_pcm(output, first, 0, 0, 2)
        count = _append_pcm(output, second, 0, 2, 2)
        self.assertEqual(count, 8)
        result = np.frombuffer(output.getvalue(), dtype="<i4").reshape(-1, 2)
        np.testing.assert_array_equal(result[:4], first[:4])
        np.testing.assert_array_equal(result[4:6], [[150, -150], [150, -150]])
        np.testing.assert_array_equal(result[6:], second[2:])

    def test_native_render_encodes_original_mp3_bitrate_without_calling_tts(self):
        profile = AudioProfile(48000, 2, 16, "mp3", 320000)
        recipe = {"source": {"path": "original.mp3"}, "settings": {}, "crossfadeMs": 0,
                  "overhangMs": 3000}
        builder = Mock()
        builder.build.return_value = [{"tts_file": "tts.wav", "vc_file": "tts.wav"}]
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output.mp3"
            def command(args):
                commands.append(args)
                if args[-1] == "-":
                    return np.zeros((4800, 2), dtype="<i4").tobytes()
                target.write_bytes(b"encoded")
                return b""
            with (patch("modules.high_quality_export.load_recipe", return_value=(recipe, Path(directory))),
                  patch("modules.high_quality_export.probe_audio", return_value=profile),
                  patch("modules.high_quality_export._native_slices", return_value=[]),
                  patch("modules.mode_builders.create_segment_builder", return_value=builder),
                  patch("modules.high_quality_export._command", side_effect=command),
                  patch("modules.audio_metadata.create_metadata_reader"),
                  patch("modules.audio_metadata.write_audio_metadata")):
                render_high_quality("recipe.json", str(target))
            self.assertIn("320000", commands[-1])
            self.assertNotIn("192k", commands[-1])
            self.assertEqual(builder.wav_params, profile.wav_params)

    def test_timeline_recipe_rerenders_the_same_arrangement_plan_natively(self):
        from modules.arrangement import ArrangedClip, ArrangementPlan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, voice = root / "original.wav", root / "chunk_0/02_tts/voice_0000.wav"
            voice.parent.mkdir(parents=True)
            source.write_bytes(b"original")
            voice.write_bytes(b"voice")
            plan = ArrangementPlan(source, (ArrangedClip(7, 0, 2000, "你好", voice, .85, 0, 1000, 0, 2300),), False)
            path = write_timeline_recipe(str(source), plan, directory)
            recipe, work = load_recipe(path)
            self.assertEqual((recipe["contractVersion"], recipe["renderer"]), (2, "arrangement"))
            self.assertEqual(recipe["clips"][0]["path"], "chunk_0/02_tts/voice_0000.wav")
            adapter = Mock()
            target = root / "output.wav"
            commands = []
            def command(args):
                commands.append(args)
                target.write_bytes(b"encoded")
                return b""
            with (patch("modules.arrangement.create_arrangement_audio_adapter", return_value=adapter),
                  patch("modules.high_quality_export.probe_audio",
                        return_value=AudioProfile(44100, 2, 16, "pcm_s16le", 0)),
                  patch("modules.high_quality_export._command", side_effect=command),
                  patch("modules.audio_metadata.create_metadata_reader"),
                  patch("modules.audio_metadata.write_audio_metadata")):
                render_high_quality(path, str(target))
            rendered = adapter.render.call_args.args[0]
            self.assertEqual(rendered.clips[0], ArrangedClip(7, 0, 2000, "你好", work / "chunk_0/02_tts/voice_0000.wav",
                                                             .85, 0, 1000, 0, 2300))
            self.assertIn("pcm_s16le", commands[-1])
            voice.write_bytes(b"changed")
            with self.assertRaises(HighQualityExportError):
                load_recipe(path)


if __name__ == "__main__":
    unittest.main()
