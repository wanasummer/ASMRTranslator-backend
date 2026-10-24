import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.high_quality_export import (
    AudioProfile, HighQualityExportError, _append_pcm, load_recipe, probe_audio,
    render_high_quality, write_export_recipe,
)
from modules.mode_builders import create_segment_builder


class RecipeTests(unittest.TestCase):
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
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data["inputs"] = {"../outside.wav": "fake"}
            Path(path).write_text(json.dumps(data), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
