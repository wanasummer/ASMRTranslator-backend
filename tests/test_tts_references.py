"""Parent ASR audio is distinct from both subtitle speech and pink coverage."""
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from modules.audio_splitter import split_audio
from modules.tts.plugin_contract import TTSSynthesisRequest
from modules.tts.references import arrangement_segment, write_reference_profile


class ParentReferenceTests(unittest.TestCase):
    def test_processed_reference_keeps_original_coverage_and_changes_asset_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source, processed = work / "original.wav", work / "uvr.wav"
            for path, sample in ((source, b"\x01\x00"), (processed, b"\x02\x00")):
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((2, 2, 24000, 0, "NONE", "not compressed"))
                    audio.writeframes(sample * 2 * 48000)
            block = SimpleNamespace(index=0, start_ms=200, end_ms=500, pink_start_ms=0,
                pink_end_ms=2000, tts_delay_ms=200, max_allowed_tts_ms=1800, speaker="A",
                text="中文", source_text="日本語", skip_tts=False,
                aligned_sentence=SimpleNamespace(source_block_id="parent", source_start_ms=100, source_end_ms=1900))
            folder = work / "chunk_0/01_segments"
            with patch("modules.audio_splitter.get_audio_duration", return_value=2), \
                 patch("modules.audio_splitter._normalize_audio", side_effect=shutil.copyfile):
                segment = split_audio(str(source), [block], str(folder), include_asr_blocks=True,
                                      reference_audio_path=str(processed))[0]
                for field, sample in (("segment_file", b"\x01\x00"), ("speech_file", b"\x02\x00"),
                                      ("asr_block_file", b"\x02\x00")):
                    with wave.open(segment[field]) as audio:
                        self.assertEqual(audio.readframes(1), sample * 2)
                cleaned = TTSSynthesisRequest.from_segment(segment)
                self.assertEqual(cleaned.context["reference_audio_processing"], "processed")
                write_reference_profile(work, str(source), str(processed), "uvr-test")
                rebound = arrangement_segment(work, {"index": 0, "text": "编辑", **segment["tts_context"]})
                self.assertEqual(rebound["asr_block_file"], segment["asr_block_file"])
                self.assertEqual(rebound["tts_context"]["reference_audio_processing"], "processed")
                original = TTSSynthesisRequest.from_segment(split_audio(str(source), [block], str(folder),
                    include_asr_blocks=True)[0])
                self.assertNotEqual([a.sha256 for a in cleaned.assets if a.role == "source_asr_block"],
                                    [a.sha256 for a in original.assets if a.role == "source_asr_block"])
            self.assertFalse((folder / "_reference_24k_stereo.wav").exists())

    def test_native_index_cache_tracks_reference_content_changes(self):
        from modules.tts.indextts_adapter import IndexTTSAdapter
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.wav"
            path.write_bytes(b"original")
            adapter = IndexTTSAdapter(reference_audio=str(path), emotion_reference=str(path))
            before = adapter.cache_inputs()
            path.write_bytes(b"processed")
            after = adapter.cache_inputs()
            self.assertNotEqual(before["voice_reference_sha256"], after["voice_reference_sha256"])
            self.assertNotEqual(before["emotion_reference_sha256"], after["emotion_reference_sha256"])

    def test_reference_with_a_different_timeline_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frames in (("original.wav", 48000), ("processed.wav", 24000)):
                with wave.open(str(root / name), "wb") as audio:
                    audio.setparams((2, 2, 24000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\0" * frames * 4)
            with patch("modules.audio_splitter.get_audio_duration", return_value=2), \
                 patch("modules.audio_splitter._normalize_audio", side_effect=shutil.copyfile):
                with self.assertRaisesRegex(ValueError, "timeline"):
                    split_audio(str(root / "original.wav"), [], str(root / "slices"),
                                reference_audio_path=str(root / "processed.wav"))
            self.assertFalse((root / "slices/_source_24k_stereo.wav").exists())
            self.assertFalse((root / "slices/_reference_24k_stereo.wav").exists())

    def test_siblings_share_parent_reference_without_changing_speech_or_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "source.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setparams((2, 2, 24000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x01\x00" * 2 * 48000)
            aligned = SimpleNamespace(source_block_id="parent", source_start_ms=100, source_end_ms=1900)
            blocks = [SimpleNamespace(index=i, start_ms=start, end_ms=end, pink_start_ms=start,
                pink_end_ms=end, tts_delay_ms=0, max_allowed_tts_ms=end-start, speaker="A", text="中文",
                source_text="日本語", skip_tts=False, aligned_sentence=aligned)
                for i, start, end in [(0, 200, 500), (1, 1000, 1200)]]
            folder = work / "chunk_0/01_segments"
            with patch("modules.audio_splitter.get_audio_duration", return_value=2), \
                 patch("modules.audio_splitter._normalize_audio", side_effect=shutil.copyfile):
                rows = split_audio(str(source), blocks, str(folder), include_asr_blocks=True)
            self.assertEqual(rows[0]["asr_block_file"], rows[1]["asr_block_file"])
            request = TTSSynthesisRequest.from_segment(rows[0])
            assets = {asset.role: asset for asset in request.assets}
            self.assertEqual(assets["source_speech"].duration_ms, 300)
            self.assertEqual(assets["source_asr_block"].duration_ms, 1800)
            self.assertEqual(request.context["source_block_id"], "parent")
            row = {"index": 0, "text": "编辑", **rows[0]["tts_context"]}
            rebound = arrangement_segment(work, row)
            self.assertEqual(rebound["asr_block_file"], rows[0]["asr_block_file"])
            self.assertNotIn("asr_block_file", row)


if __name__ == "__main__":
    unittest.main()
