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
from modules.tts.references import arrangement_segment


class ParentReferenceTests(unittest.TestCase):
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
