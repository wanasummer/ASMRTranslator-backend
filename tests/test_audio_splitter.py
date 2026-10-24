import os
import shutil
import struct
import tempfile
import unittest
import wave
from unittest import mock

from modules.audio_splitter import split_audio
from modules.vtt_parser import VTTBlock
from modules.tts.indextts_adapter import IndexTTSAdapter
from modules.tts_engine import generate_tts


class AudioSplitterTest(unittest.TestCase):
    def test_speech_reference_excludes_pink_padding_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "source.wav")
            with wave.open(path, "wb") as target:
                target.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
                target.writeframes(struct.pack("<1000h", *([10] * 250 + [100] * 250 + [20] * 500)))
            block = VTTBlock(42, 250, 500, "你好", speaker="角色", pink_start_ms=0,
                             pink_end_ms=750, tts_delay_ms=250, max_allowed_tts_ms=500)
            with mock.patch("modules.audio_splitter.get_audio_duration", return_value=1), \
                 mock.patch("modules.audio_splitter._normalize_audio", side_effect=shutil.copyfile):
                segments = split_audio(path, [block], os.path.join(directory, "segments"))
            segment = segments[0]
            self.assertEqual(segment["speaker"], "角色")
            self.assertEqual(segment["original_start_ms"], 250)
            self.assertEqual(segment["tts_delay_ms"], 250)
            with wave.open(segment["speech_file"], "rb") as source:
                self.assertEqual(source.getnframes(), 250)
                self.assertEqual(struct.unpack("<250h", source.readframes(250)), (100,) * 250)
            with wave.open(segment["segment_file"], "rb") as source:
                self.assertEqual(source.getnframes(), 750)
            references = []
            def synthesize(adapter, text, output, rate):
                references.append((adapter._reference_audio, adapter._emotion_reference))
                with open(output, "wb") as target:
                    target.write(b"ID3" + b"x" * 300)
                return output
            with mock.patch.object(IndexTTSAdapter, "_synthesize_blocking", autospec=True, side_effect=synthesize):
                result = generate_tts(segments, os.path.join(directory, "tts"), backend="indextts-2.5",
                                      backend_kwargs={"options": {}})
            self.assertEqual(references, [(segment["speech_file"], None)])
            self.assertEqual(result[0]["text"], "你好")
            self.assertEqual(result[0]["speaker"], "角色")


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

class AudioSplitterContractTest(unittest.TestCase):
    def test_speech_and_full_slice_keep_distinct_boundaries_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(2); audio.setsampwidth(2); audio.setframerate(24000)
                audio.writeframes(b"\0" * 24000 * 4)
            block = SimpleNamespace(index=42, start_ms=100, end_ms=400, pink_start_ms=0,
                pink_end_ms=1000, tts_delay_ms=100, max_allowed_tts_ms=900, speaker="A",
                text="中文", source_text="日本語", skip_tts=False)
            with patch("modules.audio_splitter.get_audio_duration", return_value=1), \
                 patch("modules.audio_splitter._normalize_audio", side_effect=shutil.copyfile):
                result = split_audio(str(source), [block], str(Path(folder) / "slices"))
            segment = result[0]
            self.assertEqual(segment["segment_id"], "42")
            self.assertEqual(segment["source_text"], "日本語")
            for key, frames in (("speech_file", 7200), ("segment_file", 24000)):
                with wave.open(segment[key], "rb") as audio:
                    self.assertEqual(audio.getnframes(), frames)
                    self.assertEqual(audio.getnchannels(), 2)
            self.assertFalse((Path(folder) / "slices/_source_24k_stereo.wav").exists())


if __name__ == "__main__":
    unittest.main()
