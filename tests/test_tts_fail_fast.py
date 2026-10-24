import tempfile
import unittest
from unittest import mock

from modules.error import PipelineError
from modules.tts_engine import generate_tts


class _FailingAdapter:
    name = "test-tts"
    output_extension = ".mp3"
    def get_voice_for_speaker(self, _speaker): return "voice"
    def prepare_text(self, text): return text
    async def synthesize(self, *_args, **_kwargs): raise RuntimeError("service failed")


class TtsFailFastTest(unittest.TestCase):
    def test_failed_segment_does_not_create_silence(self):
        segment = {"index": 0, "speaker": None, "text": "你好"}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("modules.tts_engine._get_adapter", return_value=_FailingAdapter()):
            with self.assertRaises(PipelineError) as caught:
                generate_tts([segment], directory, max_concurrent=1)
            self.assertEqual(caught.exception.descriptor.code, "TTS_SEGMENT_FAILED")
            self.assertNotIn("tts_file", segment)


if __name__ == "__main__": unittest.main()
