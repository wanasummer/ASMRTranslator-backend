import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import httpx
from modules.audio_preprocessing import AudioPreparationError, UvrAudio, create_audio_preprocessor


def wav(frames=4410):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * frames * 4)
    return buffer.getvalue()


class AnalysisAudioTests(unittest.TestCase):
    def test_original_does_not_contact_a_service(self):
        self.assertEqual(create_audio_preprocessor("original").prepare("source.wav", Path("output")).path, "source.wav")

    def exercise(self, returned_frames):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "source.wav"
        source.write_bytes(wav())
        calls = []
        def handle(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"service": "uvr", "api_version": "uvr/1", "ready": True,
                                               "version": "0.1.0", "model_sha256": "model"})
            calls.append(request)
            body = request.read()
            # Multipart request ID is the only ASCII 32-character field in this fixture.
            import re
            request_id = re.search(rb'name="request_id"\r\n\r\n([0-9a-f]{32})', body)[1].decode()
            return httpx.Response(200, content=wav(returned_frames), headers={"X-Request-ID": request_id, "X-UVR-API": "uvr/1"})
        adapter = UvrAudio("http://uvr.test", client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle)))
        def normalize(command, **kwargs):
            shutil.copyfile(source, command[-1])
        with patch("modules.audio_preprocessing.subprocess.run", side_effect=normalize):
            result = adapter.prepare(str(source), root / "generated")
            self.assertNotEqual(result.path, str(source))
            adapter.prepare(str(source), root / "generated")
        self.assertEqual(len(calls), 1)
        self.assertEqual(source.read_bytes(), wav())

    def test_valid_audio_is_cached_and_original_is_untouched(self):
        self.exercise(4410)

    def test_timeline_change_is_rejected(self):
        with self.assertRaisesRegex(AudioPreparationError, "timeline"):
            self.exercise(4409)


if __name__ == "__main__":
    unittest.main()
