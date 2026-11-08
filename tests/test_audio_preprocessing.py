import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave
import threading

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

    def exercise(self, returned_frames, *, progress_supported=False, wrong_identity=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "source.wav"
        source.write_bytes(wav())
        calls = []
        events = []
        polled = threading.Event()
        def handle(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"service": "uvr", "api_version": "uvr/1", "ready": True,
                                               "version": "0.2.0", "model_sha256": "model",
                                               "capabilities": ["progress/1"] if progress_supported else []})
            if request.url.path == "/progress":
                polled.set()
                return httpx.Response(200, json={"service": "uvr", "api_version": "uvr/1",
                    "request_id": "wrong" if wrong_identity else request.url.params["request_id"],
                    "status": "running", "current": 2205, "total": 4410})
            calls.append(request)
            body = request.read()
            # Multipart request ID is the only ASCII 32-character field in this fixture.
            import re
            request_id = re.search(rb'name="request_id"\r\n\r\n([0-9a-f]{32})', body)[1].decode()
            if progress_supported:
                self.assertTrue(polled.wait(2))
                # Wait for the progress callback, rather than just the HTTP request.
                self.assertTrue(reported.wait(2))
            return httpx.Response(200, content=wav(returned_frames), headers={"X-Request-ID": request_id, "X-UVR-API": "uvr/1"})
        reported = threading.Event()
        def report(*event):
            events.append(event)
            if event[2] == 4410 or "暂时无法获取" in event[0]:
                reported.set()
        adapter = UvrAudio("http://uvr.test", client_factory=lambda **kwargs: httpx.Client(transport=httpx.MockTransport(handle)),
                           progress=report, poll_interval=.01)
        def normalize(command, **kwargs):
            shutil.copyfile(source, command[-1])
        with patch("modules.audio_preprocessing.subprocess.run", side_effect=normalize):
            result = adapter.prepare(str(source), root / "generated")
            self.assertNotEqual(result.path, str(source))
            adapter.prepare(str(source), root / "generated")
        self.assertEqual(len(calls), 1)
        self.assertEqual(source.read_bytes(), wav())
        if progress_supported:
            measured = [event for event in events if event[1:] == (2205, 4410)]
            self.assertEqual(bool(measured), not wrong_identity)
        self.assertEqual(events[-1], ("已复用 UVR 人声缓存", 1, 1))

    def test_valid_audio_is_cached_and_original_is_untouched(self):
        self.exercise(4410)

    def test_timeline_change_is_rejected(self):
        with self.assertRaisesRegex(AudioPreparationError, "timeline"):
            self.exercise(4409)

    def test_measured_progress_is_forwarded_while_post_is_waiting(self):
        self.exercise(4410, progress_supported=True)

    def test_progress_from_another_request_is_not_forwarded(self):
        self.exercise(4410, progress_supported=True, wrong_identity=True)


if __name__ == "__main__":
    unittest.main()
