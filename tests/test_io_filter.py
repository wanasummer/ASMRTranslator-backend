"""Audio filesystem writes use one layout in source and packaged execution."""
from pathlib import Path
import tempfile
import unittest
from modules.io import intermediate_dir, open_write, write_bytes


class AudioIoTests(unittest.TestCase):
    def test_audio_writers_create_parents_and_close_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(intermediate_dir(directory, 'segments'))/'clip.wav'
            write_bytes(str(path), b'first')
            with open_write(str(path), 'ab') as output:
                output.write(b'second')
            self.assertTrue(output.closed)
            self.assertEqual(path.read_bytes(), b'firstsecond')
