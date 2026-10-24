"""Tests for modules.io — dev/build share one project-local output layout.

Run from the project root:  python -m unittest tests.test_io_filter
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.io import (
    _NullWriter,
    intermediate_dir,
    is_allowed_output,
    open_write,
    write_bytes,
    write_json,
    write_text,
)


class EnvTest(unittest.TestCase):
    def test_dev_allows_everything(self):
        # sys.frozen 不存在（源码运行）→ 全放行
        self.assertTrue(is_allowed_output(r"C:\any\where\chunk_0\01.wav"))
        self.assertTrue(is_allowed_output(r"C:\any\where\output.mp3"))

    def test_build_allows_final_outputs(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            for name in ("output.mp3", "output.wav", "OUTPUT.MP3",
                         "pipeline.log", "pipeline_2026.log"):
                p = os.path.join(r"C:\out", name)
                self.assertTrue(is_allowed_output(p), name)

    def test_build_allows_intermediates(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            for p in (
                r"C:\out\chunk_0\01_segments\seg.wav",
                r"C:\out\llm_subtitles\optimized.vtt",
                r"C:\out\vad_output\final_slices.json",
                r"C:\out\_parsed_vtt.json",
                r"C:\out\asr_debug.json",
            ):
                self.assertTrue(is_allowed_output(p), p)

    def test_build_allows_temp_and_config(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            tmp = tempfile.gettempdir()
            self.assertTrue(is_allowed_output(os.path.join(tmp, "work", "a.wav")))
            home_cfg = os.path.join(os.path.expanduser("~"), ".asmrts")
            self.assertTrue(is_allowed_output(home_cfg))


class WriteEntryTest(unittest.TestCase):
    def _tmp(self):
        return tempfile.mkdtemp()

    def test_write_json_creates_parents(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "ok", "out", "data.json")
            write_json(p, {"a": 1})
            with open(p, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"a": 1})

    def test_write_json_intermediate_is_written(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "chunk_0", "data.json")
            write_json(p, {"a": 1})
            with open(p, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"a": 1})

    def test_write_text_intermediate_is_written(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "asr_subtitles", "x.vtt")
            write_text(p, "WEBVTT")
            with open(p, encoding="utf-8") as f:
                self.assertEqual(f.read(), "WEBVTT")

    def test_write_bytes_output_allowed_in_build(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "output.mp3")
            write_bytes(p, b"ID3")
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"ID3")

    def test_open_write_intermediate_is_written(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "chunk_0", "tts", "a.mp3")
            with open_write(p, "wb") as f:
                f.write(b"12345")
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"12345")

    def test_open_write_streaming_allowed(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = self._tmp()
            p = os.path.join(d, "output.wav")
            with open_write(p, "wb") as f:
                f.write(b"RIFF")
            with open(p, "rb") as f:
                self.assertEqual(f.read(), b"RIFF")

    def test_null_writer_discards(self):
        w = _NullWriter()
        self.assertEqual(w.write(b"abc"), 3)
        self.assertEqual(w.write("text"), 4)


class IntermediateDirTest(unittest.TestCase):
    def test_dev_under_output_dir(self):
        d = tempfile.mkdtemp()  # sys.frozen 不存在 = dev
        sub = intermediate_dir(d, "chunk_0")
        self.assertTrue(sub.startswith(d))
        self.assertTrue(os.path.isdir(sub))

    def test_build_under_output_dir(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = tempfile.mkdtemp()
            sub = intermediate_dir(d, "chunk_0")
            self.assertEqual(sub, os.path.join(d, "chunk_0"))
            self.assertTrue(os.path.isdir(sub))
            self.assertEqual(os.listdir(d), ["chunk_0"])

    def test_build_intermediates_share_project_parent(self):
        with mock.patch("modules.io.sys.frozen", True, create=True):
            d = tempfile.mkdtemp()
            a = intermediate_dir(d, "chunk_0")
            b = intermediate_dir(d, "chunk_1")
            self.assertEqual(os.path.dirname(a), d)
            self.assertEqual(os.path.dirname(b), d)
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
