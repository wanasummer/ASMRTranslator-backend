"""LRC encoding compatibility and the pipeline's subtitle input boundary."""

import codecs
from pathlib import Path
import tempfile
import unittest

from modules.subtitles.factory import parse_subtitle
from modules.vtt_parser import parse_vtt
from modules.text_decoder import TextDecodingError
from modules.error import ConfigurationError


class LRCEncodingTests(unittest.TestCase):
    TEXT = "[ti:测试]\r\n[00:01.00]遥：こんにちは，你好\r\n[00:02.50]第二句\r\n"

    def test_default_encodings_preserve_blocks_and_source_bytes(self):
        samples = {
            "utf8": self.TEXT.encode("utf-8"),
            "utf8_bom": self.TEXT.encode("utf-8-sig"),
            "utf16_le": codecs.BOM_UTF16_LE + self.TEXT.encode("utf-16-le"),
            "utf16_be": codecs.BOM_UTF16_BE + self.TEXT.encode("utf-16-be"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "字幕.lrc"
            for name, data in samples.items():
                with self.subTest(encoding=name):
                    path.write_bytes(data)
                    document = parse_subtitle(str(path))
                    self.assertEqual(document.format_name, "lrc")
                    self.assertEqual([
                        (b.index, b.start_ms, b.end_ms, b.text, b.speaker, b.raw_text)
                        for b in document.blocks
                    ], [
                        (0, 1000, 2499, "こんにちは，你好", "遥", "遥：こんにちは，你好"),
                        (1, 2500, 5500, "第二句", None, "第二句"),
                    ])
                    self.assertEqual(path.read_bytes(), data)

    def test_pipeline_parser_accepts_utf16_lrc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "字幕.lrc"
            path.write_bytes(codecs.BOM_UTF16_BE + self.TEXT.encode("utf-16-be"))
            blocks = parse_vtt(str(path))
            self.assertEqual([(b.index, b.start_ms, b.end_ms, b.text, b.speaker)
                              for b in blocks], [
                (0, 1000, 2499, "こんにちは，你好", "遥"),
                (1, 2500, 5500, "第二句", None),
            ])

    def test_explicit_encoding_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "字幕.lrc"
            path.write_bytes("[00:01.00]你好世界\n".encode("gb18030"))
            document = parse_subtitle(str(path), encoding="gb18030")
            self.assertEqual(document.blocks[0].text, "你好世界")

    def test_malformed_text_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "字幕.lrc"
            for data, encoding in ((b"[00:01.00]\xff", "utf-8"),
                                   (codecs.BOM_UTF16_LE + b"\x5b", "auto")):
                with self.subTest(data=data):
                    path.write_bytes(data)
                    with self.assertRaises(ConfigurationError) as caught:
                        parse_subtitle(str(path), encoding=encoding)
                    self.assertIsInstance(caught.exception.cause, TextDecodingError)


if __name__ == "__main__":
    unittest.main()
