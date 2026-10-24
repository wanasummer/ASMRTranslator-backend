"""Encoding contracts and integration across all subtitle import adapters."""

import codecs
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from modules.error import ConfigurationError
from modules.subtitles.factory import parse_subtitle
from modules.subtitles.lrc_adapter import LRCAdapter
from modules.subtitles.srt_adapter import SRTAdapter
from modules.subtitles.vtt_adapter import VTTAdapter
from modules.text_decoder import TextDecodingError, decode_text, read_text


LEGACY_TEXTS = {
    "gb18030": "这是一个用于测试字幕编码自动识别的中文句子。今天的天气很好，我们一起出去散步吧。",
    "big5": "這是一個用於測試字幕編碼自動識別的中文句子。今天的天氣很好，我們一起出去散步吧。",
    "cp932": "こんにちは、今日はとてもいい天気ですね。これは字幕の文字コードを確認するための文章です。",
    "euc_jp": "こんにちは、今日はとてもいい天気ですね。これは字幕の文字コードを確認するための文章です。",
    "iso2022_jp": "こんにちは、今日はとてもいい天気ですね。これは字幕の文字コードを確認するための文章です。",
    "euc_kr": "안녕하세요 오늘은 날씨가 아주 좋습니다. 이것은 자막 문자 인코딩을 확인하기 위한 문장입니다.",
    "cp1252": "Bonjour, déjà très agréable ! Une élève française préfère écouter la musique à Noël.",
    "cp1251": "Здравствуйте, сегодня прекрасная погода. Это предложение для проверки кодировки субтитров.",
}


class TextDecoderTests(unittest.TestCase):
    def test_unicode_variants_produce_identical_utf8(self):
        text = "[00:01.00]中文，日本語，한국어，🎵\r\n" * 5
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-32",
                         "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"):
            with self.subTest(encoding=encoding):
                result = decode_text(text.encode(encoding))
                self.assertEqual(result.text, text.replace("\r\n", "\n"))
                self.assertEqual(result.utf8_bytes, result.text.encode("utf-8"))
                self.assertFalse(result.utf8_bytes.startswith(codecs.BOM_UTF8))

    def test_legacy_encodings(self):
        for encoding, text in LEGACY_TEXTS.items():
            with self.subTest(encoding=encoding):
                self.assertEqual(decode_text(text.encode(encoding)).text, text)

    def test_explicit_encoding_overrides_ambiguous_detection(self):
        data = "你好".encode("gb18030")
        with patch("modules.text_decoder.chardet.detect") as detector:
            self.assertEqual(decode_text(data, "gb18030").text, "你好")
            detector.assert_not_called()
        with self.assertRaises(TextDecodingError):
            decode_text(data, "utf-8")
        with self.assertRaises(TextDecodingError):
            decode_text(data, "unknown-review-codec")

    def test_bom_decoding_never_falls_back_on_truncated_input(self):
        with patch("modules.text_decoder.chardet.detect") as detector:
            for data in (codecs.BOM_UTF16_LE + b"x", codecs.BOM_UTF32_BE + b"x",
                         codecs.BOM_UTF8 + b"\xff"):
                with self.subTest(data=data), self.assertRaises(TextDecodingError):
                    decode_text(data)
            detector.assert_not_called()

    def test_low_confidence_or_binary_input_is_rejected(self):
        for guess in ({"encoding": None, "confidence": 0},
                      {"encoding": "cp1252", "confidence": 0.4}):
            with patch("modules.text_decoder.chardet.detect", return_value=guess):
                with self.assertRaises(TextDecodingError):
                    decode_text(b"\xff\x81")
        with self.assertRaises(TextDecodingError):
            decode_text(b"\x00\x01\x02\x03\x04")

    def test_empty_ascii_and_universal_newlines(self):
        self.assertEqual(decode_text(b"").utf8_bytes, b"")
        self.assertEqual(decode_text(b"one\rtwo\r\nthree\n").text, "one\ntwo\nthree\n")

    def test_reader_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            original = "你好世界".encode("utf-16")
            path.write_bytes(original)
            self.assertEqual(read_text(str(path)).utf8_bytes, "你好世界".encode("utf-8"))
            self.assertEqual(path.read_bytes(), original)


class SubtitleEncodingIntegrationTests(unittest.TestCase):
    def test_import_decodes_once_then_passes_only_text_to_adapter(self):
        text = "[00:01.00]你好世界\n"
        adapter = LRCAdapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.lrc"
            path.write_bytes(text.encode("utf-16"))
            with (patch("modules.subtitles.factory.read_text", wraps=read_text) as reader,
                  patch("modules.subtitles.factory.get_adapter", return_value=adapter),
                  patch.object(adapter, "parse", side_effect=AssertionError("file API called")),
                  patch.object(adapter, "parse_text", wraps=adapter.parse_text) as parser):
                document = parse_subtitle(str(path))
            reader.assert_called_once_with(str(path), "auto")
            parser.assert_called_once_with(text)
            self.assertEqual(document.blocks[0].text, "你好世界")

    def test_adapters_parse_text_without_file_io_or_detection(self):
        samples = (
            (LRCAdapter(), "[00:01.00]你好世界\n"),
            (SRTAdapter(), "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n"),
            (VTTAdapter(), "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n你好世界\n"),
        )
        with (patch("builtins.open", side_effect=AssertionError("unexpected I/O")),
              patch("modules.text_decoder.chardet.detect", side_effect=AssertionError("unexpected detection"))):
            for adapter, content in samples:
                with self.subTest(format=adapter.format_name):
                    self.assertEqual(adapter.parse_text(content).blocks[0].text, "你好世界")

    def test_direct_adapter_file_api_defaults_to_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.lrc"
            path.write_bytes("[00:01.00]你好世界\n".encode("utf-8"))
            with patch("modules.text_decoder.chardet.detect", side_effect=AssertionError("unexpected detection")):
                self.assertEqual(LRCAdapter().parse(str(path)).blocks[0].text, "你好世界")

    def test_formats_share_decoding_and_preserve_timeline(self):
        templates = {
            "lrc": "[00:01.00]{text}\r\n[00:02.50]END\r\n",
            "srt": "1\r\n00:00:01,000 --> 00:00:02,499\r\n{text}\r\n\r\n2\r\n00:00:02,500 --> 00:00:05,500\r\nEND\r\n",
            "vtt": "WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.499\r\n{text}\r\n\r\n00:00:02.500 --> 00:00:05.500\r\nEND\r\n",
        }
        samples = {**LEGACY_TEXTS, **{encoding: "你好，日本語，🎵" for encoding in
                   ("utf-8", "utf-8-sig", "utf-16", "utf-32", "utf-16-le", "utf-16-be",
                    "utf-32-le", "utf-32-be")}}
        with tempfile.TemporaryDirectory() as directory:
            for extension, template in templates.items():
                for encoding, text in samples.items():
                    with self.subTest(format=extension, encoding=encoding):
                        path = Path(directory) / ("字幕." + extension)
                        original = template.format(text=text).encode(encoding)
                        path.write_bytes(original)
                        document = parse_subtitle(str(path))
                        self.assertEqual([(b.index, b.start_ms, b.end_ms, b.text)
                                          for b in document.blocks],
                                         [(0, 1000, 2499, text), (1, 2500, 5500, "END")])
                        self.assertEqual(path.read_bytes(), original)

    def test_decoding_error_has_actionable_pipeline_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.srt"
            path.write_bytes(codecs.BOM_UTF16_LE + b"x")
            with self.assertRaises(ConfigurationError) as caught:
                parse_subtitle(str(path))
            payload = caught.exception.to_payload()
            self.assertEqual(payload["code"], "CFG_SUBTITLE_ENCODING")
            self.assertEqual(payload["action"], "locate_subtitle")
            self.assertFalse(payload["retryable"])
