"""Strict text decoding shared by import adapters; source bytes stay untouched.

BOM and valid UTF-8 take precedence over statistical detection. Legacy encodings
are necessarily heuristic; callers can supply an explicit encoding to override
ambiguous detection. No replacement characters or lossy decoding are introduced.
"""

import codecs
from dataclasses import dataclass
from pathlib import Path

import chardet


class TextDecodingError(ValueError):
    """Input cannot be decoded reliably as supported text."""


@dataclass(frozen=True)
class DecodedText:
    text: str
    encoding: str
    confidence: float

    @property
    def utf8_bytes(self) -> bytes:
        """Canonical UTF-8 without BOM, ready for an independent writer."""
        return self.text.encode("utf-8")


def _decoded(data: bytes, encoding: str, confidence: float) -> DecodedText:
    try:
        text = data.decode(encoding)
    except (UnicodeError, LookupError) as error:
        raise TextDecodingError(f"无法按 {encoding} 解码文本，请检查文件编码。") from error
    if any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
        raise TextDecodingError("文件包含二进制控制字符，无法作为文本导入。")
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    return DecodedText(text, codecs.lookup(encoding).name, confidence)


def decode_text(data: bytes, encoding: str | None = None) -> DecodedText:
    """Decode Unicode and common CJK/Western legacy encodings without data loss.

    ``None``/``auto`` detect encoding. Explicit encodings are honored strictly.
    Low-confidence statistical results fail instead of guessing a permissive
    single-byte codec that could hide corrupt input.
    """
    if encoding is not None and encoding.lower() != "auto":
        return _decoded(data, encoding, 1.0)
    # UTF-32 LE starts with the UTF-16 LE marker, so test longer BOMs first.
    for marker, codec in (
        (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if data.startswith(marker):
            return _decoded(data, codec, 1.0)
    # Unmarked Unicode has characteristic zero-byte lanes in timestamp/ASCII
    # text. Only accept a lane match when the entire input decodes strictly.
    if b"\0" in data:
        for width, candidates in (
            (4, (("utf-32-le", (2, 3)), ("utf-32-be", (0, 1)))),
            (2, (("utf-16-le", (1,)), ("utf-16-be", (0,)))),
        ):
            if len(data) < width * 4 or len(data) % width:
                continue
            lanes = [data[offset::width] for offset in range(width)]
            zeros = [lane.count(0) / len(lane) for lane in lanes]
            for codec, positions in candidates:
                if all(zeros[position] >= 0.4 for position in positions):
                    try:
                        return _decoded(data, codec, 0.9)
                    except TextDecodingError:
                        pass
    # NULs can indicate BOM-less UTF-16/32; ESC can indicate ISO-2022 text.
    if b"\0" not in data and b"\x1b" not in data:
        try:
            return _decoded(data, "utf-8", 1.0)
        except TextDecodingError:
            pass
    result = chardet.detect(data)
    candidate = result.get("encoding")
    confidence = float(result.get("confidence") or 0)
    if not candidate or confidence < 0.6:
        raise TextDecodingError("无法可靠识别文本编码，请指定原文件编码或转存为 UTF-8。")
    # Detected GB2312/GBK and Shift-JIS inputs can include common extensions.
    candidate = {"gb2312": "gb18030", "gbk": "gb18030", "shift_jis": "cp932"}.get(
        candidate.lower(), candidate)
    return _decoded(data, candidate, confidence)


def read_text(path: str, encoding: str | None = None) -> DecodedText:
    """Read without rewriting the original; log only encoding and byte counts."""
    data = Path(path).read_bytes()
    result = decode_text(data, encoding)
    print(f"       [text.decode] encoding={result.encoding} confidence={result.confidence:.2f} "
          f"source_bytes={len(data)} utf8_bytes={len(result.utf8_bytes)}")
    return result
