"""Shared file-writing helpers — the project's UTF-8 write conventions.

JSON artifacts across the codebase all used the same inline pattern
(``open(path, "w", encoding="utf-8")`` + ``json.dump(obj, f,
ensure_ascii=False, indent=2)``); text/VTT files a sibling pattern. These
helpers centralize those choices so encoding/indentation stay consistent.

源码运行和打包 EXE 使用完全相同的写入布局：最终音频、中间切片、字幕、
调试文件均写入当前项目的输出目录。项目级目录天然隔离并发任务，也避免
一个打包进程清理另一个仍在运行进程的系统临时目录。
"""

import json
import os
import sys
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# 环境判定 + 兼容写入接口
# ---------------------------------------------------------------------------


def is_build() -> bool:
    """True when running as a packaged EXE (PyInstaller or Nuitka)."""
    return bool(getattr(sys, "frozen", False) or "__compiled__" in globals())


def is_allowed_output(path: str) -> bool:
    """All project outputs are writable in both dev and packaged builds."""
    return True


def _check_write(path: str) -> bool:
    """Compatibility hook retained for callers; writes are never filtered."""
    return True


def intermediate_dir(output_dir: str, name: str) -> str:
    """Resolve an intermediate directory under the current project output."""
    d = os.path.join(output_dir, name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 统一写入入口（全部写操作经此）
# ---------------------------------------------------------------------------


def write_json(path: str, obj, indent: int = 2) -> str:
    """Serialize ``obj`` to ``path`` as UTF-8 JSON (non-ASCII kept readable).

    Creates the parent directory if missing and returns ``path``.
    """
    if not _check_write(path):
        return path
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path


def write_text(path: str, text: str, encoding: str = "utf-8") -> str:
    """Write ``text`` to ``path``, creating the parent directory if missing."""
    if not _check_write(path):
        return path
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as f:
        f.write(text)
    return path


def write_bytes(path: str, data: bytes) -> str:
    """Write raw bytes to ``path`` (audio files, downloads, etc.)."""
    if not _check_write(path):
        return path
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def write_log(path: str, text: str, encoding: str = "utf-8") -> str:
    """Write a log file."""
    return write_text(path, text, encoding)


def open_log(path: str, encoding: str = "utf-8"):
    """Open a whole-run log file for writing (e.g. PipelineTee).

    Returns a writable UTF-8 file object.
    """
    if not _check_write(path):
        return _NullWriter()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return open(path, "w", encoding=encoding)


class _NullWriter:
    """Drop-in file object for filtered writes: accepts any write call and
    discards the data, so streaming callers keep working without a file."""

    def write(self, data) -> int:
        return len(data)

    def close(self):
        pass

    def flush(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def open_write(path: str, mode: str = "wb", **kwargs):
    """Open ``path`` for writing, creating its parent directory."""
    if not _check_write(path):
        yield _NullWriter()
        return
    dir_path = os.path.dirname(path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with open(path, mode, **kwargs) as f:
        yield f
