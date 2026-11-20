"""Filesystem persistence for actual audio only; project documents use storage.py."""
from contextlib import contextmanager
from pathlib import Path


def intermediate_dir(output_dir: str, name: str) -> str:
    directory = Path(output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def write_bytes(path: str, data: bytes) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return path


@contextmanager
def open_write(path: str, mode: str = "wb", **kwargs):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open(mode, **kwargs) as output:
        yield output
