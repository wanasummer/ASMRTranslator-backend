"""Generic ffmpeg range extraction for the ASR path.

The VAD stage produces optimal speech slices; ``extract_ranges`` pulls each
``(start, end)`` span out of the original audio with ffmpeg input-seek in
parallel. ``extract_chunk`` is the single-span primitive the slice extraction
is built on.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from ...error import ASRError
from ..models import AudioChunk
from ...audio_splitter import get_ffmpeg_path


def extract_chunk(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    sample_rate: int = 24000,
    channels: int = 2,
) -> None:
    """Extract ``[start_sec, end_sec]`` from ``audio_path`` into ``out_path``.

    Uses ffmpeg input-seek (``-ss`` before ``-i``) so the seek is fast and exact,
    then re-encodes to PCM WAV at ``sample_rate``/``channels``.
    """
    dur = max(0.0, end_sec - start_sec)
    if dur <= 0:
        raise ASRError(f"extract_chunk: 非正时长 {start_sec}-{end_sec}")
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", audio_path,
        "-c", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise ASRError(
            f"extract_chunk ffmpeg 失败 ({start_sec:.2f}-{end_sec:.2f}s): "
            f"{result.stderr.decode('utf-8', 'replace')[:300]}"
        )


def extract_ranges(
    audio_path: str,
    ranges,
    output_dir: str,
    sample_rate: int = 24000,
    channels: int = 2,
    max_workers: int = 8,
) -> list[AudioChunk]:
    """Extract every ``(start, end)`` span in ``ranges`` to ``output_dir``.

    Returns one ``AudioChunk`` per span (id ``slice_%04d``), extracted in
    parallel with a ``ThreadPoolExecutor``. The first span's id uses ``total_sec``
    to keep the 4-digit width stable for any realistic file length.
    """
    os.makedirs(output_dir, exist_ok=True)
    spans = [tuple(r) for r in ranges]
    width = max(4, len(str(len(spans))))

    def _extract(i: int, s: float, e: float) -> str:
        path = os.path.join(output_dir, f"slice_{i:0{width}d}.wav")
        extract_chunk(audio_path, s, e, path, sample_rate, channels)
        return path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_extract, i, s, e) for i, (s, e) in enumerate(spans)
        ]
        paths = [f.result() for f in futures]

    return [
        AudioChunk(
            id=f"slice_{i:0{width}d}",
            start=spans[i][0],
            end=spans[i][1],
            path=paths[i],
        )
        for i in range(len(spans))
    ]
