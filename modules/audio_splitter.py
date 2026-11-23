"""Split the continuous pink timeline after decoding the source exactly once."""

import os
import shutil
import subprocess
import sys
import wave
from contextlib import ExitStack
from pathlib import Path

from .pink_segments import calculate_pink_segments
from .vtt_parser import VTTBlock


def _candidate_ffmpeg_dirs(module_file: str | Path = __file__) -> list[Path]:
    """Return ffmpeg_bin candidate directories in priority order (resources first)."""
    candidates: list[Path] = []
    env_dir = os.environ.get("ASMR_FFMPEG_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    if sys.argv and sys.argv[0]:
        argv_dir = Path(sys.argv[0]).resolve().parent
        candidates.append(argv_dir / "ffmpeg_bin")
        candidates.append(argv_dir / "resources" / "ffmpeg_bin")
    candidates.append(Path(module_file).resolve().parent.parent / "ffmpeg_bin")
    executable = getattr(sys, "executable", None)
    if executable:
        exe_dir = Path(executable).resolve().parent
        candidates.append(exe_dir / "ffmpeg_bin")
        candidates.append(exe_dir / "resources" / "ffmpeg_bin")

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = os.path.normcase(os.path.normpath(str(candidate)))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _prepend_path_dir(directory: Path) -> None:
    """Ensure `directory` is at the front of PATH for pydub and subprocess lookups."""
    dir_str = str(directory)
    current = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    norm_dir = os.path.normcase(os.path.normpath(dir_str))
    if current and os.path.normcase(os.path.normpath(current[0])) == norm_dir:
        return
    filtered = [entry for entry in current if os.path.normcase(os.path.normpath(entry)) != norm_dir]
    os.environ["PATH"] = os.pathsep.join([dir_str, *filtered])


def _binary_names(stem: str) -> tuple[str, str]:
    return (f"{stem}.exe", stem) if os.name == "nt" else (stem, f"{stem}.exe")


def _resolve_audio_binary(stem: str, module_file: str | Path = __file__) -> str:
    """Resolve `ffmpeg`/`ffprobe` from bundled resources first, falling back to PATH."""
    names = _binary_names(stem)
    for directory in _candidate_ffmpeg_dirs(module_file):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                _prepend_path_dir(directory)
                return str(candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return stem


def ensure_ffmpeg_on_path(module_file: str | Path = __file__) -> None:
    """Prepend the bundled resources/ffmpeg_bin directory to PATH when available."""
    dirs = _candidate_ffmpeg_dirs(module_file)
    names = _binary_names("ffmpeg")
    ffmpeg_dir = next(
        (d for d in dirs if any((d / name).is_file() for name in names)),
        next((d for d in dirs if d.is_dir()), None),
    )
    if ffmpeg_dir is not None:
        _prepend_path_dir(ffmpeg_dir)


def get_ffmpeg_path() -> str:
    return _resolve_audio_binary("ffmpeg")


def get_ffprobe_path() -> str:
    return _resolve_audio_binary("ffprobe")


ensure_ffmpeg_on_path()


def get_audio_duration(audio_path: str) -> float:
    cmd = [
        get_ffprobe_path(), "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or not result.stdout.strip():
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"ffprobe 无法读取音频时长: {detail or audio_path}")
    return float(result.stdout.strip())


def _normalize_audio(audio_path: str, normalized_path: str) -> str:
    """Decode the compressed source to the format used downstream."""
    cmd = [
        get_ffmpeg_path(), "-y", "-v", "error", "-i", audio_path,
        "-vn", "-map_metadata", "-1", "-c:a", "pcm_s16le",
        "-ar", "24000", "-ac", "2", normalized_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(normalized_path):
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg 音频预解码失败: {stderr or '未生成中间 WAV'}")
    return normalized_path


def split_audio(
    audio_path: str,
    blocks: list[VTTBlock],
    output_dir: str,
    prefix: str = "seg",
    max_workers: int = 1,
    include_asr_blocks: bool = False,
    reference_audio_path: str | None = None,
) -> list[dict]:
    """Write original coverage and references from independently selected audio.

    ``reference_audio_path`` supplies timeline-preserving processed speech for
    automatic TTS conditioning. Original pink audio always uses ``audio_path``.

    ``max_workers`` is retained for API compatibility. PCM slicing is kept
    sequential because the spans are ordered; concurrent disk readers add
    contention and were the source of the former Windows process explosion.
    """
    del max_workers
    os.makedirs(output_dir, exist_ok=True)
    total_duration_ms = int(get_audio_duration(audio_path) * 1000)

    if not all(b.pink_start_ms is not None for b in blocks):
        for pink in calculate_pink_segments(blocks, total_duration_ms):
            block = pink.subtitle_data
            block.pink_start_ms = pink.pink_start_ms
            block.pink_end_ms = pink.pink_end_ms
            block.tts_delay_ms = pink.tts_delay_ms
            block.max_allowed_tts_ms = pink.max_allowed_tts_ms

    normalized_path = os.path.join(output_dir, "_source_24k_stereo.wav")
    reference_path = os.path.join(output_dir, "_reference_24k_stereo.wav")
    _normalize_audio(audio_path, normalized_path)

    segments: list[dict] = []
    written_parents: set[Path] = set()
    try:
        with ExitStack() as stack:
            source = stack.enter_context(wave.open(normalized_path, "rb"))
            reference = source
            if reference_audio_path is not None and os.path.abspath(reference_audio_path) != os.path.abspath(audio_path):
                _normalize_audio(reference_audio_path, reference_path)
                reference = stack.enter_context(wave.open(reference_path, "rb"))
                if ((reference.getframerate(), reference.getnchannels(), reference.getsampwidth()) !=
                        (source.getframerate(), source.getnchannels(), source.getsampwidth()) or
                        abs(reference.getnframes() - source.getnframes()) > round(source.getframerate() / 1000)):
                    raise ValueError("TTS reference audio does not match the original timeline")
            frame_rate = source.getframerate()
            frame_count = source.getnframes()
            params = source.getparams()

            for idx, block in enumerate(blocks):
                start_ms = max(0, block.pink_start_ms
                               if block.pink_start_ms is not None else block.start_ms)
                end_ms = min(total_duration_ms, block.pink_end_ms
                             if block.pink_end_ms is not None else block.end_ms)
                start_frame = min(frame_count, round(start_ms * frame_rate / 1000))
                end_frame = min(frame_count, round(end_ms * frame_rate / 1000))
                source.setpos(start_frame)
                frames = source.readframes(max(0, end_frame - start_frame))

                seg_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.wav")
                with wave.open(seg_path, "wb") as target:
                    target.setnchannels(params.nchannels)
                    target.setsampwidth(params.sampwidth)
                    target.setframerate(params.framerate)
                    target.writeframes(frames)

                # Pink spans contain surrounding silence and may include another
                # speaker. Emotion conditioning must use the subtitle speech span.
                speech_start = min(reference.getnframes(), round(max(0, block.start_ms) * frame_rate / 1000))
                speech_end = min(reference.getnframes(), round(max(0, block.end_ms) * frame_rate / 1000))
                reference.setpos(speech_start)
                speech_frames = reference.readframes(max(0, speech_end - speech_start))
                speech_path = os.path.join(output_dir, f"speech_{idx:04d}.wav")
                with wave.open(speech_path, "wb") as target:
                    target.setparams(reference.getparams())
                    target.writeframes(speech_frames)

                parent = {}
                if include_asr_blocks:
                    from .tts.references import parent_span, block_reference_path
                    identity, parent_start, parent_end = parent_span(block)
                    if not 0 <= parent_start < parent_end <= total_duration_ms:
                        raise ValueError("ASR block reference is outside the source timeline")
                    parent_path = block_reference_path(output_dir, identity, parent_start, parent_end)
                    if parent_path not in written_parents:
                        reference.setpos(min(reference.getnframes(), round(parent_start * frame_rate / 1000)))
                        with wave.open(str(parent_path), "wb") as target:
                            target.setparams(reference.getparams())
                            target.writeframes(reference.readframes(round(parent_end * frame_rate / 1000)
                                                                 - round(parent_start * frame_rate / 1000)))
                        written_parents.add(parent_path)
                    parent = {"asr_block_file": str(parent_path), "tts_context": {
                        "source_block_id": identity, "source_start_ms": parent_start, "source_end_ms": parent_end}}
                segments.append({
                    **parent,
                    "tts_context": {"reference_audio_processing": "processed" if reference is not source else "original",
                                    **parent.get("tts_context", {})},
                    "index": idx,
                    "segment_id": str(block.index),
                    "speech_file": speech_path,
                    "source_text": getattr(block, "source_text", ""),
                    "segment_file": seg_path,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "original_start_ms": block.start_ms,
                    "original_end_ms": block.end_ms,
                    "tts_delay_ms": block.tts_delay_ms or 0,
                    "max_allowed_tts_ms": block.max_allowed_tts_ms or 0,
                    "speaker": block.speaker,
                    "text": block.text,
                    "skip_tts": block.skip_tts,
                })
    finally:
        # The per-segment WAVs are needed by later stages; the decoded full
        # source is not. Releasing it here avoids retaining ~128 MiB for a
        # 23-minute source until the next run.
        for path in (normalized_path, reference_path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    return segments
