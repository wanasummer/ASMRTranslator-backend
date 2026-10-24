"""Split the continuous pink timeline after decoding the source exactly once."""

import os
import subprocess
import wave
from pathlib import Path

from .pink_segments import calculate_pink_segments
from .vtt_parser import VTTBlock


def get_ffmpeg_path() -> str:
    local_path = Path(__file__).parent.parent / "ffmpeg_bin" / "ffmpeg.exe"
    return str(local_path) if local_path.exists() else "ffmpeg"


def get_audio_duration(audio_path: str) -> float:
    local_ffprobe = str(Path(__file__).parent.parent / "ffmpeg_bin" / "ffprobe.exe")
    if not Path(local_ffprobe).exists():
        local_ffprobe = "ffprobe"
    cmd = [
        local_ffprobe, "-v", "quiet", "-show_entries", "format=duration",
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
) -> list[dict]:
    """Decode once and write one PCM slice per subtitle block.

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
    _normalize_audio(audio_path, normalized_path)

    segments: list[dict] = []
    try:
        with wave.open(normalized_path, "rb") as source:
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
                speech_start = min(frame_count, round(max(0, block.start_ms) * frame_rate / 1000))
                speech_end = min(frame_count, round(max(0, block.end_ms) * frame_rate / 1000))
                source.setpos(speech_start)
                speech_frames = source.readframes(max(0, speech_end - speech_start))
                speech_path = os.path.join(output_dir, f"speech_{idx:04d}.wav")
                with wave.open(speech_path, "wb") as target:
                    target.setparams(params)
                    target.writeframes(speech_frames)

                segments.append({
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
        try:
            os.remove(normalized_path)
        except OSError:
            pass

    return segments
