"""Rebuild exports from the original audio and returned TTS stage artifacts.

Normal processing remains lightweight. The recipe is an immutable, versioned
boundary: export never re-runs ASR/TTS or upscales the low-quality mixed output.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .audio_splitter import get_ffmpeg_path
from .io import write_json
from .task_cache import file_digest

RECIPE_PATH = "debug/high_quality_export.json"


class HighQualityExportError(RuntimeError):
    """Missing/stale render data or failed native-quality reconstruction."""


@dataclass(frozen=True)
class AudioProfile:
    sample_rate: int
    channels: int
    bits: int
    codec: str
    bitrate: int

    @property
    def wav_params(self) -> list[str]:
        return ["-acodec", "pcm_s32le", "-ar", str(self.sample_rate), "-ac", str(self.channels)]

    @property
    def mp3_bitrate(self) -> int:
        """Preserve CBR; round a VBR average up to a supported MP3 rate."""
        rates = ((32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
                 if self.sample_rate >= 32000 else
                 (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160))
        for rate in rates:
            if rate * 1000 >= self.bitrate > 0:
                return rate * 1000
        raise HighQualityExportError("原 MP3 码率不在支持范围内，未降低码率导出。")


def _command(args: list[str]) -> bytes:
    try:
        result = subprocess.run(args, capture_output=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as error:
        raise HighQualityExportError("无法启动随应用提供的音频工具。") from error
    if result.returncode:
        raise HighQualityExportError(result.stderr.decode("utf-8", errors="replace")[-1500:])
    return result.stdout


def probe_audio(path: str) -> AudioProfile:
    ffmpeg = Path(get_ffmpeg_path())
    probe = str(ffmpeg.with_name("ffprobe.exe")) if ffmpeg.name == "ffmpeg.exe" else "ffprobe"
    data = json.loads(_command([probe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,bits_per_sample,bits_per_raw_sample,codec_name,bit_rate",
        "-of", "json", path]))
    stream = data["streams"][0]
    profile = AudioProfile(int(stream["sample_rate"]), int(stream["channels"]),
        int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 16),
        stream["codec_name"], int(stream.get("bit_rate") or 0))
    if profile.sample_rate <= 0 or not 1 <= profile.channels <= 8:
        raise HighQualityExportError("原音频的采样率或声道无效。")
    return profile


def write_export_recipe(audio_path: str, segments: list[dict], settings: dict,
                        output_dir: str, crossfade_ms: int, overhang_ms: int) -> str:
    """Snapshot only the returned build contract and explicit render settings.

    Keep source identity and all TTS fingerprints so cleared/changed caches can
    never silently produce a different reconstruction. No input is mutated.
    """
    work = Path(output_dir).resolve()
    records, inputs = [], {}
    fields = ("index", "start_ms", "end_ms", "skip_tts", "tts_delay_ms", "max_allowed_tts_ms")
    for segment in segments:
        record = {key: segment[key] for key in fields if key in segment}
        for key in ("tts_file", "phantom_dsp_file", "phantom_voice_src"):
            if segment.get("skip_tts") or not segment.get(key):
                continue
            path = Path(segment[key]).resolve()
            relative = path.relative_to(work).as_posix()
            inputs[relative] = file_digest(str(path))
            record[key] = relative
        records.append(record)
    return write_json(str(work / RECIPE_PATH), {
        "contractVersion": 1, "source": {"path": str(Path(audio_path).resolve()),
        "sha256": file_digest(audio_path)}, "settings": settings,
        "crossfadeMs": crossfade_ms, "overhangMs": overhang_ms,
        "segments": records, "inputs": inputs,
    })


def load_recipe(path: str) -> tuple[dict, Path]:
    recipe_path = Path(path).resolve()
    work = recipe_path.parent.parent
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("contractVersion") != 1 or not recipe.get("segments"):
        raise HighQualityExportError("高音质重建记录无效，请重新生成任务。")
    source = recipe["source"]
    if file_digest(source["path"]) != source["sha256"]:
        raise HighQualityExportError("原音频已改变，请重新生成任务。")
    for relative, digest in recipe["inputs"].items():
        target = (work / relative).resolve()
        if not target.is_relative_to(work) or not target.is_file() or file_digest(str(target)) != digest:
            raise HighQualityExportError("TTS 缓存已清理或改变，请重新生成任务后导出高音质。")
    return recipe, work


def _native_slices(recipe: dict, work: Path, folder: Path, profile: AudioProfile) -> list[dict]:
    segments = []
    for index, original in enumerate(recipe["segments"]):
        segment = dict(original)
        for key in ("tts_file", "phantom_dsp_file", "phantom_voice_src"):
            if key in segment:
                if segment[key] not in recipe["inputs"]:
                    raise HighQualityExportError("TTS 路径缺少完整性记录。")
                segment[key] = str(work / segment[key])
        start, end = float(segment["start_ms"]), float(segment["end_ms"])
        if start < 0 or end <= start:
            raise HighQualityExportError("高音质重建时间轴无效。")
        path = folder / f"original_{index:05d}.wav"
        _command([get_ffmpeg_path(), "-y", "-v", "error", "-ss", str(start / 1000),
            "-i", recipe["source"]["path"], "-t", str((end - start) / 1000),
            "-vn", *profile.wav_params, str(path)])
        segment["segment_file"] = str(path)
        segments.append(segment)
    return segments


def _append_pcm(output, data: np.ndarray, crossfade: int, overlap: int, channels: int) -> int:
    """Append a bounded PCM slice, mixing only the overlapping tail in float64."""
    frame_bytes = channels * 4
    total = output.tell() // frame_bytes
    length = len(data)
    virtual = overlap > 0 and total > overlap
    count = min(overlap, length) if virtual else min(crossfade, total // 2, length // 2)
    if count:
        output.seek(-count * frame_bytes, os.SEEK_CUR)
        tail = np.frombuffer(output.read(count * frame_bytes), dtype="<i4").reshape(-1, channels)
        head = data[:count].astype(np.float64)
        if virtual:
            mixed = tail.astype(np.float64) + head
        else:
            ramp = np.arange(count, dtype=np.float64).reshape(-1, 1) / count
            mixed = tail * (1 - ramp) + head * ramp
        output.seek(-count * frame_bytes, os.SEEK_CUR)
        output.write(np.clip(mixed, -2147483648, 2147483647).astype("<i4").tobytes())
    output.write(data[count:].tobytes())
    return total + length - count


def _render_high_quality(recipe_path: str, target_path: str) -> str:
    """Rebuild through existing mode strategies at the native source profile."""
    from .mode_builders import create_segment_builder
    from .audio_joiner import apply_follow_translation_cap
    from .audio_metadata import create_metadata_reader, write_audio_metadata

    recipe, work = load_recipe(recipe_path)
    source = recipe["source"]["path"]
    profile = probe_audio(source)
    target = Path(target_path)
    if target.suffix.lower() not in {".mp3", ".wav"}:
        raise HighQualityExportError("高音质导出目标需要 MP3 或 WAV。")
    if target.suffix.lower() == ".mp3" and (profile.codec != "mp3" or profile.bitrate <= 0):
        raise HighQualityExportError("无法读取原 MP3 码率，未使用默认码率替代。")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hq-", dir=target.parent) as temporary:
        folder = Path(temporary)
        segments = _native_slices(recipe, work, folder, profile)
        builder = create_segment_builder(**recipe["settings"])
        builder.wav_params = profile.wav_params
        segments = builder.build(segments, str(folder / "build"), max_workers=1)
        for segment in segments:
            if segment.get("skip_tts"):
                segment["vc_file"] = segment["segment_file"]
            elif not segment.get("vc_file"):
                segment["vc_file"] = segment["tts_file"]
        if recipe["settings"].get("sync_mode") == "follow_translation":
            segments = apply_follow_translation_cap(segments, max_overhang_ms=recipe["overhangMs"],
                                                    wav_params=profile.wav_params)
        pcm = folder / "combined.pcm"
        count = 0
        with pcm.open("w+b") as output:
            for segment in segments:
                raw = _command([get_ffmpeg_path(), "-v", "error", "-i", segment["vc_file"],
                    "-map", "0:a:0", "-f", "s32le", "-ar", str(profile.sample_rate),
                    "-ac", str(profile.channels), "-"])
                samples = np.frombuffer(raw, dtype="<i4").reshape(-1, profile.channels)
                if len(samples) < profile.sample_rate * 0.05:
                    continue
                count = _append_pcm(output, samples,
                    round(recipe["crossfadeMs"] * profile.sample_rate / 1000),
                    round(segment.get("overlap_ms", 0) * profile.sample_rate / 1000), profile.channels)
        if not count:
            raise HighQualityExportError("没有可导出的有效高音质片段。")
        bits = profile.bits if profile.bits in {8, 16, 24, 32} else 24
        codec = f"pcm_s{bits}le" if bits != 8 else "pcm_u8"
        if profile.codec.startswith("pcm_f"):
            codec = "pcm_f32le"
        encoding = ["-c:a", "libmp3lame", "-b:a", str(profile.mp3_bitrate)] if target.suffix.lower() == ".mp3" else ["-c:a", codec]
        _command([get_ffmpeg_path(), "-y", "-v", "error", "-f", "s32le", "-ar", str(profile.sample_rate),
            "-ac", str(profile.channels), "-i", str(pcm), *encoding, str(target)])
        actual = probe_audio(str(target))
        if (actual.sample_rate != profile.sample_rate or actual.channels != profile.channels
                or (target.suffix.lower() == ".mp3" and abs(actual.bitrate - profile.mp3_bitrate) > 1000)
                or (target.suffix.lower() == ".wav" and actual.bits != bits)):
            raise HighQualityExportError("编码器未能保持原音频参数，已停止发布。")
        metadata = create_metadata_reader().read(source)
        write_audio_metadata(metadata, (str(target),), str(folder / "metadata"))
    print(f"[export.high-quality] sample_rate={profile.sample_rate} bits={profile.bits} "
          f"channels={profile.channels} bitrate={profile.bitrate} output={target}")
    return str(target)


def render_high_quality(recipe_path: str, target_path: str) -> str:
    """Public adapter boundary; keep filesystem/codec errors inside the domain."""
    try:
        return _render_high_quality(recipe_path, target_path)
    except HighQualityExportError:
        raise
    except Exception as error:
        raise HighQualityExportError(f"高音质重建失败：{error}") from error
