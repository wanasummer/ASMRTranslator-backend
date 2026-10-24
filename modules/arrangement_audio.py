"""Replaceable arrangement audio adapter; transformations return new audio values."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .arrangement_contracts import ArrangementError, ArrangedClip, ArrangementPlan, ArrangementAudioResult, ArrangementAudioAdapter
from .audio_joiner import get_ffmpeg_path
from .io import write_json, write_text


def perceived_db(audio) -> float | None:
    """Same gated multi-channel 20 ms RMS reference used by studioLoudness.ts."""
    samples = audio.get_array_of_samples()
    window = max(1, int(audio.frame_rate * .02)) * audio.channels
    scale = float(1 << (audio.sample_width * 8 - 1))
    energies = [sum((sample / scale) ** 2 for sample in samples[i:i + window]) / len(samples[i:i + window])
                for i in range(0, len(samples), window)]
    values = [energy for energy in energies if energy > 0]
    if not values:
        return None
    gate = max(max(values) * .0001, .000001)
    kept = [energy for energy in values if energy >= gate] or values
    return 10 * math.log10(sum(kept) / len(kept))


class PydubArrangementAdapter:
    def render(self, plan: ArrangementPlan) -> ArrangementAudioResult:
        from pydub import AudioSegment
        original = AudioSegment.from_file(plan.source_audio)
        original = original.set_channels(2)
        if plan.clips and plan.clips[0].end_ms < -5000:
            raise ArrangementError("首段 TTS 与原声的间隔不能超过 5 秒")
        if plan.clips and plan.clips[-1].start_ms > len(original) + 5000:
            raise ArrangementError("末段 TTS 与原声的间隔不能超过 5 秒")
        origin_shift_ms = max(0, -plan.clips[0].start_ms) if plan.clips else 0
        if origin_shift_ms:
            original = AudioSegment.silent(duration=origin_shift_ms, frame_rate=original.frame_rate).set_channels(2) + original
        parts: list[bytes] = []
        cursor = 0
        for clip in plan.clips:
            clip_start = clip.start_ms + origin_shift_ms
            clip_end = clip.end_ms + origin_shift_ms
            if clip_end > len(original):
                original += AudioSegment.silent(duration=clip_end - len(original), frame_rate=original.frame_rate).set_channels(2)
            audio = AudioSegment.from_file(clip.path)
            if clip.sample_end_ms is not None:
                audio = audio[clip.sample_start_ms:clip.sample_end_ms]
            if not len(audio):
                raise ArrangementError(f"片段 {clip.index} 的 TTS 音频为空")
            target_ms = clip.end_ms - clip.start_ms
            required_tempo = len(audio) / target_ms
            tempo = max(.5, min(2, required_tempo))
            if abs(required_tempo - tempo) > .001:
                print(f"[arrangement] 片段 {clip.index}: 语速限制为 {tempo:.2f}×，按试听区间截取或保留尾部原声")
            if abs(tempo - 1) > .001:
                audio = self.stretch(audio, max(1, round(len(audio) / tempo)))
            target_db = perceived_db(original[clip.reference_start_ms + origin_shift_ms:clip.reference_end_ms + origin_shift_ms])
            source_db = perceived_db(audio)
            if clip.volume == 0:
                audio = audio._spawn(bytes(len(audio.raw_data)))
            elif source_db is not None:
                gain_db = (target_db - source_db if target_db is not None else 0) + 20 * math.log10(clip.volume)
                ceiling_db = 20 * math.log10(.99) - audio.max_dBFS
                audio = audio.apply_gain(min(gain_db, ceiling_db))
            # Upmix in memory, preserving mono TTS level as in the Web Audio preview.
            if clip_start < cursor:
                raise ArrangementError(f"片段 {clip.index} 与前一片段重叠")
            parts.append(original[cursor:clip_start].raw_data)
            parts.append(original[clip_start:clip_end].overlay(audio.set_channels(2)[:target_ms]).raw_data)
            cursor = clip_end
            print(f"[arrangement] 片段 {clip.index}: 区间 {clip.start_ms}–{clip.end_ms} ms，语速 {tempo:.2f}×，响度 {clip.volume:.0%}")
        parts.append(original[cursor:].raw_data)
        # Assemble once instead of copying the full long ASMR track for every clip.
        return ArrangementAudioResult(original._spawn(b"".join(parts)), plan.clips, origin_shift_ms)

    def stretch(self, audio, target_ms: int):
        from pydub import AudioSegment
        ffmpeg = get_ffmpeg_path()
        if not Path(ffmpeg).is_file():
            raise ArrangementError("未找到随应用提供的 ffmpeg")
        with TemporaryDirectory(prefix="asmr-arrangement-") as directory:
            source, target = Path(directory) / "source.wav", Path(directory) / "target.wav"
            audio.export(source, format="wav")
            try:
                subprocess.run([ffmpeg, "-y", "-i", str(source), "-filter:a",
                                f"atempo={len(audio) / target_ms:.8f}", str(target)],
                               check=True, capture_output=True, creationflags=0x08000000 if __import__("os").name == "nt" else 0)
            except subprocess.CalledProcessError as exc:
                raise ArrangementError("编排语速处理失败") from exc
            return AudioSegment.from_wav(target)


_AUDIO_ADAPTERS = {"pydub": PydubArrangementAdapter}


def create_arrangement_audio_adapter(name: str = "pydub") -> ArrangementAudioAdapter:
    try:
        return _AUDIO_ADAPTERS[name]()
    except KeyError as exc:
        raise ArrangementError(f"未知编排音频实现：{name}") from exc


def write_arrangement_result(result: ArrangementAudioResult, directory: Path, snapshot: dict) -> dict:
    """Persist final artifacts separately from audio transformation and orchestration."""
    directory.mkdir(parents=True, exist_ok=True)
    wav, mp3 = directory / "output.wav", directory / "output.mp3"
    result.audio.export(wav, format="wav")
    result.audio.export(mp3, format="mp3", bitrate="192k")
    write_json(str(directory / "snapshot.json"), snapshot)
    write_json(str(directory / "audit.json"), [{"index": c.index, "path": str(c.path), "start_ms": c.start_ms,
                                               "end_ms": c.end_ms, "output_start_ms": c.start_ms + result.origin_shift_ms,
                                               "output_end_ms": c.end_ms + result.origin_shift_ms,
                                               "volume": c.volume} for c in result.clips])
    def stamp(ms):
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"
    vtt = "WEBVTT\n\n" + "\n\n".join(f"{clip.index}\n{stamp(clip.start_ms + result.origin_shift_ms)} --> {stamp(clip.end_ms + result.origin_shift_ms)}\n{clip.text}" for clip in result.clips) + "\n"
    path = write_text(str(directory / "final.vtt"), vtt)
    return {"output_wav": str(wav), "output_mp3": str(mp3), "final_vtt": path}
