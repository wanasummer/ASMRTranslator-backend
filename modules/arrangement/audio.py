"""Replaceable arrangement audio adapter; transformations return new audio values."""
from __future__ import annotations

from dataclasses import replace
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .contracts import ArrangementError, ArrangedClip, ArrangementPlan, ArrangementAudioResult, ArrangementAudioAdapter
from ..audio_joiner import get_ffmpeg_path
from ..storage import save_json, save_text


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


class _PydubArrangementBase:
    """Shared clip preparation and tempo stretching for pydub arrangement adapters."""

    def _stretch_audio(self, audio, target_ms: int, index: int, label: str = "TTS"):
        required_tempo = len(audio) / max(1, target_ms)
        tempo = max(.5, required_tempo)
        if abs(required_tempo - tempo) > .001:
            print(f"[arrangement] 片段 {index} ({label}): 语速限制为 {tempo:.2f}×，按试听区间截取或保留尾部原声")
        if abs(tempo - 1) > .001:
            audio = self.stretch(audio, max(1, round(len(audio) / tempo)))
        return audio, tempo

    def _prepare_clip_audio(self, original, clip: ArrangedClip, origin_shift_ms: int = 0, target_ms: int | None = None):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(clip.path)
        if clip.sample_end_ms is not None:
            audio = audio[clip.sample_start_ms:clip.sample_end_ms]
        if not len(audio):
            raise ArrangementError(f"片段 {clip.index} 的 TTS 音频为空")
        desired_ms = max(1, target_ms if target_ms is not None else clip.end_ms - clip.start_ms)
        audio, tempo = self._stretch_audio(audio, desired_ms, clip.index, "TTS")
        target_db = perceived_db(original[clip.reference_start_ms + origin_shift_ms:clip.reference_end_ms + origin_shift_ms])
        source_db = perceived_db(audio)
        if clip.volume == 0:
            audio = audio._spawn(bytes(len(audio.raw_data)))
        elif source_db is not None:
            gain_db = (target_db - source_db if target_db is not None else 0) + 20 * math.log10(clip.volume)
            ceiling_db = 20 * math.log10(.99) - audio.max_dBFS
            audio = audio.apply_gain(min(gain_db, ceiling_db))
        return audio, tempo

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


class PydubArrangementAdapter(_PydubArrangementBase):
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
            audio, tempo = self._prepare_clip_audio(original, clip, origin_shift_ms)
            target_ms = clip.end_ms - clip.start_ms
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


class DualArrangementAdapter(_PydubArrangementBase):
    """Render retain-original mode per coarse group: [original -> tts -> original -> tts] canvas."""

    def _render_coarse_group(self, original, group: list[ArrangedClip]):
        from pydub import AudioSegment
        active_ends = [
            *(c.dual_original_end_ms for c in group if not c.original_deleted and c.dual_original_end_ms is not None),
            *(c.dual_tts_end_ms for c in group if not c.tts_deleted and c.path is not None and c.dual_tts_end_ms is not None),
        ]
        group_duration_ms = max(0, max(active_ends, default=0) + sum(max(0, c.trailing_gap_ms) for c in group))
        canvas = AudioSegment.silent(duration=group_duration_ms, frame_rate=original.frame_rate).set_channels(2)
        group_clips: list[tuple[ArrangedClip, int, int]] = []
        for clip in group:
            local_starts, local_ends = [], []
            if not clip.original_deleted and clip.dual_original_start_ms is not None and clip.dual_original_end_ms is not None:
                ref_start = max(0, clip.reference_start_ms)
                ref_end = max(ref_start, min(len(original), clip.reference_end_ms))
                orig_seg = AudioSegment.from_file(clip.original_path) if clip.original_path is not None else original[ref_start:ref_end]
                if clip.original_sample_end_ms is not None:
                    orig_seg = orig_seg[clip.original_sample_start_ms:clip.original_sample_end_ms]
                orig_target_ms = max(1, clip.dual_original_end_ms - clip.dual_original_start_ms)
                if len(orig_seg):
                    orig_seg, orig_tempo = self._stretch_audio(orig_seg, orig_target_ms, clip.index, "原声")
                    orig_seg = orig_seg.set_frame_rate(original.frame_rate).set_channels(2)[:orig_target_ms]
                    canvas = canvas.overlay(orig_seg, position=max(0, clip.dual_original_start_ms))
                    local_starts.append(clip.dual_original_start_ms)
                    local_ends.append(clip.dual_original_start_ms + len(orig_seg))
                    print(f"[arrangement] 片段 {clip.index} 原声块: {clip.dual_original_start_ms}–{clip.dual_original_end_ms} ms，语速 {orig_tempo:.2f}×")
            if not clip.tts_deleted and clip.path is not None and clip.dual_tts_start_ms is not None and clip.dual_tts_end_ms is not None:
                tts_target_ms = max(1, clip.dual_tts_end_ms - clip.dual_tts_start_ms)
                tts_seg, tts_tempo = self._prepare_clip_audio(original, clip, origin_shift_ms=0, target_ms=tts_target_ms)
                tts_seg = tts_seg.set_frame_rate(original.frame_rate).set_channels(2)[:tts_target_ms]
                canvas = canvas.overlay(tts_seg, position=max(0, clip.dual_tts_start_ms))
                local_starts.append(clip.dual_tts_start_ms)
                local_ends.append(clip.dual_tts_start_ms + len(tts_seg))
                print(f"[arrangement] 片段 {clip.index} TTS块: {clip.dual_tts_start_ms}–{clip.dual_tts_end_ms} ms，语速 {tts_tempo:.2f}×，响度 {clip.volume:.0%}")
            if local_starts:
                group_clips.append((clip, min(local_starts), max(local_ends)))
        return canvas, group_clips

    def render(self, plan: ArrangementPlan) -> ArrangementAudioResult:
        from pydub import AudioSegment
        original = AudioSegment.from_file(plan.source_audio).set_channels(2)
        silence = AudioSegment.silent(duration=max(0, plan.gap_ms), frame_rate=original.frame_rate).set_channels(2)
        groups: dict[str, list[ArrangedClip]] = {}
        for clip in plan.clips:
            groups.setdefault(clip.block_id or f"clip-{clip.index}", []).append(clip)
        parts: list[bytes] = []
        rendered_clips: list[ArrangedClip] = []
        cursor = 0
        output_ms = 0
        used_raw_original = False
        for group in groups.values():
            has_dual_layout = any(c.dual_original_start_ms is not None or c.dual_tts_start_ms is not None for c in group)
            if has_dual_layout:
                uses_segment_files = any(c.original_path is not None for c in group)
                if not uses_segment_files:
                    used_raw_original = True
                    group_ref_start = min(max(0, c.reference_start_ms) for c in group)
                    group_ref_end = max(group_ref_start, *(min(len(original), c.reference_end_ms) for c in group))
                    if group_ref_start > cursor:
                        leading = original[cursor:group_ref_start]
                        parts.append(leading.raw_data)
                        output_ms += len(leading)
                    cursor = max(cursor, group_ref_end)
                canvas, group_clips = self._render_coarse_group(original, group)
                if len(canvas):
                    parts.append(canvas.raw_data)
                for clip, local_start, local_end in group_clips:
                    rendered_clips.append(replace(clip, start_ms=output_ms + local_start, end_ms=output_ms + local_end))
                output_ms += len(canvas)
            else:
                used_raw_original = True
                for clip in group:
                    ref_start = max(0, clip.reference_start_ms)
                    ref_end = max(ref_start, min(len(original), clip.reference_end_ms))
                    if ref_start > cursor:
                        leading = original[cursor:ref_start]
                        parts.append(leading.raw_data)
                        output_ms += len(leading)
                    orig_slice = original[ref_start:ref_end]
                    audio, tempo = self._prepare_clip_audio(original, clip, origin_shift_ms=0)
                    target_ms = max(1, clip.end_ms - clip.start_ms)
                    audio = audio.set_frame_rate(original.frame_rate).set_channels(2)[:target_ms]
                    parts.append(orig_slice.raw_data)
                    parts.append(silence.raw_data)
                    parts.append(audio.raw_data)
                    seg_end_ms = output_ms + len(orig_slice) + len(silence) + len(audio)
                    rendered_clips.append(replace(clip, start_ms=output_ms, end_ms=seg_end_ms))
                    print(f"[arrangement] 片段 {clip.index} (保留原声): 原声 {len(orig_slice)} ms + 间隔 {len(silence)} ms + TTS {len(audio)} ms，语速 {tempo:.2f}×，响度 {clip.volume:.0%}")
                    output_ms = seg_end_ms
                    cursor = max(cursor, ref_end)
        if used_raw_original and cursor < len(original):
            parts.append(original[cursor:].raw_data)
        return ArrangementAudioResult(original._spawn(b"".join(parts)), tuple(rendered_clips), 0)


_AUDIO_ADAPTERS = {
    "pydub": PydubArrangementAdapter,
    "overlay": PydubArrangementAdapter,
    "dual": DualArrangementAdapter,
}


def create_arrangement_audio_adapter(name: str = "pydub") -> ArrangementAudioAdapter:
    try:
        return _AUDIO_ADAPTERS[name]()
    except KeyError as exc:
        raise ArrangementError(f"未知编排音频实现：{name}") from exc


def write_arrangement_result(result: ArrangementAudioResult, directory: Path, snapshot: dict) -> dict:
    """Persist final artifacts separately from audio transformation and orchestration."""
    from ..storage import initialize_project
    initialize_project(directory)
    directory.mkdir(parents=True, exist_ok=True)
    wav, mp3 = directory / "output.wav", directory / "output.mp3"
    result.audio.export(wav, format="wav")
    result.audio.export(mp3, format="mp3", bitrate="192k")
    save_json(str(directory / "snapshot.json"), snapshot)
    save_json(str(directory / "audit.json"), [{"index": c.index, "path": str(c.path) if c.path else "", "start_ms": c.start_ms,
                                               "end_ms": c.end_ms, "output_start_ms": c.start_ms + result.origin_shift_ms,
                                               "output_end_ms": c.end_ms + result.origin_shift_ms,
                                               "volume": c.volume} for c in result.clips])
    def stamp(ms):
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02}.{ms % 1000:03}"
    vtt = "WEBVTT\n\n" + "\n\n".join(f"{clip.index}\n{stamp(clip.start_ms + result.origin_shift_ms)} --> {stamp(clip.end_ms + result.origin_shift_ms)}\n{clip.text}" for clip in result.clips) + "\n"
    path = save_text(str(directory / "final.vtt"), vtt)
    return {"output_wav": str(wav), "output_mp3": str(mp3), "final_vtt": path}
