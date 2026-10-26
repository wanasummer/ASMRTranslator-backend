"""Audio joiner - concatenates, duals, syncs segments, and merges chunk outputs."""

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, overload

from .io import write_json


# Ensure ffmpeg is on PATH before pydub imports
_ffmpeg_dir = str(Path(__file__).parent.parent / "ffmpeg_bin")
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ.setdefault("PATH", "")
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ["PATH"]


from pydub import AudioSegment
from .io import write_json


def get_ffmpeg_path() -> str:
    local_path = Path(__file__).parent.parent / "ffmpeg_bin" / "ffmpeg.exe"
    if local_path.exists():
        return str(local_path)
    return "ffmpeg"


def _time_stretch(audio: AudioSegment, target_ms: int) -> AudioSegment:
    """Time-stretch audio to target duration while preserving pitch (ffmpeg atempo)."""
    current_ms = len(audio)
    if current_ms <= target_ms:
        return audio

    ratio = current_ms / target_ms

    # atempo max = 2.0, chain multiple if needed
    if ratio <= 2.0:
        filter_str = f"atempo={ratio:.5f}"
    else:
        # e.g. ratio=3 → atempo=2,atempo=1.5
        factors = []
        r = ratio
        while r > 2.0:
            factors.append("atempo=2.0")
            r /= 2.0
        factors.append(f"atempo={r:.5f}")
        filter_str = ",".join(factors)

    ffmpeg = get_ffmpeg_path()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)

    try:
        in_tmp = tempfile.mktemp(suffix=".wav")
        audio.export(in_tmp, format="wav")
        subprocess.run([
            ffmpeg, "-y", "-i", in_tmp,
            "-filter:a", filter_str,
            "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
            tmp_path
        ], capture_output=True, check=True)
        stretched = AudioSegment.from_wav(tmp_path)
        os.unlink(in_tmp)
        return stretched
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def fit_translation_to_original(audio, content_ms, orig_ms, max_rate):
    """Fit `audio` so its `content_ms` speech ends ≤ orig_ms (cap-accelerate).

    content_ms == len(audio) for plain TTS; for phantom DSP output pass the raw
    voice length (the audio carries a reverb tail).  need ≤ max_rate → stretch to
    exactly orig_ms; need > max_rate → stretch at max_rate then truncate overflow.
    """
    if content_ms <= orig_ms:
        return audio
    needed = content_ms / orig_ms
    rate = min(needed, max_rate)
    if rate > 1.0:
        audio = _time_stretch(audio, max(1, int(len(audio) / rate)))
    if len(audio) > orig_ms:
        audio = audio[:orig_ms]
    return audio


def apply_follow_translation_cap(
    segments: list[dict],
    delay_ms: Optional[int] = None,
    max_overhang_ms: int = 3000,
    wav_params: list[str] | None = None,
) -> list[dict]:
    """Pluggable follow-translation length cap — a sound-sync (声音同步) property.

    "跟随翻译" lets the original stretch so the Chinese translation plays to
    completion. As a fallback (保底), the combined per-segment audio is truncated
    when the translation runs more than ``max_overhang_ms`` past its allowed
    window. The window is the pink segment's TTS budget (``max_allowed_tts_ms``,
    i.e. P_end − O_start) when the segment carries it, else the original slice
    length (``orig_ms``) — either way the cap target is ``window + max_overhang_ms``.
    Translations within that window always play in full. Runs after the mode
    build so it covers plain sync AND phantom uniformly (neither builder
    truncates internally). ``delay_ms`` is the TTS trigger offset inside the
    segment (0 for the old no-delay model); ``None`` reads each segment's own
    ``tts_delay_ms``. ``seg["vc_file"]`` is rewritten in place.
    """
    for seg in segments:
        comb_file = seg.get("vc_file") or seg.get("tts_file")
        orig_file = seg.get("segment_file")
        if seg.get("skip_tts"):
            if not orig_file or not Path(orig_file).exists():
                raise FileNotFoundError(f"seg {seg.get('index', '?')} original input missing")
            seg["vc_file"] = orig_file
            continue
        if (not comb_file or not orig_file
                or not Path(comb_file).exists() or not Path(orig_file).exists()):
            raise FileNotFoundError(f"seg {seg.get('index', '?')} sync input missing")
        try:
            orig_ms = len(AudioSegment.from_file(orig_file))
            comb_ms = len(AudioSegment.from_file(comb_file))
        except Exception:
            raise
        # The translation is the long side only when the combined audio is longer
        # than the original; otherwise it already fits and must never be capped.
        if comb_ms <= orig_ms:
            continue
        delay = delay_ms if delay_ms is not None else seg.get("tts_delay_ms", 0)
        budget = seg.get("max_allowed_tts_ms")
        window_ms = budget if (budget and budget > 0) else orig_ms
        trans_ms = comb_ms - delay
        cap_ms = window_ms + max_overhang_ms
        if trans_ms > cap_ms and cap_ms > 0:
            target = delay + cap_ms
            if target < comb_ms:
                overhang = trans_ms - window_ms
                print(f"       [sync·截断] seg {seg.get('index', '?')}: 翻译 {trans_ms}ms "
                      f"超过窗口 {window_ms}ms 达 {overhang}ms（>保底 {max_overhang_ms}ms），"
                      f"截断到 {target}ms")
                audio = AudioSegment.from_file(comb_file)
                audio[:target].export(comb_file, format="wav",
                                      parameters=wav_params or ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
    return segments


def enforce_tts_budget(
    segments: list[dict],
    max_speed: float = 2.0,
    pad_ms: int = 3000,
    mode: str = "follow_original",
    max_workers: int = 4,
) -> list[dict]:
    """Cap each segment's TTS to its pink-segment time budget (two fallbacks).

    ``budget_ms = seg["max_allowed_tts_ms"]`` = P_end − O_start, the time from
    the TTS trigger to the end of its pink slice. When the TTS VOICE is longer
    than the budget, one of the existing fallback strategies runs:

      follow_original (fallback 1, default): time-stretch the TTS (atempo,
      pitch-preserving) up to ``max_speed``× to fit the budget; overflow past
      the budget is hard-cut. Applies to replace / dual / sync-follow_original /
      phantom.

      follow_translation (fallback 2): the effective budget is ``budget_ms +
      pad_ms`` (default 3000) — the follow_translation build pads the segment
      with trailing silence so the TTS may finish within that window; only
      overflow past ``budget_ms + pad_ms`` is hard-cut.

    For phantom DSP audio the voice length is read from ``seg["phantom_voice_src"]``
    (the raw pre-DSP TTS) so the reverb tail is not counted against the budget;
    the whole DSP'd audio is scaled/truncated by the same factor. ``skip_tts``
    segments are left untouched. Rewrites ``seg["tts_file"]`` to point at the
    capped WAV.
    """
    def _one(seg):
        idx = seg.get("index", 0)
        budget = seg.get("max_allowed_tts_ms")
        tts_file = seg.get("tts_file")
        if seg.get("skip_tts") or not budget or budget <= 0:
            return seg
        if not tts_file or not Path(tts_file).exists():
            raise FileNotFoundError(f"seg {idx} TTS budget input missing: {tts_file}")
        tempo_path = os.path.join(os.path.dirname(tts_file), f"tts_tempo_{idx:04d}.json")
        try:
            effective = budget + pad_ms if mode == "follow_translation" else budget
            voice_src = seg.get("phantom_voice_src") or tts_file
            voice_ms = len(AudioSegment.from_file(voice_src))
            if voice_ms <= effective:
                write_json(tempo_path, {"rate": 1.0})
                return seg
            audio = AudioSegment.from_file(tts_file)
            rate = min(voice_ms / effective, max_speed)
            if rate > 1.0:
                audio = _time_stretch(audio, max(1, int(len(audio) / rate)))
            if len(audio) > effective:
                audio = audio[:effective]
            out_path = os.path.join(
                os.path.dirname(tts_file), f"tts_budget_{idx:04d}.wav")
            audio.export(out_path, format="wav",
                         parameters=["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1"])
            seg["tts_file"] = out_path
            write_json(tempo_path, {"rate": rate})
            print(f"       [budget] seg {idx}: 人声 {voice_ms}ms > 预算 {effective}ms"
                  f"（{mode}）→ {'加速%.2fx' % rate if rate > 1.0 else '直接截断'} "
                  f"至 {len(audio)}ms")
        except Exception:
            raise
        return seg

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_one, segments))
    return segments


def _cache_fresh(path: str, sources: list[str]) -> bool:
    """True when `path` exists, is non-trivial, and is newer than every source.

    Keeps the dual/sync build cache from going stale when the underlying TTS or
    original changes (e.g. after a loudness-alignment rerun at a new volume).
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 1000:
            return False
        mtime = os.path.getmtime(path)
        return all(mtime >= os.path.getmtime(s) for s in sources)
    except OSError:
        return False


def build_dual_segments(
    segments: list[dict],
    output_dir: str,
    gap_ms: int = 200,
    max_workers: int = 4,
    wav_params: list[str] | None = None,
) -> list[dict]:
    """Build [original → gap → Chinese TTS] dual segments for retain_original mode.

    TTS loudness is already aligned per-segment by modules.audio_loudness, so no
    gain is applied here. Uses ThreadPoolExecutor to process segments in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _build_one(seg: dict) -> dict:
        orig_file = seg.get("segment_file")
        tts_file = seg.get("tts_file")
        if not orig_file or not Path(orig_file).exists():
            raise FileNotFoundError(f"seg {seg.get('index')} original input missing")
        if seg.get("skip_tts"):
            seg["vc_file"] = orig_file
            return seg
        if not tts_file or not Path(tts_file).exists():
            raise FileNotFoundError(f"seg {seg.get('index')} TTS input missing")

        combined_path = os.path.join(output_dir, f"dual_{seg['index']:04d}.wav")

        if _cache_fresh(combined_path, [orig_file, tts_file]):
            seg["vc_file"] = combined_path
            return seg

        try:
            orig = AudioSegment.from_file(orig_file)
            tts = AudioSegment.from_file(tts_file)
            if wav_params:
                tts = tts.set_frame_rate(orig.frame_rate).set_channels(orig.channels)
            silence = AudioSegment.silent(duration=gap_ms, frame_rate=orig.frame_rate).set_channels(orig.channels)
            combined = orig + silence + tts
            combined.export(combined_path, format="wav",
                            parameters=wav_params or ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
            seg["vc_file"] = combined_path
            print(f"       [dual] seg {seg['index']}: orig({len(orig)}ms) + gap({gap_ms}ms) + tts({len(tts)}ms) = {len(combined)}ms")

        except Exception:
            raise

        return seg

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_build_one, segments))

    return segments


def build_sync_segments(
    segments: list[dict],
    output_dir: str,
    sync_mode: str = "follow_original",
    follow_original_rate: float = 1.15,
    max_workers: int = 4,
    wav_params: list[str] | None = None,
) -> list[dict]:
    """Build sync segments — TTS overlays original audio simultaneously.

    sync_mode="follow_original": TTS is cap-accelerated to fit ≤ original's duration
                                 (up to `follow_original_rate`×; overflow truncated),
                                 then mixed on top. Original length unchanged.

    sync_mode="follow_translation": Original is padded with silence to fit TTS duration,
                                    then TTS is mixed on top.

    TTS loudness is already aligned per-segment by modules.audio_loudness, so no
    gain is applied here. Note that the overlay SUMS orig + TTS, so a TTS matched
    to the original lands ~3 dB above the original in the mix.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _build_one(seg: dict) -> dict:
        orig_file = seg.get("segment_file")
        tts_file = seg.get("tts_file")
        if not orig_file or not Path(orig_file).exists():
            raise FileNotFoundError(f"seg {seg.get('index')} original input missing")
        if seg.get("skip_tts"):
            seg["vc_file"] = orig_file
            return seg
        if not tts_file or not Path(tts_file).exists():
            raise FileNotFoundError(f"seg {seg.get('index')} TTS input missing")

        combined_path = os.path.join(output_dir, f"sync_{seg['index']:04d}.wav")

        if _cache_fresh(combined_path, [orig_file, tts_file]):
            seg["vc_file"] = combined_path
            return seg

        try:
            orig = AudioSegment.from_file(orig_file)
            tts = AudioSegment.from_file(tts_file)

            if wav_params:
                tts = tts.set_frame_rate(orig.frame_rate).set_channels(orig.channels)

            orig_ms = len(orig)
            tts_ms = len(tts)

            if sync_mode == "follow_original":
                # Budget = P_end - O_start (the pink slice's TTS window); the TTS
                # is cap-accelerated into it, then triggered at the orange start.
                budget_ms = seg.get("max_allowed_tts_ms") or orig_ms
                if tts_ms > budget_ms:
                    tts = fit_translation_to_original(tts, tts_ms, budget_ms, follow_original_rate)
                    print(f"       [sync·原声] seg {seg['index']}: TTS {tts_ms}ms → {len(tts)}ms "
                          f"(need {tts_ms/budget_ms:.2f}x, cap {follow_original_rate:.2f}x, "
                          f"budget {budget_ms}ms)")
                else:
                    print(f"       [sync·原声] seg {seg['index']}: TTS {tts_ms}ms ≤ budget {budget_ms}ms, no stretch")
                mixed = orig.overlay(tts, position=seg.get("tts_delay_ms", 0))

            elif sync_mode == "follow_translation":
                delay = seg.get("tts_delay_ms", 0)
                need = delay + tts_ms
                if orig_ms < need:
                    pad = AudioSegment.silent(duration=need - orig_ms, frame_rate=orig.frame_rate).set_channels(orig.channels)
                    orig = orig + pad
                    print(f"       [sync·翻译] seg {seg['index']}: orig {orig_ms}ms → {need}ms (padded)")
                else:
                    print(f"       [sync·翻译] seg {seg['index']}: orig {orig_ms}ms ≥ need {need}ms, no pad")
                mixed = orig.overlay(tts, position=delay)

            else:
                raise ValueError(f"Unknown sync_mode: {sync_mode}")

            mixed.export(combined_path, format="wav",
                         parameters=wav_params or ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
            seg["vc_file"] = combined_path

        except Exception:
            raise

        return seg

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(_build_one, segments))

    return segments


@dataclass(frozen=True)
class SegmentPlacement:
    block_id: str
    start_ms: int
    end_ms: int
    subtitle_start_ms: int
    subtitle_end_ms: int


@dataclass(frozen=True)
class JoinedAudio:
    path: str
    placements: tuple[SegmentPlacement, ...]


def _segment_placement(seg: dict, start_ms: int, duration_ms: int) -> SegmentPlacement:
    """Use source speech in retained audio; direct TTS occupies the full file."""
    audio_file = seg.get("vc_file") or seg.get("tts_file")
    if not seg.get("skip_tts") and audio_file == seg.get("tts_file"):
        start, end = 0, duration_ms
    else:
        start = int(seg["original_start_ms"]) - int(seg["start_ms"])
        end = int(seg["original_end_ms"]) - int(seg["start_ms"])
    start = min(duration_ms, max(0, start))
    end = min(duration_ms, max(start, end))
    return SegmentPlacement(str(seg.get("segment_id", seg["index"])),
                            start_ms, start_ms + duration_ms,
                            start_ms + start, start_ms + end)


def _write_join_audit(output_path: str, valid_count: int, skipped: list[dict],
                      placements: list[SegmentPlacement]) -> str:
    """Persist omissions beside the joined audio; source slices remain intact."""
    return write_json(str(Path(output_path).with_suffix(".join_audit.json")), {
        "schemaVersion": 1,
        "stage": "audio.concat",
        "minimum_duration_ms": 50,
        "joined_count": valid_count,
        "skipped_count": len(skipped),
        "skipped_segments": skipped,
        "placements": [asdict(item) for item in placements],
    })


@overload
def join_audio(segments: list[dict], output_path: str, crossfade_ms: int = 0,
               *, return_timeline: Literal[False] = False) -> str: ...


@overload
def join_audio(segments: list[dict], output_path: str, crossfade_ms: int = 0,
               *, return_timeline: Literal[True]) -> JoinedAudio: ...


def join_audio(
    segments: list[dict],
    output_path: str,
    crossfade_ms: int = 0,
    *, return_timeline: bool = False,
) -> str | JoinedAudio:
    """Join in order, omitting decoded segments shorter than 50 ms.

    Inputs are not mutated. Missing or undecodable files still fail; if every
    segment is omitted, no final audio is exported. Omissions are audited.
    Opt into an immutable placement result for output-timeline subtitles; the
    default return value remains the output path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    combined = None
    valid_count = 0
    skipped: list[dict] = []
    placements: list[SegmentPlacement] = []

    for seg in segments:
        audio_file = seg.get("vc_file") or seg.get("tts_file")
        if not audio_file or not Path(audio_file).exists():
            raise FileNotFoundError(f"seg {seg['index']} file not found: {audio_file}")

        try:
            seg_audio = AudioSegment.from_file(audio_file)
            if len(seg_audio) < 50:
                skipped.append({"index": seg["index"], "file": str(audio_file),
                                "duration_ms": len(seg_audio),
                                "reason": "decoded_duration_below_minimum"})
                print(f"       [join-skip] seg {seg['index']}: decoded audio "
                      f"{len(seg_audio)}ms < 50ms, skipped ({audio_file})")
                continue

            if combined is None:
                combined = seg_audio
                start_ms = 0
            else:
                # 虚拟淡化：本段标记 overlap_ms 时，上一段的淡化尾巴保留，
                # 下一段开头重叠其上，再接下一段剩余部分——总长扣除 overlap。
                before_ms = len(combined)
                overlap = int(seg.get("overlap_ms") or 0)
                if overlap > 0 and len(combined) > overlap and len(seg_audio) > 0:
                    overlap = min(overlap, len(seg_audio))
                    body = combined[:-overlap]
                    tail = combined[-overlap:].set_channels(2)
                    head_over = seg_audio[:overlap].set_channels(2)
                    rest = seg_audio[overlap:].set_channels(2)
                    combined = (body + tail.overlay(head_over, position=0) + rest)
                    start_ms = before_ms - overlap
                elif crossfade_ms > 0:
                    crossfade = min(crossfade_ms, len(combined) // 2, len(seg_audio) // 2)
                    combined = combined.append(seg_audio, crossfade=crossfade)
                    start_ms = before_ms - crossfade
                else:
                    combined = combined + seg_audio
                    start_ms = before_ms

            if return_timeline:
                placements.append(_segment_placement(seg, start_ms, len(combined) - start_ms))
            valid_count += 1
        except Exception:
            raise

    audit_path = _write_join_audit(output_path, valid_count, skipped, placements)
    print(f"       [audio.concat] Joined {valid_count}, skipped {len(skipped)} "
          f"segments; audit: {audit_path}")
    if combined is None:
        raise RuntimeError("No valid audio segments to join")

    print(f"       Joined {valid_count} segments ({len(combined)} ms total)")

    # In replace mode every segment is mono TTS, so the mix would be mono; up-mix
    # in pydub (level-preserving) before export — ffmpeg -ac 2 on mono would apply
    # a −3 dB energy normalization and shift loudness off the aligned target.
    combined = combined.set_channels(2)
    combined.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
    mp3_path = output_path.replace(".wav", ".mp3")
    combined.export(mp3_path, format="mp3", bitrate="192k")

    return JoinedAudio(output_path, tuple(placements)) if return_timeline else output_path


def concat_audio_files(
    file_paths: list[str],
    output_path: str,
    crossfade_ms: int = 0,
) -> str:
    """Concatenate multiple pre-built audio files into one final output.

    Each file is a complete chunk output (already has all internal segments
    joined).  This simply appends them sequentially with optional crossfade.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    combined = None
    valid_count = 0

    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"       [skip] chunk file not found: {fp}")
            continue
        if os.path.getsize(fp) < 100:
            print(f"       [skip] chunk file too small: {fp}")
            continue

        try:
            seg = AudioSegment.from_file(fp)
            if len(seg) < 50:
                continue

            if combined is None:
                combined = seg
            else:
                if crossfade_ms > 0:
                    crossfade = min(crossfade_ms, len(combined) // 2, len(seg) // 2)
                    combined = combined.append(seg, crossfade=crossfade)
                else:
                    combined = combined + seg

            valid_count += 1
        except Exception as e:
            print(f"       [error] concat {fp}: {e}")

    if combined is None:
        raise RuntimeError("No valid chunk files to concatenate")

    print(f"       Concatenated {valid_count} chunk files ({len(combined)} ms total)")

    combined = combined.set_channels(2)  # see join_audio: mono→stereo must stay level-preserving
    combined.export(output_path, format="wav",
                    parameters=["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
    mp3_path = output_path.replace(".wav", ".mp3")
    combined.export(mp3_path, format="mp3", bitrate="192k")

    return output_path
