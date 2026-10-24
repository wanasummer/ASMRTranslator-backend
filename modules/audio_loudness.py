"""Per-segment loudness matching: align TTS audio to the matched original segment.

TTS backends output speech at their own default loudness, which varies per
backend/voice and usually differs from the Japanese original. This module
normalizes each TTS segment so its RMS loudness equals the original segment's
loudness times a user factor (tts_volume: 1.0 = match the original exactly).

The measurement is windowed RMS in dBFS using stdlib `audioop` (with a pure
Python fallback) — no new dependencies, no ffmpeg round-trip for analysis.
Only the final export of the normalized WAV goes through ffmpeg (same as the
rest of the pipeline).
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure ffmpeg is on PATH before pydub imports (same pattern as audio_joiner)
_ffmpeg_dir = str(Path(__file__).parent.parent / "ffmpeg_bin")
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ.setdefault("PATH", "")
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ["PATH"]

from pydub import AudioSegment

try:
    import audioop as _audioop
except ImportError:
    # audioop is deprecated and removed in Python 3.13+; fall back to pure Python
    _audioop = None

# Full-scale amplitude reference per pydub sample width (bytes)
_FULL_SCALE = {1: 128.0, 2: 32768.0, 4: 2147483648.0}

# Export parameters used everywhere in the pipeline (24 kHz mono s16 PCM WAV).
# TTS stays mono here: ffmpeg's -ac 2 mono→stereo upmix would apply a −3 dB
# energy normalization and silently break loudness matching. The upmix to
# centered stereo happens in-memory via pydub when TTS meets the stereo original.
_WAV_PARAMS = ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1"]


def _rms(chunk, width):
    """RMS of one sample window. `chunk` is an array.array slice of ints."""
    if _audioop is not None:
        return _audioop.rms(chunk, width)
    n = len(chunk)
    if n == 0:
        return 0.0
    s = 0.0
    for v in chunk:
        s += v * v
    return math.sqrt(s / n)


def _peak(chunk, width):
    """Peak absolute sample value of one chunk (full-scale units)."""
    if _audioop is not None:
        return _audioop.max(chunk, width)
    return max((abs(v) for v in chunk), default=0)


def loudness_db(audio, window_ms=20, rel_gate_db=40.0, floor_db=-60.0):
    """Perceived-loudness proxy: RMS of a segment in dBFS.

    Windows are 20 ms long. A *relative* gate keeps everything within
    `rel_gate_db` of the loudest window (and above an absolute `floor_db`), so
    whisper-range ASMR content is preserved while trailing silence/room tone is
    excluded. If no window passes, falls back to all windows. Returns None only
    for pure digital silence (all-zero samples) or unreadable audio.
    """
    try:
        audio = audio.set_channels(1)
        samples = audio.get_array_of_samples()
        width = audio.sample_width
    except Exception:
        return None
    full_scale = _FULL_SCALE.get(width)
    if full_scale is None:
        return None
    n = len(samples)
    if n == 0:
        return None

    win = max(1, int(audio.frame_rate * window_ms / 1000))
    rms_vals = []
    for start in range(0, n, win):
        r = _rms(samples[start:start + win], width)
        if r > 0:
            rms_vals.append(r)
    if not rms_vals:
        return None

    dbs = [20.0 * math.log10(r / full_scale) for r in rms_vals]
    gate = max(max(dbs) - rel_gate_db, floor_db)
    kept = [r for r, db in zip(rms_vals, dbs) if db >= gate]
    if not kept:
        kept = rms_vals

    rms_lin = math.sqrt(sum(r * r for r in kept) / len(kept))
    if rms_lin <= 0:
        return None
    return 20.0 * math.log10(rms_lin / full_scale)


def _peak_db(audio):
    """Peak level of a segment in dBFS (used by the limiter)."""
    full_scale = _FULL_SCALE.get(audio.sample_width)
    if full_scale is None:
        return None
    try:
        samples = audio.set_channels(1).get_array_of_samples()
    except Exception:
        return None
    p = _peak(samples, audio.sample_width)
    return 20.0 * math.log10(max(p, 1) / full_scale)


def compute_gain_db(orig_db, tts_db, tts_volume, match_clamp_db=20.0,
                    headroom_db=0.5, tts_peak_db=None):
    """Gain (dB) to apply to a TTS segment so it matches the original.

    The user's explicit `tts_volume` is applied in full; only the automatic
    normalization component is clamped (so 10% really means -20 dB rather than
    being swallowed by the clamp). The result is limited so the boosted peak
    never exceeds `-headroom_db` dBFS. Returns -inf when volume is 0 (silence).
    """
    if orig_db is None or tts_db is None:
        return 0.0
    if tts_volume <= 0:
        return -float("inf")

    match_gain = max(-match_clamp_db, min(match_clamp_db, orig_db - tts_db))
    gain = match_gain + 20.0 * math.log10(tts_volume)
    if tts_peak_db is not None:
        gain = min(gain, -headroom_db - tts_peak_db)
    return gain


def _original_core(orig, seg):
    """Slice the unpadded speech-only region out of the original segment.

    split_audio pads each segment by 50 ms (audio_splitter.py); measuring the
    padded region would drag the loudness down with the neighbor's speech or
    silence. Falls back to the full segment when timing fields are missing or
    the offsets are invalid.
    """
    orig_start = seg.get("original_start_ms")
    orig_end = seg.get("original_end_ms")
    seg_start = seg.get("start_ms")
    if orig_start is None or orig_end is None or seg_start is None:
        return orig
    offset = orig_start - seg_start
    dur = orig_end - orig_start
    if offset < 0 or dur <= 0 or offset >= len(orig):
        return orig
    return orig[offset:offset + dur]


def _output_path(seg, tts_volume):
    """Path for the normalized file, with the volume baked into the name so a
    slider change naturally invalidates the cache key."""
    pct = max(0, round(tts_volume * 100))
    return os.path.join(
        os.path.dirname(seg["tts_file"]),
        f"tts_loud_{seg['index']:04d}_v{pct:03d}.wav",
    )


def _cache_fresh(path, sources):
    """True when `path` exists, is non-trivial, and is newer than all sources."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 100:
            return False
        mtime = os.path.getmtime(path)
        return all(mtime >= os.path.getmtime(s) for s in sources)
    except OSError:
        return False


def _write_silence(tts, out_path):
    """Write an all-zero WAV of the same length as the TTS (volume 0 = silence)."""
    silent = AudioSegment.silent(duration=len(tts), frame_rate=tts.frame_rate)
    silent.export(out_path, format="wav", parameters=_WAV_PARAMS)


def _align_one(seg, tts_volume):
    orig_file = seg.get("segment_file")
    tts_file = seg.get("tts_file")
    if not orig_file or not Path(orig_file).exists():
        raise FileNotFoundError(f"seg {seg.get('index')} original input missing")
    if seg.get("skip_tts"):
        return seg
    if not tts_file or not Path(tts_file).exists():
        raise FileNotFoundError(f"seg {seg.get('index')} TTS input missing")

    out_path = _output_path(seg, tts_volume)
    if _cache_fresh(out_path, [tts_file, orig_file]):
        seg["tts_file"] = out_path
        return seg

    try:
        orig = AudioSegment.from_file(orig_file)
        tts = AudioSegment.from_file(tts_file)

        if tts_volume <= 0:
            _write_silence(tts, out_path)
            seg["tts_file"] = out_path
            print(f"       [loudness] seg {seg.get('index')}: volume {tts_volume} → silence")
            return seg

        orig_db = loudness_db(_original_core(orig, seg))
        tts_db = loudness_db(tts)
        if orig_db is None or tts_db is None:
            return seg  # nothing to match against — leave as-is

        gain_db = compute_gain_db(orig_db, tts_db, tts_volume,
                                  tts_peak_db=_peak_db(tts))
        if abs(gain_db) < 0.1:
            return seg  # already aligned — no write needed

        tts.apply_gain(gain_db).export(out_path, format="wav", parameters=_WAV_PARAMS)
        seg["tts_file"] = out_path
        print(f"       [loudness] seg {seg.get('index')}: orig {orig_db:.1f} dB / "
              f"tts {tts_db:.1f} dB / gain {gain_db:+.1f} dB (vol {tts_volume})")
    except Exception:
        raise

    return seg


def align_tts_to_original(segments, tts_volume=1.0, max_workers=4):
    """Normalize each segment's TTS loudness to its matched original segment.

    Runs after TTS generation and before VC/build/join so every downstream
    stage consumes the aligned audio. Mutates `seg["tts_file"]` to point at the
    normalized WAV (or writes silence when volume is 0). ``skip_tts`` segments
    retain their original audio and do not require a TTS file. Segments that
    can't be measured (pure-silence original, failed TTS placeholder) are left
    unchanged.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda seg: _align_one(seg, tts_volume), segments))
    return segments
