"""Phantom Whisper build mode: a near-imperceptible Chinese voice overlaid on pure stereo original.

The user listens for the original Japanese ASMR; the Chinese translation surfaces as
a "thought echo" — quiet, low-passed, convolution-reverberated, delayed, and centered.
This module builds each segment as:

    phantom_segment = stereo_original.overlay(processed_tts, position=delay_ms)

where processed_tts is the loudness-aligned mono TTS run through:
    lowpass (sibilance softening) → convolution reverb (ffmpeg afir) → fade in/out.

Level philosophy: the aligned TTS is already scaled to `tts_volume` (default 15%)
relative to the original by modules.audio_loudness. The reverb wet is then normalized
to `reverb_mix` × dry level, so the whisper stays well below the original throughout.
The original stereo is never touched or compressed (no ducking).

Note on afir: in the bundled ffmpeg build, afir's individual `dry=`/`wet=` paths are
unreliable (dry-only or wet-only output is silence), but the summed `dry=1:wet=1`
path works. So we run the summed afir, then recover the pure wet tail by subtracting
the padded lowpass (the known dry) in pydub. Wet RMS is normalized to `reverb_mix` ×
dry RMS so the reverb intensity is meaningful regardless of afir's internal gain.
TTS stays mono through the DSP; pydub up-mixes it to centered stereo (level-preserving)
when overlaid on the original.
"""

import math
import os
import random
import struct
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio_joiner import fit_translation_to_original
from .audio_splitter import get_ffmpeg_path

# Deterministic synthetic room IR (small-room tail, ~0.8 s at 24 kHz).
IR_DURATION_S = 0.8
IR_SEED = 42
IR_DECAY_S = 0.2
IR_PEAK = 0.3  # peak amplitude (fraction of full scale) — keeps convolution in range

_WAV_PARAMS = ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1"]


def _rms_lin(audio):
    """Plain RMS of the whole segment in sample units (not gated)."""
    samples = audio.set_channels(1).get_array_of_samples()
    if len(samples) == 0:
        return 0.0
    s = 0.0
    for v in samples:
        s += v * v
    return math.sqrt(s / len(samples))


def get_ir_path(output_dir):
    """Return the synthetic IR path, generating it lazily and deterministically."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "_ir_smallroom.wav")
    if not os.path.exists(path) or os.path.getsize(path) <= 100:
        _generate_ir(path)
    return path


def _generate_ir(path):
    """Write a deterministic convolution IR: seeded white noise × exponential decay."""
    rate = 24000
    n = int(IR_DURATION_S * rate)
    rnd = random.Random(IR_SEED)
    raw = [rnd.gauss(0, 1) * math.exp(-i / (IR_DECAY_S * rate)) for i in range(n)]
    peak = max(abs(x) for x in raw) or 1.0
    scale = IR_PEAK * 32767 / peak
    samples = [int(max(-32768, min(32767, x * scale))) for x in raw]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def _dsp_cmd(tts_path, ir_path, lowpass_hz, dry_path, full_path):
    """ffmpeg command: lowpass+asplit for dry; apad+afir(dry+wet) for the reverb tail.

    `full` is afir's summed dry+wet (individual dry/wet-only paths are silent in the
    bundled build); `apply_phantom_dsp` subtracts the padded dry to isolate the wet.
    """
    return [
        get_ffmpeg_path(), "-y",
        "-i", tts_path,
        "-i", ir_path,
        "-filter_complex",
        f"[0:a]lowpass=f={lowpass_hz},asplit=2[dry][r];"
        f"[r]apad=pad_dur={IR_DURATION_S}[p];"
        f"[p][1:a]afir=dry=1:wet=1:gtype=none:irnorm=0[full]",
        "-map", "[dry]", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", dry_path,
        "-map", "[full]", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", full_path,
    ]


def apply_phantom_dsp(tts_path, out_path, lowpass_hz, reverb_mix, ir_path,
                      fade_in_ms=120, fade_out_ms=200):
    """Run the DSP chain on one TTS file and write the processed mono WAV.

    dry + wet, where wet is recovered from afir's summed output by subtracting the
    padded lowpass (the known dry), then normalized to `reverb_mix` × dry RMS so the
    reverb intensity is meaningful regardless of afir's internal gain. The wet is the
    overlay base so its room tail rings out past the voice end (pydub truncates the
    overlaid segment, not the base). Fades melt the voice; the tail decays via the IR.
    """
    import warnings
    import audioop
    from pydub import AudioSegment  # deferred so IR/ffmpeg helpers stay import-safe

    # audioop is deprecated in 3.13 but stable through 3.11 (bundled runtime)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    tmpdir = tempfile.mkdtemp(prefix="phantom_dsp_")
    dry_path = os.path.join(tmpdir, "dry.wav")
    full_path = os.path.join(tmpdir, "full.wav")
    try:
        result = subprocess.run(
            _dsp_cmd(tts_path, ir_path, lowpass_hz, dry_path, full_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg DSP failed: {result.stderr.strip()[:300]}")

        dry = AudioSegment.from_file(dry_path)
        full = AudioSegment.from_file(full_path)

        if reverb_mix > 0:
            # wet = afir(dry+wet) − dry; pad dry to the tail-extended length first.
            # audioop.add demands EXACT byte equality: pydub's ms-rounding can
            # leave dry_padded 1ms (~24 samples) longer/shorter than full1, so
            # align the buffers explicitly before subtracting.
            pad = AudioSegment.silent(duration=max(0, len(full) - len(dry)),
                                      frame_rate=dry.frame_rate)
            dry_padded = (dry + pad).set_channels(1)
            full1 = full.set_channels(1)
            dry_bytes = audioop.mul(
                dry_padded.get_array_of_samples().tobytes(), 2, -1.0)
            full_bytes = full1.get_array_of_samples().tobytes()
            if len(full_bytes) > len(dry_bytes):
                dry_bytes += b"\x00" * (len(full_bytes) - len(dry_bytes))
            elif len(dry_bytes) > len(full_bytes):
                dry_bytes = dry_bytes[:len(full_bytes)]
            wet_bytes = audioop.add(full_bytes, dry_bytes, 2)
            wet = AudioSegment(wet_bytes, frame_rate=full1.frame_rate,
                               sample_width=full1.sample_width, channels=1)

            dry_rms = _rms_lin(dry)
            wet_rms = _rms_lin(wet)
            if dry_rms > 0 and wet_rms > 0:
                target = dry_rms * reverb_mix
                wet = wet.apply_gain(20.0 * math.log10(target / wet_rms))
            else:
                wet = AudioSegment.silent(duration=len(wet), frame_rate=wet.frame_rate)
            # wet is the base so its 400-1200 ms tail survives; overlay the faded voice on it
            voice = dry.fade_in(fade_in_ms).fade_out(fade_out_ms)
            mixed = wet.overlay(voice, position=0)
        else:
            mixed = dry.fade_in(fade_in_ms).fade_out(fade_out_ms)
        mixed.export(out_path, format="wav", parameters=_WAV_PARAMS)
    finally:
        for p in (dry_path, full_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def _cache_fresh(path, sources):
    """True when `path` exists, is non-trivial, and is newer than all sources."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= 100:
            return False
        mtime = os.path.getmtime(path)
        return all(mtime >= os.path.getmtime(s) for s in sources)
    except OSError:
        return False


def _build_one(seg, output_dir, delay_ms, reverb_mix, lowpass_hz,
               fade_in_ms, fade_out_ms, sync_mode, follow_original_rate, ir_path,
               orig_fade_ms=0, tts_fade_ms=0, virtual_fade=False, wav_params=None):
    orig_file = seg.get("segment_file")
    tts_file = seg.get("tts_file")
    if not orig_file or not Path(orig_file).exists():
        return seg
    if seg.get("skip_tts"):
        seg["vc_file"] = orig_file
        return seg
    if not tts_file or not Path(tts_file).exists():
        return seg

    # 1. DSP. When the handler chain already ran the phantom DSP on this segment
    # (seg["phantom_dsp_file"]), tts_file already points at the DSP'd audio —
    # skip re-running the chain and overlay it directly.
    pre_dsp = seg.get("phantom_dsp_file")
    if pre_dsp and Path(pre_dsp).exists():
        dsp_path = tts_file
    else:
        dsp_path = os.path.join(output_dir, f"phantom_tts_{seg['index']:04d}_lp{lowpass_hz}_rv{int(round(reverb_mix * 100)):03d}.wav")
        if not _cache_fresh(dsp_path, [tts_file, ir_path]):
            try:
                apply_phantom_dsp(tts_file, dsp_path, lowpass_hz, reverb_mix, ir_path,
                                  fade_in_ms, fade_out_ms)
            except Exception as e:
                print(f"       [phantom error] seg {seg['index']}: DSP failed ({e}) — using raw TTS")
                dsp_path = tts_file

    # 2. Delayed overlay onto the stereo original (cache per orig/dsp inputs)
    combined_path = os.path.join(output_dir, f"phantom_{seg['index']:04d}.wav")
    sources = [orig_file] + ([dsp_path] if dsp_path != tts_file else [tts_file])
    if not _cache_fresh(combined_path, sources):
        try:
            from pydub import AudioSegment
            orig = AudioSegment.from_file(orig_file)
            tts = AudioSegment.from_file(dsp_path)
            if wav_params:
                tts = tts.set_frame_rate(orig.frame_rate).set_channels(orig.channels)

            # The TTS is triggered at the segment's orange start (tts_delay_ms);
            # fall back to the phantom delay config for segments without geometry.
            req_delay = seg.get("tts_delay_ms", delay_ms)
            if sync_mode == "follow_translation":
                # Honor the requested delay and extend the original with trailing
                # silence so the phantom translation (voice + reverb tail) is never
                # cut off at the segment boundary — pydub overlay truncates at the
                # base length, so the base must be padded first. No truncation is
                # done here: the follow_translation length cap is a sound-sync
                # property applied after the mode build (audio_joiner
                # apply_follow_translation_cap), not a phantom-mode concern.
                delay = max(0, req_delay)
                # 原声淡化：延长/叠加前，对原声真实内容的末尾做 fade out，
                # 避免原声播完时突然切断（对延长出的空白静音无影响）。
                if orig_fade_ms > 0 and len(orig) > orig_fade_ms:
                    orig = orig.fade_out(orig_fade_ms)
                # 尾声淡化：TTS 播完后声音慢慢变弱（叠加前对 TTS 末尾 fade）。
                tts_real = len(tts)
                if tts_fade_ms > 0 and tts_real > tts_fade_ms:
                    tts = tts.fade_out(tts_fade_ms)
                # 虚拟淡化：尾声淡化时间不计入空白——淡化尾巴仍在段内，
                # 但标记 overlap_ms，让 joiner 在淡化开始处直接接下一段（重叠）。
                need = delay + tts_real
                if len(orig) < need:
                    orig = orig + AudioSegment.silent(
                        duration=need - len(orig), frame_rate=orig.frame_rate).set_channels(orig.channels)
                seg["overlap_ms"] = tts_fade_ms if (
                    virtual_fade and tts_fade_ms > 0) else 0
            else:
                # follow_original: fit the VOICE within the original at up to
                # `follow_original_rate` (speed up, truncate overflow). When the
                # voice already fits, clamp the delay so the voice still ends
                # within the original; the reverb tail rings out past the segment
                # boundary. When the handler chain pre-processed the TTS, tts_file
                # already carries the DSP tail — measure the voice from the
                # recorded pre-DSP source instead.
                voice_src = seg.get("phantom_voice_src") or tts_file
                voice_len = len(AudioSegment.from_file(voice_src))
                orig_ms = len(orig)
                if voice_len > orig_ms:
                    tts = fit_translation_to_original(tts, voice_len, orig_ms,
                                                      follow_original_rate)
                    delay = 0
                else:
                    delay = max(0, min(req_delay, max(0, orig_ms - voice_len - fade_out_ms)))
            mixed = orig.overlay(tts, position=delay)
            mixed.export(combined_path, format="wav",
                         parameters=wav_params or ["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2"])
            print(f"       [phantom] seg {seg['index']}: delay={delay}ms lp={lowpass_hz}Hz "
                  f"rv={reverb_mix} -> {len(mixed)}ms stereo")
        except Exception as e:
            print(f"       [phantom error] seg {seg['index']}: {e}")
            seg["vc_file"] = seg.get("tts_file")
            return seg

    seg["vc_file"] = combined_path
    return seg


def build_phantom_segments(segments, output_dir, delay_ms=400, reverb_mix=0.4,
                           lowpass_hz=3800, fade_in_ms=120, fade_out_ms=200,
                           sync_mode="follow_original", follow_original_rate=1.15,
                           max_workers=4, orig_fade_ms=0, tts_fade_ms=0,
                           virtual_fade=False, wav_params=None):
    """Build [stereo original + delayed phantom TTS] segments for phantom mode.

    sync_mode="follow_translation": the original is padded with trailing silence so
    the phantom translation plays to completion (no truncation at the segment end);
    the follow_translation length cap lives in audio_joiner.apply_follow_translation_cap
    and is applied by the pipeline after the mode build — this module never truncates.
    sync_mode="follow_original" (default): the voice is cap-accelerated to fit within
    the original segment length (up to `follow_original_rate`×; overflow truncated).

    follow_translation extras (ignored in follow_original):
      orig_fade_ms  原声淡化 — original fades out at its real content end.
      tts_fade_ms   尾声淡化 — TTS fades out after finishing.
      virtual_fade  虚拟淡化 — the fade tail is not counted into the padded
                    blank; segments are marked with ``overlap_ms`` so the
                    joiner starts the next segment early (overlapping tail).

    Runs after loudness alignment. Mutates `seg["vc_file"]` to point at each
    combined stereo WAV. The IR is generated once per output_dir and reused.
    """
    os.makedirs(output_dir, exist_ok=True)
    ir_path = get_ir_path(output_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(
            lambda seg: _build_one(seg, output_dir, delay_ms, reverb_mix, lowpass_hz,
                                   fade_in_ms, fade_out_ms, sync_mode,
                                   follow_original_rate, ir_path,
                                   orig_fade_ms=orig_fade_ms,
                                   tts_fade_ms=tts_fade_ms,
                                   virtual_fade=virtual_fade, wav_params=wav_params),
            segments,
        ))
    return segments
