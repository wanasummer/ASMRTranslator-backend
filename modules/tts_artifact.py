# -*- coding: utf-8 -*-
"""TTS artifact detector — pure-numpy port of 测试1/analysis/detect_tts_artifact.py.

Detects "male-like / mechanical" segments in an edge-tts file. Root cause the
original validated on tts_0004.mp3 clause 2: the synthesized voice loses its
fundamental — E(f0)/E(2*f0) collapses (fundamental is 1/10~1/100 of the 2nd
harmonic), the YIN pitch tracker locks an octave too low (female ~300 Hz ->
reported ~150 Hz), and the missing low energy + noisy tail is what a listener
hears as '男声混机械音'.

All metrics are ratios/percentages (E(f0)/E(2f0), spectral flatness median), so
they are gain-invariant: the verdict is identical before and after loudness
alignment. No scipy/librosa dependency — the YIN pitch tracker, STFT, and
spectral flatness are re-implemented in pure numpy, replicating the exact
librosa.yin / librosa.stft / librosa.feature.spectral_flatness behavior the
original script relied on.

Thresholds (same as source):
  dark : weakFund% > 50 and n >= 8   (缺基频/偏男声)
  mech : flatMed > 0.05 and dur > 0.5s  (噪声/机械感)
"""

import sys

import numpy as np

SR = 24000


def _load_mono(path, sr):
    """Decode to mono, resample to sr, return float64 samples normalized to [-1,1]."""
    from pydub import AudioSegment
    seg = (AudioSegment.from_file(str(path))
           .set_channels(1).set_frame_rate(sr).set_sample_width(2))
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16).astype(np.float64)
    return samples / 32768.0


def _speech_regions(y, sr, thresh_db=-35, min_gap_s=0.3):
    """Windows of 50 ms with RMS > thresh_db; gaps < min_gap_s are merged."""
    hop = int(0.05 * sr)
    loud = []
    for i in range(0, len(y) - hop, hop):
        rdb = 20 * np.log10(np.sqrt(np.mean(y[i:i + hop] ** 2)) + 1e-9)
        loud.append((i / sr, rdb > thresh_db))
    regions = []
    start = None
    for t, on in loud:
        if on and start is None:
            start = t
        elif not on and start is not None:
            if t - start > min_gap_s:
                regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, loud[-1][0]))
    return regions


def _energy_at(seg, sr, f, bw=40):
    """Sum of the magnitude spectrum of `seg` within ±bw Hz of f."""
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freq = np.fft.rfftfreq(len(seg), 1 / sr)
    return S[np.abs(freq - f) <= bw].sum() + 1e-12


# ---- YIN pitch tracker (faithful numpy port of librosa.yin) ------------------

def _next_pow2(n):
    return 1 << ((n - 1).bit_length())


def _cumulative_mean_normalized_difference(y_frames, min_period, max_period):
    """librosa.core.pitch._cumulative_mean_normalized_difference."""
    n_fft = _next_pow2(2 * y_frames.shape[0] - 1)
    acf = np.fft.irfft(np.abs(np.fft.rfft(y_frames, n=n_fft, axis=0)) ** 2,
                       n=n_fft, axis=0)[:max_period + 1]

    cum = np.cumsum(y_frames * y_frames, axis=0)
    d = np.empty((max_period + 1, y_frames.shape[1]))
    d[0] = 0.0
    d[1:] = 2 * (acf[0:1] - acf[1:]) - cum[:max_period]

    k_range = np.arange(1, max_period + 1)
    running_mean = np.cumsum(d[1:], axis=0) / k_range[:, None]
    yin = d[min_period:max_period + 1] / (running_mean[min_period - 1:max_period]
                                          + np.finfo(np.float64).tiny)
    return yin


def _parabolic_shifts(x):
    """librosa.core.pitch._parabolic_interpolation along axis 0 (bin axis)."""
    xp = np.zeros_like(x)
    xp[1:] = x[:-1]
    xn = np.zeros_like(x)
    xn[:-1] = x[1:]
    a = xp + xn - 2 * x
    b = (xn - xp) / 2
    shift = np.where(np.abs(b) >= np.abs(a), 0.0, -b / a)
    shift[0] = 0.0
    shift[-1] = 0.0
    return shift


def _localmin(x):
    """librosa.util.localmin along axis 0: strict left, non-strict right."""
    lmin = np.zeros(x.shape, dtype=bool)
    lmin[1:-1] = (x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])
    lmin[-1] = x[-1] < x[-2]
    return lmin


def _yin_frames(y, sr, fmin=60, fmax=500, frame_length=2048,
                hop_length=None, trough_threshold=0.1):
    """YIN f0 curve, same frame grid and fallback semantics as librosa.yin."""
    if hop_length is None:
        hop_length = frame_length // 4
    if len(y) < frame_length:
        return np.array([])

    y = np.pad(y, (frame_length // 2, frame_length // 2), mode="constant")
    n_frames = (len(y) - frame_length) // hop_length + 1
    idx = np.arange(frame_length)[:, None] + hop_length * np.arange(n_frames)[None, :]
    y_frames = y[idx]

    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - 1)

    yin = _cumulative_mean_normalized_difference(y_frames, min_period, max_period)

    is_trough = _localmin(yin)
    is_trough[0] = yin[0] < yin[1]
    is_threshold_trough = is_trough & (yin < trough_threshold)

    global_min = np.argmin(yin, axis=0)
    yin_period = np.argmax(is_threshold_trough, axis=0)
    yin_period[~np.any(is_threshold_trough, axis=0)] = global_min[
        ~np.any(is_threshold_trough, axis=0)]

    shifts = _parabolic_shifts(yin)
    yin_period = min_period + yin_period + shifts[yin_period, np.arange(n_frames)]
    return sr / yin_period


# ---- STFT spectral flatness (librosa.stft + feature.spectral_flatness) -------

_PERIODIC_HANN = None


def _stft_first_frame_mag(seg, n_fft=1024, hop_length=512):
    """Magnitude of STFT frame 0 of `seg` (librosa.stft, center=True, hann)."""
    global _PERIODIC_HANN
    if _PERIODIC_HANN is None or len(_PERIODIC_HANN) != n_fft:
        _PERIODIC_HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    pad = n_fft // 2
    frame0 = np.zeros(n_fft)
    frame0[pad:pad + len(seg[:n_fft - pad])] = seg[:n_fft - pad]
    return np.abs(np.fft.rfft(frame0 * _PERIODIC_HANN))


def _spectral_flatness(S, amin=1e-10, power=2.0):
    """librosa.feature.spectral_flatness on a 1-D magnitude spectrum."""
    S_thresh = np.maximum(amin, S ** power)
    return float(np.exp(np.mean(np.log(S_thresh))) / np.mean(S_thresh))


# ---- per-region analysis ----------------------------------------------------

def _analyze_region(y, sr, a, b, hop_s=0.05, win_s=0.1):
    hop = int(hop_s * sr)
    win = int(win_s * sr)
    seg = y[int(a * sr):int(b * sr)]
    f0 = _yin_frames(seg, sr, hop_length=hop)
    n = weak = sub = 0
    f0v = []
    flats = []
    for i in range(0, len(seg) - win, hop):
        sl = seg[i:i + win]
        if np.sqrt(np.mean(sl ** 2)) < 2e-4:
            continue
        flats.append(_spectral_flatness(_stft_first_frame_mag(sl)))
        j = i // hop
        f = f0[j] if j < len(f0) else np.nan
        if np.isnan(f) or not (60 < f < 500):
            continue
        n += 1
        f0v.append(f)
        r = _energy_at(sl, sr, f) / _energy_at(sl, sr, 2 * f)
        if r < 0.6:
            weak += 1
        if r < 1.3:
            sub += 1
    if n == 0:
        return None
    return dict(n=n, f0med=float(np.median(f0v)), weak=100 * weak / n,
                sublock=100 * sub / n, flat=float(np.median(flats)))


# ---- public entry point -----------------------------------------------------

def detect_tts_artifact(path, sr=SR, thresh_db=-35, min_gap_s=0.3):
    """Detect 'male-like/mechanical' segments in a TTS file.

    Returns a dict: {flagged, duration_s, regions, flags, error}
      flagged : True if any speech region is dark (缺基频) or mech (机械感)
      regions : list of {start_s, end_s, n, f0med, weak, sublock, flat, flags}
      flags   : set of all triggered flag names across regions
      error   : set when the file cannot be decoded (never flagged)
    """
    try:
        y = _load_mono(path, sr)
    except Exception as e:  # noqa: BLE001 — decode failures are a per-file report
        return dict(flagged=False, duration_s=None, regions=[],
                    flags=set(), error=str(e))

    if len(y) < int(0.05 * sr):
        return dict(flagged=False, duration_s=len(y) / sr, regions=[],
                    flags=set(), error=None)

    regions = []
    for a, b in _speech_regions(y, sr, thresh_db, min_gap_s):
        r = _analyze_region(y, sr, a, b)
        if r is None:
            continue
        dark = r["weak"] > 50 and r["n"] >= 8
        mech = r["flat"] > 0.05 and (b - a) > 0.5
        flags = []
        if dark:
            flags.append("dark")
        if mech:
            flags.append("mech")
        regions.append({"start_s": a, "end_s": b, **r, "flags": flags})

    flagged = any(r["flags"] for r in regions)
    all_flags = set(f for r in regions for f in r["flags"])
    return dict(flagged=flagged, duration_s=len(y) / sr, regions=regions,
                flags=all_flags, error=None)


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("path", help="path to TTS file (mp3/wav)")
    ap.add_argument("--sr", type=int, default=SR)
    args = ap.parse_args()

    out = detect_tts_artifact(args.path, sr=args.sr)
    print(f"file: {args.path}   dur={out['duration_s']:.2f}s" if out["duration_s"] else
          f"file: {args.path}   decode-failed: {out.get('error')}")
    print(f"{'region[s]':>12} {'F0med':>6} {'weakFund%':>9} {'subLock%':>8} "
          f"{'flatMed':>8}  verdict")
    for r in out["regions"]:
        v = []
        if "dark" in r["flags"]:
            v.append("<<< 变暗/偏男声 (缺基频)")
        if "mech" in r["flags"]:
            v.append("<<< 噪声/机械感")
        line = (f"{r['start_s']:7.2f}-{r['end_s']:6.2f} {round(r['f0med']):6d} "
                f"{r['weak']:8.0f}% {r['sublock']:7.0f}% {r['flat']:8.4f}  "
                f"{' '.join(v)}")
        sys.stdout.buffer.write((line + "\n").encode("utf-8", "replace"))
    print(f"\nflagged regions: {len([r for r in out['regions'] if r['flags']])}  "
          f"(weakFund>50% => 缺基频/变暗; flat>0.05且时长>0.5s => 机械感)")
    print("判定: 有劣化 —— 建议对该段重新生成" if out["flagged"] else "判定: 干净 —— 无需重生成")


if __name__ == "__main__":
    main()
