"""Tests for modules.phantom_engine (Phantom Whisper build mode).

Run from the project root:  python -m unittest tests.test_phantom_engine
Zero new dependencies — synthetic WAVs are written with the stdlib `wave` module;
the DSP chain goes through the bundled ffmpeg_bin (same as the pipeline).
"""

import math
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pydub import AudioSegment

from modules.phantom_engine import (
    apply_phantom_dsp,
    build_phantom_segments,
    get_ir_path,
)

RATE = 24000


def _mono_samples(amp, dur_ms, freq=1000.0):
    n = int(RATE * dur_ms / 1000)
    return [int(amp * 32767 * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)]


def _write_wav(path, samples, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        if channels == 1:
            w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
        else:
            fr = b""
            for i in range(0, len(samples), 2):
                fr += struct.pack("<hh", samples[i], samples[i + 1])
            w.writeframes(fr)


def _read_seg(path):
    return AudioSegment.from_file(str(path))


def _rms_db(audio):
    samples = audio.set_channels(1).get_array_of_samples()
    if len(samples) == 0:
        return None
    rms = math.sqrt(sum(v * v for v in samples) / len(samples))
    if rms <= 0:
        return None
    return 20.0 * math.log10(rms / 32768.0)


class IrTest(unittest.TestCase):
    def test_deterministic_and_reused(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = get_ir_path(d)
            first = open(p1, "rb").read()
            self.assertGreater(len(first), 100)
            mtime1 = os.path.getmtime(p1)

            # second call must reuse (same path, no rewrite)
            p2 = get_ir_path(d)
            self.assertEqual(p1, p2)
            self.assertEqual(os.path.getmtime(p1), mtime1)

            # deterministic across a fresh output dir
            with tempfile.TemporaryDirectory() as d2:
                p3 = get_ir_path(d2)
                self.assertEqual(first, open(p3, "rb").read())

    def test_ir_is_not_silence(self):
        with tempfile.TemporaryDirectory() as d:
            ir = get_ir_path(d)
            seg = _read_seg(ir)
            self.assertEqual(seg.channels, 1)
            self.assertIsNotNone(_rms_db(seg))


class DspTest(unittest.TestCase):
    def _dsp(self, d, freq, reverb_mix=0.4):
        tts = os.path.join(d, "tts.wav")
        _write_wav(tts, _mono_samples(0.5, 400, freq))
        ir = get_ir_path(d)
        out = os.path.join(d, "out.wav")
        apply_phantom_dsp(tts, out, lowpass_hz=3800, reverb_mix=reverb_mix, ir_path=ir)
        return _read_seg(out)

    def test_lowpass_removes_high_sibilance(self):
        # 8 kHz is far above the 3800 Hz cutoff → heavily attenuated
        with tempfile.TemporaryDirectory() as d:
            hi = self._dsp(d, 8000.0)
            hi_db = _rms_db(hi)
            with tempfile.TemporaryDirectory() as d2:
                lo = self._dsp(d2, 1000.0)
                lo_db = _rms_db(lo)
            self.assertIsNotNone(hi_db)
            self.assertIsNotNone(lo_db)
            # low-freq voice keeps far more energy than the 8k sibilance
            self.assertGreater(lo_db, hi_db + 15.0)

    def test_reverb_output_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._dsp(d, 1000.0)
            self.assertIsNotNone(_rms_db(out))

    def test_reverb_tail_rings_out_past_voice(self):
        # Regression: pydub overlay truncates the OVERLAID segment to the base length,
        # so dry.overlay(wet) discarded the wet tail. The wet must be the overlay base
        # so the 400-1200 ms reverb ring-out survives into the delivered output.
        with tempfile.TemporaryDirectory() as d:
            tts = os.path.join(d, "tts.wav")
            _write_wav(tts, _mono_samples(0.5, 400, 1000.0))
            ir = get_ir_path(d)
            out = os.path.join(d, "out.wav")
            apply_phantom_dsp(tts, out, lowpass_hz=3800, reverb_mix=0.9, ir_path=ir)
            seg = _read_seg(out)
            # voice is 400 ms; output must extend well past it with a non-silent tail
            self.assertGreater(len(seg), 400 + 400)
            self.assertIsNotNone(_rms_db(seg[500:900]))

    def test_zero_reverb_mix_is_just_dry(self):
        with tempfile.TemporaryDirectory() as d:
            out = self._dsp(d, 1000.0, reverb_mix=0.0)
            self.assertIsNotNone(_rms_db(out))

    def test_reverb_mix_changes_output(self):
        # Regression: wet-only afir was silent in the bundled ffmpeg build, so
        # reverb_mix had no audible effect. The wet tail must actually be mixed in.
        with tempfile.TemporaryDirectory() as d:
            tts = os.path.join(d, "tts.wav")
            _write_wav(tts, _mono_samples(0.5, 400, 1000.0))
            ir = get_ir_path(d)
            out0 = os.path.join(d, "out0.wav")
            out9 = os.path.join(d, "out9.wav")
            apply_phantom_dsp(tts, out0, lowpass_hz=3800, reverb_mix=0.0, ir_path=ir)
            apply_phantom_dsp(tts, out9, lowpass_hz=3800, reverb_mix=0.9, ir_path=ir)
            dry_only = _read_seg(out0)
            reverb = _read_seg(out9)
            self.assertIsNotNone(_rms_db(dry_only))
            self.assertNotEqual(dry_only.get_array_of_samples().tolist(),
                                reverb.get_array_of_samples().tolist())


class BuildPhantomTest(unittest.TestCase):
    def _make_seg(self, d, idx, orig_amp_l=0.3, orig_amp_r=0.15, tts_amp=0.5,
                  orig_dur_ms=2500, tts_dur_ms=800, delay_ms=400):
        orig_samples = []
        for i in range(int(RATE * orig_dur_ms / 1000)):
            l = int(orig_amp_l * 32767 * math.sin(2 * math.pi * 440 * i / RATE))
            r = int(orig_amp_r * 32767 * math.sin(2 * math.pi * 440 * i / RATE))
            orig_samples.append(l)
            orig_samples.append(r)
        orig_path = os.path.join(d, f"orig_{idx}.wav")
        tts_path = os.path.join(d, f"tts_{idx}.wav")
        _write_wav(orig_path, orig_samples, channels=2)
        _write_wav(tts_path, _mono_samples(tts_amp, tts_dur_ms, 880.0))
        return {
            "index": idx,
            "segment_file": orig_path,
            "tts_file": tts_path,
            "start_ms": 0,
            "original_start_ms": 0,
            "original_end_ms": orig_dur_ms,
        }

    def _window_rms(self, audio, start_ms, win_ms=100):
        start = int(start_ms / 1000 * RATE) * audio.channels
        n = int(win_ms / 1000 * RATE) * audio.channels
        samples = audio.get_array_of_samples()
        chunk = samples[start:start + n]
        if not chunk:
            return None
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk))
        return 20.0 * math.log10(rms / 32768.0)

    def test_output_is_stereo_and_preserves_binaural(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 0)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                                   max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(out.channels, 2)
            # original stereo is not downmixed to mono: L ≠ R before the phantom
            # voice lands (first 300 ms — delay is 400 ms, TTS is centered)
            samples = out.get_array_of_samples()
            n = RATE // 10 * 3  # 300 ms per channel
            L = samples[0:2 * n:2]
            R = samples[1:2 * n:2]
            lrms = math.sqrt(sum(v * v for v in L) / n)
            rrms = math.sqrt(sum(v * v for v in R) / n)
            self.assertGreater(abs(lrms - rrms) / max(lrms, 1), 0.2)

    def test_voice_appears_after_delay(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 1, tts_dur_ms=800, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   max_workers=1)
            out = _read_seg(seg["vc_file"])
            pre = self._window_rms(out, 0)          # before delay — original only
            during = self._window_rms(out, 400)      # after delay — original + TTS
            self.assertIsNotNone(pre)
            self.assertIsNotNone(during)
            self.assertGreater(during, pre + 3.0)

    def test_delay_clamped_to_segment_length(self):
        # TTS is longer than original → delay must collapse to 0, not go negative
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 2, tts_dur_ms=1500, orig_dur_ms=1200, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(len(out), 1200)

    def test_follow_original_fits_voice_when_over_cap(self):
        # Voice (1500ms) > original (1200ms): need 1.25 ≤ cap 1.25 → the
        # voice is cap-accelerated to fit and the segment stays original-length;
        # delay collapses to 0, so the voice is audible right from the start.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 20, tts_dur_ms=1500, orig_dur_ms=1200, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_original",
                                   follow_original_rate=1.25, max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(len(out), 1200)
            orig_only = _read_seg(seg["segment_file"])
            self.assertGreater(self._window_rms(out, 200),
                               self._window_rms(orig_only, 200) + 3.0)

    def test_follow_original_truncates_when_cap_hit(self):
        # Voice (1500ms) > original (1200ms) with cap 1.0× (no speed-up allowed) →
        # overflow is truncated; delay collapses to 0.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 21, tts_dur_ms=1500, orig_dur_ms=1200, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_original",
                                   follow_original_rate=1.0, max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(len(out), 1200)
            orig_only = _read_seg(seg["segment_file"])
            self.assertGreater(self._window_rms(out, 200),
                               self._window_rms(orig_only, 200) + 3.0)

    def test_follow_translation_pads_original_when_tts_longer(self):
        # follow_translation extends the original so the phantom translation
        # (voice + tail) is never cut at the segment boundary.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 4, tts_dur_ms=1500, orig_dur_ms=1200, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertGreater(len(out), 1200)
            self.assertGreaterEqual(len(out), 400 + 1500)

    def test_follow_translation_no_pad_when_translation_fits(self):
        # original is long enough for delay + full phantom TTS → length unchanged
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 5, tts_dur_ms=800, orig_dur_ms=2500, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(len(out), 2500)

    def test_follow_translation_pads_for_voice_and_tail(self):
        # The reverb tail is part of the phantom translation, so padding must
        # cover delay + full DSP output (voice + tail), not just the voice.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 6, tts_dur_ms=800, orig_dur_ms=1000, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            # 400 (delay) + 1600 (800 voice + 800 reverb tail) = 2000, padded from 1000
            self.assertEqual(len(out), 2000)

    def test_follow_translation_honors_full_delay(self):
        # The requested delay is used (not collapsed to 0) so the voice lands
        # at delay_ms, and the original is padded to fit.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 7, tts_dur_ms=800, orig_dur_ms=1000, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            # 400 (delay) + 800 (voice) = 1200, padded from 1000
            self.assertEqual(len(out), 1200)
            pre = self._window_rms(out, 0)      # before delay — original only
            during = self._window_rms(out, 400)  # at delay — original + TTS voice
            self.assertIsNotNone(pre)
            self.assertIsNotNone(during)
            self.assertGreater(during, pre + 3.0)

    def test_follow_translation_keeps_tail_over_2x(self):
        # Phantom no longer truncates: follow_translation pads the original so the
        # translation (voice + reverb tail) plays to completion. The length cap is
        # a sound-sync property applied later by audio_joiner.apply_follow_translation_cap.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 8, tts_dur_ms=1500, orig_dur_ms=800, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            # old behavior truncated to 400+2×800=2000; the full DSP tail must survive
            self.assertGreater(len(out), 2000)
            self.assertGreaterEqual(len(out), 400 + 1500)

    def test_follow_translation_keeps_voice_over_2x(self):
        # Phantom no longer truncates: with reverb off the full voice plays to
        # completion regardless of how it compares to the original segment.
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 9, tts_dur_ms=1500, orig_dur_ms=500, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1)
            out = _read_seg(seg["vc_file"])
            # 400 (delay) + 1500 (full voice) = 1900 — no 2× orig (1000) cap applied
            self.assertEqual(len(out), 1900)

    def test_follow_translation_orig_fade_applies(self):
        # 原声淡化：原声真实内容末尾 fade out（TTS 较短不触发 pad 时仍生效）
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 10, tts_dur_ms=800, orig_dur_ms=2500, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1,
                                   orig_fade_ms=1000)
            out = _read_seg(seg["vc_file"])
            # 原声内容末尾（2100-2500ms）能量应明显低于中段（1500-1900ms）
            tail = self._window_rms(out, 2200)
            mid = self._window_rms(out, 1500)
            self.assertIsNotNone(tail)
            self.assertIsNotNone(mid)
            self.assertLess(tail, mid - 6.0)

    def test_follow_translation_tts_fade_sets_overlap_on_virtual(self):
        # 尾声淡化：virtual_fade=False 时 overlap_ms=0；开启后 overlap=淡化时长，
        # 且段长保持 delay + 完整 TTS（淡化尾巴仍在段内，由 joiner 重叠）。
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 11, tts_dur_ms=800, orig_dur_ms=1000, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1,
                                   tts_fade_ms=200, virtual_fade=False)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(seg.get("overlap_ms", 0), 0)
            self.assertEqual(len(out), 1200)

        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 12, tts_dur_ms=800, orig_dur_ms=1000, delay_ms=400)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   sync_mode="follow_translation", max_workers=1,
                                   tts_fade_ms=200, virtual_fade=True)
            out = _read_seg(seg["vc_file"])
            self.assertEqual(seg.get("overlap_ms", 0), 200)
            self.assertEqual(len(out), 1200)  # 淡化尾巴仍在段内

    def test_cache_invalidated_when_tts_changes(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 3, tts_amp=0.5)
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   max_workers=1)
            out1 = seg["vc_file"]
            m1 = os.path.getmtime(out1)
            db1 = _rms_db(_read_seg(out1))

            # same pipeline again → cache hit, no rewrite
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   max_workers=1)
            self.assertEqual(os.path.getmtime(out1), m1)

            # louder TTS at the same path → must rebuild
            _write_wav(seg["tts_file"], _mono_samples(0.9, 800, 880.0))
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.0, lowpass_hz=3800,
                                   max_workers=1)
            self.assertEqual(seg["vc_file"], out1)  # same cache key
            db2 = _rms_db(_read_seg(out1))
            self.assertGreater(db2, db1 + 3.0)

    def test_missing_files_noop(self):
        with tempfile.TemporaryDirectory() as d:
            seg = {"index": 9,
                   "segment_file": os.path.join(d, "nope.wav"),
                   "tts_file": os.path.join(d, "nope2.wav")}
            build_phantom_segments([seg], os.path.join(d, "phantom"),
                                   delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                                   max_workers=1)
            self.assertIsNone(seg.get("vc_file"))

    def test_skip_tts_keeps_original_without_tts_file(self):
        with tempfile.TemporaryDirectory() as d:
            seg = self._make_seg(d, 13)
            os.unlink(seg["tts_file"])
            seg["skip_tts"] = True

            build_phantom_segments([seg], os.path.join(d, "phantom"), max_workers=1)

            self.assertEqual(seg["vc_file"], seg["segment_file"])

    def test_pre_dsp_skips_chain_dsp_and_honors_voice_len(self):
        # seg carries phantom_dsp_file (already processed by the handler chain)
        # + the raw voice path. _build_one must NOT re-run apply_phantom_dsp,
        # and the follow_original delay clamp must measure the raw voice (800 ms),
        # not the DSP'd file with its reverb tail (1600 ms) — otherwise the
        # 400 ms delay would collapse to 0.
        with tempfile.TemporaryDirectory() as d:
            orig_path = os.path.join(d, "orig.wav")
            orig_samples = []
            for i in range(int(RATE * 1200 / 1000)):
                l = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / RATE))
                r = int(0.15 * 32767 * math.sin(2 * math.pi * 440 * i / RATE))
                orig_samples.append(l)
                orig_samples.append(r)
            _write_wav(orig_path, orig_samples, channels=2)
            raw_path = os.path.join(d, "raw.wav")     # pre-DSP voice, 800 ms
            _write_wav(raw_path, _mono_samples(0.5, 800, 880.0))
            dsp_path = os.path.join(d, "dsp.wav")     # voice + reverb tail, 1600 ms
            _write_wav(dsp_path, _mono_samples(0.5, 1600, 880.0))
            seg = {"index": 10,
                   "segment_file": orig_path,
                   "tts_file": dsp_path,
                   "phantom_dsp_file": dsp_path,
                   "phantom_voice_src": raw_path,
                   "start_ms": 0,
                   "original_start_ms": 0,
                   "original_end_ms": 1200}
            with mock.patch("modules.phantom_engine.apply_phantom_dsp") as m_dsp:
                build_phantom_segments([seg], os.path.join(d, "phantom"),
                                       delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                                       sync_mode="follow_original", max_workers=1)
            m_dsp.assert_not_called()
            out = _read_seg(seg["vc_file"])
            # delay = max(0, min(400, 1200 - 800 - 200)) = 200 ms
            pre = self._window_rms(out, 0)      # before the voice lands
            during = self._window_rms(out, 250)  # orig + phantom voice
            self.assertIsNotNone(pre)
            self.assertIsNotNone(during)
            self.assertGreater(during, pre + 3.0)


if __name__ == "__main__":
    unittest.main()
