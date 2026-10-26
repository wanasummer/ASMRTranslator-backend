"""Tests for modules.audio_joiner (follow-original cap-accelerate + follow-translation cap).

Run from the project root:  python -m unittest tests.test_audio_joiner
Offline — synthetic WAVs are written with the stdlib `wave` module; the atempo
stretch goes through the bundled ffmpeg_bin (same as the pipeline).
"""

import json
import math
import json
import os
import struct
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pydub import AudioSegment

from modules.audio_joiner import (
    apply_follow_translation_cap,
    build_dual_segments,
    build_sync_segments,
    enforce_tts_budget,
    fit_translation_to_original,
    join_audio,
)

SR = 24000


def _sine(dur_ms, freq=300.0, amp=0.5):
    n = int(SR * dur_ms / 1000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        p = f.name
    try:
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"".join(
                struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / SR)))
                for i in range(n)))
        return AudioSegment.from_file(p)
    finally:
        os.unlink(p)


def _write_wav(path, dur_ms, freq=300.0, amp=0.5):
    n = int(SR * dur_ms / 1000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / SR)))
            for i in range(n)))


def _cap_seg(d, idx, orig_ms, vc_ms):
    orig = os.path.join(d, f"orig{idx}.wav")
    vc = os.path.join(d, f"vc{idx}.wav")
    _write_wav(orig, orig_ms)
    _write_wav(vc, vc_ms)
    return {"index": idx, "segment_file": orig, "vc_file": vc}


class FitTest(unittest.TestCase):
    def test_shorter_is_returned_unchanged(self):
        a = _sine(800)
        out = fit_translation_to_original(a, 800, 1000, 1.5)
        self.assertIs(out, a)
        self.assertEqual(len(out), 800)

    def test_cap_sufficient_stretches_to_exact(self):
        # need 1500/1000 = 1.5 ≤ 1.5 → stretch to exactly orig_ms (atempo rounds ±ms)
        a = _sine(1500)
        out = fit_translation_to_original(a, 1500, 1000, 1.5)
        self.assertLess(abs(len(out) - 1000), 20)

    def test_cap_insufficient_stretches_then_truncates(self):
        # need 2000/1000 = 2.0 > 1.5 → stretch at 1.5× then truncate to orig_ms
        a = _sine(2000)
        out = fit_translation_to_original(a, 2000, 1000, 1.5)
        self.assertEqual(len(out), 1000)

    def test_rate_one_is_pure_truncation(self):
        a = _sine(2000)
        out = fit_translation_to_original(a, 2000, 1000, 1.0)
        self.assertEqual(len(out), 1000)

    def test_phantom_dsp_audio_fits_on_voice_length(self):
        # audio carries a reverb tail (2300ms) but content is the 1500ms voice:
        # need = 1500/1000 = 1.5 ≤ 1.5 → voice exactly fills orig, tail truncated.
        audio = _sine(2300)
        out = fit_translation_to_original(audio, 1500, 1000, 1.5)
        self.assertEqual(len(out), 1000)


class SkipTtsTest(unittest.TestCase):
    def test_mode_builders_keep_the_original_without_a_tts_file(self):
        with tempfile.TemporaryDirectory() as d:
            original = os.path.join(d, "original.wav")
            _write_wav(original, 400)
            builders = (
                lambda seg: build_dual_segments([seg], os.path.join(d, "dual"), max_workers=1),
                lambda seg: build_sync_segments([seg], os.path.join(d, "sync"), max_workers=1),
            )
            for build in builders:
                with self.subTest(builder=build):
                    seg = {"index": 3, "segment_file": original, "skip_tts": True}
                    build(seg)
                    self.assertEqual(seg["vc_file"], original)

    def test_follow_translation_cap_keeps_skipped_original(self):
        with tempfile.TemporaryDirectory() as d:
            original = os.path.join(d, "original.wav")
            _write_wav(original, 400)
            seg = {"index": 3, "segment_file": original, "skip_tts": True}

            apply_follow_translation_cap([seg])

            self.assertEqual(seg["vc_file"], original)

class CapTest(unittest.TestCase):
    """apply_follow_translation_cap — sound-sync length cap (after mode build).

    Policy: truncate only when the translation runs more than max_overhang_ms
    (default 3000 ms) past the original slice. Target = delay_ms + orig_ms +
    max_overhang_ms (delay 0 for plain sync). Within that window the translation
    plays in full.
    """

    def _run(self, d, seg, **kwargs):
        return apply_follow_translation_cap([seg], **kwargs)

    def test_truncates_over_3s_overhang(self):
        # 7000 − 1000 = 6000 overhang > 3000 → truncated to 1000 + 3000 = 4000.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 1, orig_ms=1000, vc_ms=7000)
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4000)

    def test_no_truncate_within_3s_overhang(self):
        # 3000 − 500 = 2500 overhang < 3000 → play in full.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 2, orig_ms=500, vc_ms=3000)
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 3000)

    def test_no_truncate_when_within_3s(self):
        # 8000 − 6000 = 2000 overhang < 3000 → play in full.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 3, orig_ms=6000, vc_ms=8000)
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 8000)

    def test_boundary_exact_3s_no_truncate(self):
        # trans == orig + 3000 exactly → "more than 3 s" is false → play in full.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 7, orig_ms=1000, vc_ms=4000)
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4000)

    def test_phantom_delay_target(self):
        # trans = 7400 − 400 = 7000; overhang vs orig 1000 → 6000 > 3000;
        # target = 400 + (1000 + 3000) = 4400.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 4, orig_ms=1000, vc_ms=7400)
            self._run(d, seg, delay_ms=400)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4400)

    def test_noop_when_combined_shorter_than_original(self):
        # translation side is not the long side → never capped.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 5, orig_ms=5000, vc_ms=4000)
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4000)

    def test_missing_files_fail_fast(self):
        with tempfile.TemporaryDirectory() as d:
            seg = {"index": 6, "segment_file": os.path.join(d, "no.wav"),
                   "vc_file": os.path.join(d, "no2.wav")}
            with self.assertRaises(FileNotFoundError):
                self._run(d, seg)

    def test_budget_window_with_per_segment_delay(self):
        # Pink model: window = budget (max_allowed_tts_ms), not orig_ms. Here
        # budget=1000, tts_delay_ms=400, orig=pink slice=1400, comb=5000:
        # trans = 5000-400 = 4600 > 1000+3000 → target = 400 + 4000 = 4400.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 8, orig_ms=1400, vc_ms=5000)
            seg["tts_delay_ms"] = 400
            seg["max_allowed_tts_ms"] = 1000
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4400)

    def test_per_segment_delay_used_when_delay_ms_none(self):
        # No budget field → falls back to orig_ms window; per-segment tts_delay_ms
        # drives the offset instead of the delay_ms param.
        with tempfile.TemporaryDirectory() as d:
            seg = _cap_seg(d, 9, orig_ms=1000, vc_ms=7400)
            seg["tts_delay_ms"] = 400
            self._run(d, seg)
            self.assertEqual(len(AudioSegment.from_file(seg["vc_file"])), 4400)


def _tts_seg(d, idx, tts_ms, budget, phantom_src_ms=None, skip=False):
    """Segment with an on-disk TTS wav (optionally a pre-DSP voice source)."""
    tts = os.path.join(d, f"tts{idx}.wav")
    _write_wav(tts, tts_ms)
    seg = {"index": idx, "tts_file": tts, "max_allowed_tts_ms": budget,
           "skip_tts": skip}
    if phantom_src_ms is not None:
        src = os.path.join(d, f"voice{idx}.wav")
        _write_wav(src, phantom_src_ms)
        seg["phantom_voice_src"] = src
    return seg


class BudgetTest(unittest.TestCase):
    """enforce_tts_budget — pink-segment TTS time budget (fallbacks 1 & 2)."""

    def _len(self, seg):
        return len(AudioSegment.from_file(seg["tts_file"]))

    def test_fits_within_budget_noop(self):
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 0, 1000, budget=2000)
            self.assertIs(enforce_tts_budget([seg])[0], seg)
            self.assertEqual(self._len(seg), 1000)
            self.assertTrue(seg["tts_file"].endswith("tts0.wav"))

    def test_speedup_fits_exactly_to_budget(self):
        # need 4000/2000 = 2.0 ≤ 2.0 → stretch to ~2000 (atempo rounds ±ms).
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 1, 4000, budget=2000)
            enforce_tts_budget([seg])
            self.assertLess(abs(self._len(seg) - 2000), 60)

    def test_speedup_capped_then_hard_cut(self):
        # need 6000/2000 = 3.0 > 2.0 → stretch at 2.0× then truncate to 2000.
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 2, 6000, budget=2000)
            enforce_tts_budget([seg])
            self.assertEqual(self._len(seg), 2000)
            self.assertEqual(json.loads((Path(d) / "tts_tempo_0002.json").read_text())["rate"], 2.0)

    def test_follow_translation_allows_pad_window(self):
        # mode=follow_translation → effective = 1000 + 3000 = 4000; 8000/4000 = 2.0
        # ≤ 2.0 → stretches to ~4000, no hard cut.
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 3, 8000, budget=1000)
            enforce_tts_budget([seg], mode="follow_translation")
            self.assertLess(abs(self._len(seg) - 4000), 60)

    def test_follow_translation_hard_cuts_past_window(self):
        # 12000 → effective 4000, need 3.0 > 2.0 → stretch 2.0× then cut to 4000.
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 4, 12000, budget=1000)
            enforce_tts_budget([seg], mode="follow_translation")
            self.assertEqual(self._len(seg), 4000)

    def test_phantom_measures_voice_not_tail(self):
        # voice 2000 vs budget 1000 → stretch the 5000ms DSP file 2.0× then cut
        # to 1000; the tail must not be double-counted.
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 5, 5000, budget=1000, phantom_src_ms=2000)
            enforce_tts_budget([seg])
            self.assertEqual(self._len(seg), 1000)

    def test_skip_tts_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            seg = _tts_seg(d, 6, 5000, budget=1000, skip=True)
            enforce_tts_budget([seg])
            self.assertEqual(self._len(seg), 5000)
            self.assertTrue(seg["tts_file"].endswith("tts6.wav"))


class JoinShortSegmentTest(unittest.TestCase):
    def _segments(self, directory, durations):
        segments = []
        for index, duration in enumerate(durations):
            path = os.path.join(directory, f"segment_{index}.wav")
            _write_wav(path, duration)
            segments.append({"index": index, "vc_file": path})
        return segments

    def test_short_segments_are_skipped_and_later_audio_is_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            segments = self._segments(directory, [0, 100, 20, 40, 49, 50, 200, 40])
            original_segments = [dict(segment) for segment in segments]
            output = os.path.join(directory, "output.wav")
            exported = []
            # Keep real decoding/concatenation, mock external encoding only.
            def capture(audio, path, **kwargs):
                exported.append((path, audio))
            with patch.object(AudioSegment, "export", autospec=True, side_effect=capture):
                self.assertEqual(join_audio(segments, output), output)
            self.assertEqual([path for path, _ in exported],
                             [output, os.path.join(directory, "output.mp3")])
            expected = (AudioSegment.from_file(segments[1]["vc_file"])
                        + AudioSegment.from_file(segments[5]["vc_file"])
                        + AudioSegment.from_file(segments[6]["vc_file"])).set_channels(2)
            self.assertEqual(exported[0][1].raw_data, expected.raw_data)
            self.assertEqual(len(exported[0][1]), 350)
            self.assertEqual(segments, original_segments)
            with open(os.path.join(directory, "output.join_audit.json"), encoding="utf-8") as file:
                audit = json.load(file)
            self.assertEqual(audit["joined_count"], 3)
            self.assertEqual([item["index"] for item in audit["skipped_segments"]],
                             [0, 2, 3, 4, 7])

    def test_all_short_segments_fail_without_exporting(self):
        with tempfile.TemporaryDirectory() as directory:
            segments = self._segments(directory, [0, 20, 40, 49])
            with patch.object(AudioSegment, "export") as export:
                with self.assertRaisesRegex(RuntimeError, "No valid audio segments"):
                    join_audio(segments, os.path.join(directory, "output.wav"))
            export.assert_not_called()

    def test_missing_and_undecodable_files_still_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "output.wav")
            missing = {"index": 92, "vc_file": os.path.join(directory, "missing.wav")}
            with self.assertRaises(FileNotFoundError):
                join_audio([missing], output)
            segments = self._segments(directory, [40])
            with patch.object(AudioSegment, "from_file", side_effect=ValueError("invalid audio")):
                with self.assertRaisesRegex(ValueError, "invalid audio"):
                    join_audio(segments, output)


class JoinOverlapTest(unittest.TestCase):
    """join_audio 的虚拟淡化重叠：overlap_ms 段提前开始，总长扣掉重叠部分。"""

    def _seg(self, d, idx, dur_ms, overlap_ms=0):
        vc = os.path.join(d, f"join{idx}.wav")
        _write_wav(vc, dur_ms)
        seg = {"index": idx, "vc_file": vc}
        if overlap_ms:
            seg["overlap_ms"] = overlap_ms
        return seg

    def test_overlap_shortens_total(self):
        from modules.audio_joiner import join_audio
        with tempfile.TemporaryDirectory() as d:
            segs = [self._seg(d, 0, 1000), self._seg(d, 1, 800, overlap_ms=200)]
            out = os.path.join(d, "joined.wav")
            join_audio(segs, out)
            joined = AudioSegment.from_file(out)
            # 1000 + 800 − 200（重叠）= 1600
            self.assertLess(abs(len(joined) - 1600), 30)

    def test_no_overlap_is_plain_concat(self):
        from modules.audio_joiner import join_audio
        with tempfile.TemporaryDirectory() as d:
            segs = [self._seg(d, 0, 1000), self._seg(d, 1, 800)]
            out = os.path.join(d, "joined.wav")
            join_audio(segs, out)
            joined = AudioSegment.from_file(out)
            self.assertLess(abs(len(joined) - 1800), 30)


class JoinTimelineTest(unittest.TestCase):
    """Use real WAV decoding/joining; mock only external output encoding."""

    def join(self, directory, durations, *, crossfade=0, overlap=0, source=False, speech_delay=0):
        segments = []
        for index, duration in enumerate(durations):
            path = os.path.join(directory, f"timeline_{index}.wav")
            _write_wav(path, duration)
            segments.append({"index": index, "segment_id": str(index + 10),
                             "start_ms": index * 1000,
                             "original_start_ms": index * 1000 + speech_delay,
                             "original_end_ms": (index + 1) * 1000 + speech_delay,
                             "vc_file": path, "tts_file": None if source else path,
                             "overlap_ms": overlap if index else 0})
        original = [dict(segment) for segment in segments]
        exported = []
        def capture(audio, path, **kwargs):
            exported.append(audio)
        with patch.object(AudioSegment, "export", autospec=True, side_effect=capture):
            result = join_audio(
                segments, os.path.join(directory, "output.wav"), crossfade,
                return_timeline=True)
        self.assertEqual(segments, original)
        self.assertIsInstance(result.placements, tuple)
        self.assertEqual(result.placements[-1].end_ms, len(exported[0]))
        return result

    def test_crossfade_placements_match_real_audio_length(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.join(d, [1000, 1000, 1000], crossfade=100)
            self.assertEqual([(p.start_ms, p.end_ms) for p in result.placements],
                             [(0, 1000), (900, 1900), (1800, 2800)])
            self.assertEqual([p.block_id for p in result.placements], ["10", "11", "12"])

    def test_virtual_overlap_and_short_segments_use_surviving_audio_only(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.join(d, [1000, 20, 800], overlap=200)
            self.assertEqual([(p.block_id, p.start_ms, p.end_ms) for p in result.placements],
                             [("10", 0, 1000), ("12", 800, 1600)])
            with open(os.path.join(d, "output.join_audit.json"), encoding="utf-8") as file:
                audit = json.load(file)
            self.assertEqual([p["block_id"] for p in audit["placements"]], ["10", "12"])
            self.assertEqual([p["index"] for p in audit["skipped_segments"]], [1])

    def test_crossfade_is_clamped_by_decoded_segment_lengths(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.join(d, [1000, 100], crossfade=500)
            self.assertEqual((result.placements[1].start_ms, result.placements[1].end_ms),
                             (950, 1050))

    def test_extended_source_segment_moves_next_cue_by_actual_extension(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.join(d, [3000, 1000], source=True)
            self.assertEqual([(p.subtitle_start_ms, p.subtitle_end_ms) for p in result.placements],
                             [(0, 1000), (3000, 4000)])

    def test_replace_discards_source_leading_silence_and_source_spans_are_clipped(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.join(d, [800], speech_delay=5000)
            self.assertEqual((result.placements[0].subtitle_start_ms,
                              result.placements[0].subtitle_end_ms), (0, 800))
            result = self.join(d, [800], source=True, speech_delay=500)
            self.assertEqual((result.placements[0].subtitle_start_ms,
                              result.placements[0].subtitle_end_ms), (500, 800))


if __name__ == "__main__":
    unittest.main()
