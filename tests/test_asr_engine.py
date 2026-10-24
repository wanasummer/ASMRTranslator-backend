"""Tests for modules.asr_engine (transcribe_slices orchestration).

Run from the project root:  python -m unittest tests.test_asr_engine
Fully offline — create_asr and extract_ranges are stubbed; no
ffmpeg, funasr/torch or network is touched. (VAD slicing is pipeline's job and
is covered by tests/test_pipeline_vad.py.)
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.asr import ASRError
from modules.error import PipelineError
from modules.asr.models import ASRSegment, ASRResult, AudioChunk
from modules.asr.no_speech import ASRNoSpeechError
from modules.asr_engine import transcribe_slices
from modules.vad import SpeechRegion
from modules.vtt_parser import VTTBlock


def _chunk(idx, start, end, path):
    return AudioChunk(id=f"chunk_{idx:04d}", start=start, end=end, path=path)


class _FakeAdapter:
    """Scripted ASR adapter: maps a path (or its basename) to an ASRResult."""

    def __init__(self, by_path=None, default=None, configured=True):
        self._by_path = dict(by_path or {})
        self._default = default or ASRResult(text="", duration=0.0, segments=[])
        self._configured = configured
        self.calls = []

    @property
    def name(self):
        return "fish-asr"

    @property
    def timeout(self):
        return 30.0

    def is_configured(self):
        return self._configured

    async def transcribe(self, audio_path, language=None, timestamps=True, client=None):
        self.calls.append(audio_path)
        for key, val in self._by_path.items():
            if audio_path == key or audio_path.endswith(os.sep + key) or audio_path.endswith("/" + key):
                return val
        return self._default


class _Flaky429Adapter:
    """Raises ``ASRError("...429...")`` the first ``fails`` calls, then succeeds."""

    def __init__(self, fails=2, configured=True):
        self._fails = fails
        self._configured = configured
        self.calls = 0

    @property
    def name(self):
        return "fish-asr"

    @property
    def timeout(self):
        return 30.0

    def is_configured(self):
        return self._configured

    async def transcribe(self, audio_path, language=None, timestamps=True, client=None):
        self.calls += 1
        if self.calls <= self._fails:
            raise ASRError("Fish Audio ASR returned 429: concurrent request limit")
        return ASRResult("こんにちは", 2.0, "ja", [ASRSegment("こんにちは", 0.0, 1.0)])


class _NoSpeechAdapter(_FakeAdapter):
    async def transcribe(self, audio_path, language=None, timestamps=True, client=None):
        self.calls.append(audio_path)
        if audio_path.endswith("silent.wav"):
            raise ASRNoSpeechError("ASR_RESPONSE_HAVE_NO_WORDS")
        return ASRResult("こんにちは", 2.0, "ja", [ASRSegment("こんにちは", 0.0, 1.0)])


class TranscribeSlicesTest(unittest.TestCase):
    def test_no_speech_chunk_is_empty_and_does_not_stop_other_chunks(self):
        adapter = _NoSpeechAdapter()
        blocks = self._run(
            [SpeechRegion(0.0, 2.0), SpeechRegion(3.0, 5.0)],
            [_chunk(0, 0.0, 2.0, "silent.wav"), _chunk(1, 3.0, 5.0, "speech.wav")],
            adapter,
            {"api_key": "k"},
        )

        self.assertEqual([block.text for block in blocks], ["こんにちは"])
        self.assertEqual(len(adapter.calls), 2)

    def test_qwen_workspace_id_reaches_the_slice_adapter(self):
        adapter = _FakeAdapter()
        with mock.patch("modules.asr_engine.create_asr", return_value=adapter) as factory, \
             tempfile.TemporaryDirectory() as directory:
            blocks = transcribe_slices(
                "audio.mp3",
                backend="qwen-audio-3.1-asr-flash",
                asr_kwargs={
                    "api_key": "key",
                    "workspace_id": "workspace-id",
                    "language": "ja",
                },
                output_dir=directory,
                slices=[],
            )

        self.assertEqual(blocks, [])
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["api_key"], "key")
        self.assertEqual(factory.call_args.kwargs["workspace_id"], "workspace-id")

    def test_decode_keeps_padding_but_output_uses_speech_without_beta(self):
        from modules.vad import optimize_asmr_segments
        slices = optimize_asmr_segments([SpeechRegion(10, 15)])
        adapter = _FakeAdapter(default=ASRResult("こんにちは", 5.8, "ja", [ASRSegment("こんにちは", .4, 5.4)]))
        with mock.patch("modules.asr_engine.create_asr", return_value=adapter), \
             mock.patch("modules.asr_engine.extract_ranges", return_value=[_chunk(0, 9.6, 15.4, 'padded.wav')]) as extract, \
             tempfile.TemporaryDirectory() as directory:
            blocks = transcribe_slices('audio.wav', slices=slices, output_dir=directory)
        self.assertEqual(extract.call_args.args[1], [(9.6, 15.4)])
        self.assertEqual([(b.start_ms, b.end_ms) for b in blocks], [(10000, 15000)])
        self.assertEqual((slices[0].start, slices[0].end), (9.6, 15.4))

    def test_padding_overlap_does_not_shift_next_speech_start(self):
        from modules.vad import optimize_asmr_segments
        from modules.asr_engine import _group_sentences_into_slices
        # Keep two slices explicitly so this tests padding, not merge policy.
        slices = optimize_asmr_segments(
            [SpeechRegion(0, 5), SpeechRegion(5.2, 10.2)], target_max_sec=5.0,
        )
        sentences = [ASRSegment('最初', 0, 5), ASRSegment('次', 5.2, 10.2)]
        blocks = _group_sentences_into_slices(sentences, slices)
        self.assertEqual([(b.start_ms, b.end_ms) for b in blocks], [(0, 5000), (5200, 10200)])

    def test_long_windows_remove_context_before_resolving_overlap(self):
        from modules.vad import optimize_asmr_segments
        from modules.asr_engine import _collapse_overlapping_slices
        slices = optimize_asmr_segments([SpeechRegion(0, 40)])
        collapsed = _collapse_overlapping_slices(slices)
        self.assertEqual([(r.start, r.end) for r in collapsed], [(0, 20), (20, 39), (39, 40)])
        self.assertEqual(slices[0].end, 20.4)

    def test_beta_uses_same_unpadded_asr_output(self):
        from dataclasses import replace
        from modules.vad.analysis import analyze_regions
        from modules.asr_engine import _group_sentences_into_slices
        slices, analysis = analyze_regions([SpeechRegion(10, 15)], {})
        slices = [replace(slices[0], asr_slice=analysis.slices[0])]
        blocks = _group_sentences_into_slices([ASRSegment('日本語', 10, 15)], slices)
        self.assertEqual((blocks[0].start_ms, blocks[0].end_ms), (10000, 15000))
        self.assertEqual(blocks[0].asr_slice.extraction_start_ms, 9600)

    def _run(self, slices, chunks, adapter, asr_kwargs=None, outdir=None,
             **extra):
        if outdir is None:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            outdir = tmp.name
        with mock.patch("modules.asr_engine.create_asr", return_value=adapter), \
             mock.patch("modules.asr_engine.extract_ranges", return_value=chunks), \
             mock.patch("modules.asr_engine.print_progress", return_value=None):
            blocks = transcribe_slices(
                "audio.mp3", "fish-asr", asr_kwargs or {},
                outdir, slices, max_workers=2, **extra
            )
        return blocks

    def test_slice_level_blocks(self):
        # Each optimize_asmr_segments slice yields exactly one block; the block
        # times come from the SLICE timeline, not the ASR timestamps.
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("こんにちは", 2.0, "ja",
                                [ASRSegment("こんにちは", 0.0, 1.0)]),
            "c1.wav": ASRResult("さようなら", 2.0, "ja",
                                [ASRSegment("さようなら", 0.0, 1.0)]),
        })
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav"), _chunk(1, 13.0, 15.0, "c1.wav")]
        blocks = self._run(
            [SpeechRegion(0.0, 6.0), SpeechRegion(13.0, 15.0)],
            chunks, adapter, {"api_key": "k"},
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "こんにちは")
        self.assertEqual((blocks[0].start_ms, blocks[0].end_ms), (0, 6000))
        self.assertEqual(blocks[1].text, "さようなら")
        self.assertEqual((blocks[1].start_ms, blocks[1].end_ms), (13000, 15000))

    def test_multi_sentence_slice_concatenates_text(self):
        # Several ASR sentences inside one slice collapse into a single block
        # whose span is the slice window and text is the concatenation.
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("こんにちは。元気ですか", 6.0, "ja", [
                ASRSegment("こんにちは", 0.0, 1.0),
                ASRSegment("元気ですか", 2.0, 3.0),
            ]),
        })
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav")]
        blocks = self._run(
            [SpeechRegion(0.0, 6.0)],
            chunks, adapter, {"api_key": "k"},
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "こんにちは元気ですか")
        self.assertEqual((blocks[0].start_ms, blocks[0].end_ms), (0, 6000))

    def test_overlapping_slices_are_trimmed(self):
        # Overlapping slice windows (split_long_segment / padding) must not
        # double-cover the timeline: the second block starts at the first one's
        # end, and each slice's own speech lands in its own block.
        adapter = _FakeAdapter(by_path={
            # c1's fragment [1.0, 4.0] → absolute [6.0, 9.0] on the original
            # timeline — past the trimmed slice boundary, so it cannot leak
            # into slice 1's block.
            "c0.wav": ASRResult("あい", 6.0, "ja",
                                [ASRSegment("あい", 0.0, 3.0)]),
            "c1.wav": ASRResult("うえお", 6.0, "ja",
                                [ASRSegment("うえお", 1.0, 4.0)]),
        })
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav"), _chunk(1, 5.0, 11.0, "c1.wav")]
        # Overlapping slices: [0,6] and [5,11] → blocks [0,6] and [6,11].
        blocks = self._run(
            [SpeechRegion(0.0, 6.0), SpeechRegion(5.0, 11.0)],
            chunks, adapter, {"api_key": "k"},
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual((blocks[0].start_ms, blocks[0].end_ms), (0, 6000))
        self.assertEqual((blocks[1].start_ms, blocks[1].end_ms), (6000, 11000))
        self.assertEqual(blocks[0].text, "あい")
        self.assertEqual(blocks[1].text, "うえお")

    def test_deduped_fragment_is_consumed_by_only_one_overlapping_slice(self):
        # Both ASR chunks see the same utterance in their overlap.  The ASR
        # deduper collapses those two observations to one absolute fragment;
        # the per-slice sentence pass must not consume that fragment once for
        # every overlapping VAD window.
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("飲んでないよ", 6.0, "ja", [
                ASRSegment("飲んでないよ", 5.5, 6.0),
            ]),
            "c1.wav": ASRResult("飲んでないよ", 6.0, "ja", [
                ASRSegment("飲んでないよ", 0.5, 1.0),
            ]),
        })
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav"),
                  _chunk(1, 5.0, 11.0, "c1.wav")]

        blocks = self._run(
            [SpeechRegion(0.0, 6.0), SpeechRegion(5.0, 11.0)],
            chunks, adapter, {"api_key": "k"},
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "飲んでないよ")

    def test_identical_utterances_at_distinct_times_are_preserved(self):
        # Text equality alone is not a reason to delete speech: two separate
        # utterances at non-overlapping times must both survive.
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("はいはい", 6.0, "ja", [
                ASRSegment("はい", 1.0, 1.5),
                ASRSegment("はい", 4.0, 4.5),
            ]),
        })

        blocks = self._run(
            [SpeechRegion(0.0, 6.0)],
            [_chunk(0, 0.0, 6.0, "c0.wav")],
            adapter, {"api_key": "k"},
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "はいはい")

    def test_overlapping_complementary_fragments_form_one_sentence(self):
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("今日は天気", 6.0, "ja", [
                ASRSegment("今日は天気", 3.0, 6.0),
            ]),
            "c1.wav": ASRResult("天気がいい", 6.0, "ja", [
                ASRSegment("天気がいい", 0.0, 3.0),
            ]),
        })
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav"),
                  _chunk(1, 5.0, 11.0, "c1.wav")]

        blocks = self._run(
            [SpeechRegion(0.0, 6.0), SpeechRegion(5.0, 11.0)],
            chunks, adapter, {"api_key": "k"},
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "今日は天気がいい")

    def test_unconfigured_raises(self):
        adapter = _FakeAdapter(configured=False)
        with self.assertRaises(ASRError):
            self._run([], [], adapter, {"api_key": ""})

    def test_debug_json_written(self):
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("こんにちは", 2.0, "ja",
                                [ASRSegment("こんにちは", 0.0, 1.0)]),
        })
        with tempfile.TemporaryDirectory() as tmp:
            self._run([SpeechRegion(0.0, 15.0)], [_chunk(0, 0.0, 15.0, "c0.wav")],
                      adapter, {"api_key": "k", "debug": True}, outdir=tmp)
            dbg_path = os.path.join(tmp, "asr_subtitles", "asr_debug.json")
            self.assertTrue(os.path.exists(dbg_path))
            with open(dbg_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["final_segments"]), 1)
            self.assertEqual(data["final_segments"][0]["text"], "こんにちは")

    async def _noop_sleep(self, delay):
        return None

    def test_429_retried_then_succeeds(self):
        # The adapter keeps 429-ing for the first 2 calls; the engine-level
        # retry (with the semaphore released) must exhaust and then succeed.
        adapter = _Flaky429Adapter(fails=2)
        chunks = [_chunk(0, 0.0, 15.0, "c0.wav")]
        with mock.patch("modules.asr_engine.asyncio.sleep", side_effect=self._noop_sleep):
            blocks = self._run([SpeechRegion(0.0, 15.0)], chunks, adapter, {"api_key": "k"})
        self.assertEqual(adapter.calls, 3)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "こんにちは")
        self.assertEqual((blocks[0].start_ms, blocks[0].end_ms), (0, 15000))

    def test_429_exhausted_fails_fast(self):
        adapter = _Flaky429Adapter(fails=10)
        chunks = [_chunk(0, 0.0, 15.0, "c0.wav")]
        with mock.patch("modules.asr_engine.asyncio.sleep", side_effect=self._noop_sleep):
            with self.assertRaises(PipelineError):
                self._run([SpeechRegion(0.0, 15.0)], chunks, adapter, {"api_key": "k"})
        self.assertEqual(adapter.calls, 3)  # initial + 2 extra tries

    def test_slices_passed_to_extract_ranges(self):
        # The slices handed in by the pipeline's VAD stage are exactly what
        # extract_ranges pulls out of the audio.
        adapter = _FakeAdapter(by_path={
            "c0.wav": ASRResult("こんにちは", 2.0, "ja",
                                [ASRSegment("こんにちは", 0.0, 1.0)]),
        })
        slices = [SpeechRegion(0.0, 6.0), SpeechRegion(10.0, 15.0)]
        chunks = [_chunk(0, 0.0, 6.0, "c0.wav"), _chunk(1, 10.0, 15.0, "c1.wav")]
        with mock.patch("modules.asr_engine.extract_ranges",
                        return_value=chunks) as er, \
             mock.patch("modules.asr_engine.create_asr", return_value=adapter), \
             mock.patch("modules.asr_engine.print_progress", return_value=None):
            blocks = transcribe_slices(
                "audio.mp3", "fish-asr", {"api_key": "k"}, "out",
                slices, max_workers=2,
            )
        self.assertEqual(er.call_args[0][1], [(0.0, 6.0), (10.0, 15.0)])
        self.assertEqual(len(blocks), 1)

    def test_empty_slices_returns_empty(self):
        # No speech regions from VAD → nothing to transcribe → empty subtitles.
        adapter = _FakeAdapter()
        blocks = self._run([], [], adapter, {"api_key": "k"})
        self.assertEqual(blocks, [])


if __name__ == "__main__":
    unittest.main()
