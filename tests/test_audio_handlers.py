"""Tests for modules.audio_handlers (post-TTS handler chain).

Run from the project root:  python -m unittest tests.test_audio_handlers
Offline — the detector, TTS adapter, and phantom DSP are all mocked; only the
chain/handler logic is exercised.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.audio_handlers import (
    AudioHandler,
    AudioHandlerChain,
    PhantomReverbHandler,
    TTSArtifactHandler,
    build_tts_handler_chain,
)

CLEAN = {"flagged": False, "error": None, "regions": [], "flags": set()}
FLAGGED = {"flagged": True, "error": None, "regions": [], "flags": {"dark"}}


class RecordingHandler(AudioHandler):
    name = "recording"

    def __init__(self, tag):
        self.tag = tag

    def process(self, segments, output_dir, max_workers=4):
        for seg in segments:
            seg.setdefault("_order", []).append(self.tag)
        return segments


class ChainTest(unittest.TestCase):
    def test_runs_handlers_in_order(self):
        segs = [{"index": 0}]
        chain = (AudioHandlerChain()
                 .add(RecordingHandler("a"))
                 .add(RecordingHandler("b")))
        out = chain.run(segs, "/tmp")
        self.assertIs(out, segs)
        self.assertEqual(segs[0]["_order"], ["a", "b"])


class TtsArtifactHandlerTest(unittest.TestCase):
    def _handler(self):
        return TTSArtifactHandler(backend="edge-tts", rate="+0%", pitch="+0Hz")

    def _seg(self, d, idx=0):
        tts = os.path.join(d, f"tts_{idx:04d}.mp3")
        with open(tts, "wb") as f:
            f.write(b"\xff\xf3" * 100)
        return {"index": idx, "text": "你好", "speaker": "遥", "tts_file": tts}

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_clean_does_not_regenerate(self, m_adapter, m_detect):
        m_detect.return_value = CLEAN
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            self._handler().process([seg], d, max_workers=1)
        self.assertEqual(seg["artifact"]["status"], "clean")
        m_adapter.return_value.synthesize_sync.assert_not_called()

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_flagged_regenerates_once_then_clean(self, m_adapter, m_detect):
        m_detect.side_effect = [FLAGGED, CLEAN]  # first bad, after-regenerate clean
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            self._handler().process([seg], d, max_workers=1)
        self.assertEqual(seg["artifact"]["status"], "regenerated")
        adapter = m_adapter.return_value
        adapter.get_voice_for_speaker.assert_called_once_with("遥")
        adapter.synthesize_sync.assert_called_once()

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_flagged_segments_use_separate_adapters(self, m_adapter, m_detect):
        adapters = [mock.Mock(), mock.Mock()]
        m_adapter.side_effect = adapters
        seen = {}

        def detect(path, **_kwargs):
            path = os.path.basename(path).removeprefix("artifact_")
            count = seen.get(path, 0)
            seen[path] = count + 1
            return FLAGGED if count == 0 else CLEAN

        m_detect.side_effect = detect
        with tempfile.TemporaryDirectory() as directory:
            segments = [self._seg(directory, index) for index in range(2)]
            self._handler().process(segments, directory, max_workers=2)

        self.assertEqual(m_adapter.call_count, 2)
        self.assertEqual([adapter.synthesize_sync.call_count for adapter in adapters], [1, 1])
        self.assertEqual([segment["artifact"]["status"] for segment in segments],
                         ["regenerated", "regenerated"])

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_regeneration_is_cached_and_preserves_original_tts(self, m_adapter, m_detect):
        m_detect.side_effect = [FLAGGED, CLEAN]
        with tempfile.TemporaryDirectory() as directory:
            segment = self._seg(directory)
            original = segment["tts_file"]
            handler = self._handler()
            handler.process([segment], directory, max_workers=1)
            regenerated = segment["tts_file"]
            self.assertNotEqual(regenerated, original)
            self.assertTrue(os.path.isfile(original))
            segment["tts_file"] = original
            handler.process([segment], directory, max_workers=1)
            self.assertEqual(segment["tts_file"], regenerated)
            m_adapter.return_value.synthesize_sync.assert_called_once()
            self.assertEqual(m_detect.call_count, 2)

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_still_flagged_after_regenerate_is_accepted(self, m_adapter, m_detect):
        m_detect.side_effect = [FLAGGED, FLAGGED]  # still bad → accept + log
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            self._handler().process([seg], d, max_workers=1)
        self.assertEqual(seg["artifact"]["status"], "accepted_flagged")
        self.assertEqual(m_adapter.return_value.synthesize_sync.call_count, 1)

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_missing_file_skipped(self, m_adapter, m_detect):
        seg = {"index": 9, "tts_file": os.path.join("nope", "missing.mp3")}
        self._handler().process([seg], "/tmp", max_workers=1)
        self.assertEqual(seg["artifact"]["status"], "skipped")
        m_detect.assert_not_called()

    @mock.patch("modules.audio_handlers.detect_tts_artifact")
    @mock.patch("modules.audio_handlers.get_tts_adapter")
    def test_detector_error_fails_fast(self, m_adapter, m_detect):
        m_detect.return_value = {"flagged": False, "error": "decode fail",
                                 "regions": [], "flags": set()}
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            with self.assertRaises(RuntimeError):
                self._handler().process([seg], d, max_workers=1)
        m_adapter.return_value.synthesize_sync.assert_not_called()


class PhantomReverbHandlerTest(unittest.TestCase):
    def _seg(self, d):
        tts = os.path.join(d, "tts_0000.mp3")
        with open(tts, "wb") as f:
            f.write(b"\xff\xf3" * 100)
        return {"index": 0, "tts_file": tts}

    @mock.patch("modules.audio_handlers.apply_phantom_dsp")
    def test_processes_and_records_keys(self, m_dsp):
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            PhantomReverbHandler(lowpass_hz=3000, reverb_mix=0.5).process([seg], d, max_workers=1)
        m_dsp.assert_called_once()
        self.assertEqual(seg["phantom_voice_src"], os.path.join(d, "tts_0000.mp3"))
        self.assertEqual(seg["phantom_dsp_file"],
                         os.path.join(d, "phantom_dsp_0000.wav"))
        self.assertEqual(seg["tts_file"], seg["phantom_dsp_file"])

    @mock.patch("modules.audio_handlers.apply_phantom_dsp")
    def test_skips_when_dsp_is_fresh(self, m_dsp):
        with tempfile.TemporaryDirectory() as d:
            seg = self._seg(d)
            dsp = os.path.join(d, "phantom_dsp_0000.wav")
            with open(dsp, "wb") as f:  # create after tts → newer mtime
                f.write(b"\x00\x00" * 200)
            PhantomReverbHandler().process([seg], d, max_workers=1)
        m_dsp.assert_not_called()
        self.assertEqual(seg["tts_file"], dsp)

    @mock.patch("modules.audio_handlers.apply_phantom_dsp")
    def test_missing_tts_skipped(self, m_dsp):
        seg = {"index": 0, "tts_file": os.path.join("nope", "missing.mp3")}
        PhantomReverbHandler().process([seg], "/tmp", max_workers=1)
        m_dsp.assert_not_called()
        self.assertIsNone(seg.get("phantom_dsp_file"))


class FactoryTest(unittest.TestCase):
    def test_default_is_detect_only(self):
        c = build_tts_handler_chain()
        self.assertEqual([h.name for h in c.handlers], ["tts-artifact"])

    def test_phantom_appends_reverb(self):
        c = build_tts_handler_chain(phantom_mode=True)
        self.assertEqual([h.name for h in c.handlers],
                         ["tts-artifact", "phantom-reverb"])

    def test_phantom_only(self):
        c = build_tts_handler_chain(detect_artifact=False, phantom_mode=True)
        self.assertEqual([h.name for h in c.handlers], ["phantom-reverb"])

    def test_detect_disabled_is_empty(self):
        c = build_tts_handler_chain(detect_artifact=False)
        self.assertEqual(c.handlers, [])

    def test_params_forwarded(self):
        c = build_tts_handler_chain(phantom_mode=True, phantom_reverb_mix=0.7,
                                    phantom_lowpass_hz=2000, tts_rate="-5%")
        artifact = c.handlers[0]
        reverb = c.handlers[1]
        self.assertEqual(artifact.rate, "-5%")
        self.assertEqual(reverb.reverb_mix, 0.7)
        self.assertEqual(reverb.lowpass_hz, 2000)


if __name__ == "__main__":
    unittest.main()
