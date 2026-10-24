"""Tests for modules.mode_builders (listen-mode Strategy pattern).

Run from the project root:  python -m unittest tests.test_mode_builders
Fully offline — delegation is verified with mock.patch, no audio is touched.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.mode_builders import (
    create_segment_builder,
    DualBuilder,
    PhantomBuilder,
    ReplaceBuilder,
    SyncBuilder,
)


class CreateBuilderTest(unittest.TestCase):
    """Factory precedence must match the old if/elif chain exactly."""

    def test_phantom_wins_over_everything(self):
        b = create_segment_builder(
            phantom_mode=True, retain_original=True, sync_mode="follow_original")
        self.assertIsInstance(b, PhantomBuilder)

    def test_phantom_alone(self):
        b = create_segment_builder(phantom_mode=True)
        self.assertIsInstance(b, PhantomBuilder)

    def test_phantom_params_forwarded(self):
        b = create_segment_builder(
            phantom_mode=True, phantom_delay_ms=500,
            phantom_reverb_mix=0.7, phantom_lowpass_hz=2000)
        self.assertIsInstance(b, PhantomBuilder)
        self.assertEqual((b.delay_ms, b.reverb_mix, b.lowpass_hz), (500, 0.7, 2000))

    def test_phantom_keeps_sync_mode(self):
        b = create_segment_builder(phantom_mode=True, sync_mode="follow_translation")
        self.assertIsInstance(b, PhantomBuilder)
        self.assertEqual(b.sync_mode, "follow_translation")

    def test_retain_plus_sync_is_sync(self):
        b = create_segment_builder(
            retain_original=True, sync_mode="follow_translation")
        self.assertIsInstance(b, SyncBuilder)
        self.assertEqual(b.sync_mode, "follow_translation")

    def test_retain_no_sync_is_dual(self):
        b = create_segment_builder(retain_original=True, sync_mode="none")
        self.assertIsInstance(b, DualBuilder)
        self.assertEqual(b.gap_ms, 200)

    def test_sync_without_retain_is_replace(self):
        b = create_segment_builder(sync_mode="follow_original")
        self.assertIsInstance(b, ReplaceBuilder)

    def test_default_is_replace(self):
        b = create_segment_builder()
        self.assertIsInstance(b, ReplaceBuilder)


class BuilderMetaTest(unittest.TestCase):
    """name/subdir per mode — subdir preserves the pre-refactor artifact layout."""

    def test_names(self):
        self.assertEqual(ReplaceBuilder().name, "replace")
        self.assertEqual(DualBuilder().name, "dual")
        self.assertEqual(SyncBuilder().name, "sync")
        self.assertEqual(PhantomBuilder().name, "phantom")

    def test_subdirs(self):
        self.assertEqual(ReplaceBuilder().subdir, "04_build")
        self.assertEqual(DualBuilder().subdir, "04_dual")
        self.assertEqual(SyncBuilder().subdir, "04_sync")
        self.assertEqual(PhantomBuilder().subdir, "04_phantom")


class DelegationTest(unittest.TestCase):
    """Each builder delegates to the right underlying function with stored config."""

    def test_replace_is_identity_and_creates_no_dir(self):
        b = ReplaceBuilder()
        segs = [{"index": 0}, {"index": 1}]
        with tempfile.TemporaryDirectory() as d:
            build_dir = os.path.join(d, "04_build")
            out = b.build(segs, build_dir, max_workers=3)
            self.assertIs(out, segs)  # identical list object, nothing mutated
            self.assertFalse(os.path.exists(build_dir))

    @mock.patch("modules.mode_builders.build_dual_segments")
    def test_dual_delegation(self, m_build):
        m_build.return_value = "RESULT"
        out = DualBuilder(gap_ms=300).build("SEGS", "/out/04_dual", max_workers=5)
        m_build.assert_called_once_with("SEGS", "/out/04_dual", gap_ms=300, max_workers=5)
        self.assertEqual(out, "RESULT")

    @mock.patch("modules.mode_builders.build_sync_segments")
    def test_sync_delegation(self, m_build):
        m_build.return_value = "RESULT"
        out = SyncBuilder(sync_mode="follow_translation").build(
            "SEGS", "/out/04_sync", max_workers=5)
        m_build.assert_called_once_with(
            "SEGS", "/out/04_sync", sync_mode="follow_translation",
            follow_original_rate=1.15, max_workers=5)
        self.assertEqual(out, "RESULT")

    @mock.patch("modules.mode_builders.build_phantom_segments")
    def test_phantom_delegation_omits_fade(self, m_build):
        # The pipeline only ever passed delay/reverb/lowpass; fade stays at defaults.
        m_build.return_value = "RESULT"
        out = PhantomBuilder(delay_ms=600, reverb_mix=0.8, lowpass_hz=1500).build(
            "SEGS", "/out/04_phantom", max_workers=5)
        m_build.assert_called_once_with(
            "SEGS", "/out/04_phantom",
            delay_ms=600, reverb_mix=0.8, lowpass_hz=1500,
            sync_mode="follow_original", follow_original_rate=1.15, max_workers=5,
            orig_fade_ms=0, tts_fade_ms=0, virtual_fade=False)
        self.assertEqual(out, "RESULT")

    @mock.patch("modules.mode_builders.build_phantom_segments")
    def test_phantom_delegation_forwards_sync_mode(self, m_build):
        m_build.return_value = "RESULT"
        out = PhantomBuilder(sync_mode="follow_translation").build(
            "SEGS", "/out/04_phantom", max_workers=5)
        m_build.assert_called_once_with(
            "SEGS", "/out/04_phantom",
            delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
            sync_mode="follow_translation", follow_original_rate=1.15, max_workers=5,
            orig_fade_ms=0, tts_fade_ms=0, virtual_fade=False)
        self.assertEqual(out, "RESULT")

    @mock.patch("modules.mode_builders.build_phantom_segments")
    def test_phantom_delegation_forwards_fades(self, m_build):
        m_build.return_value = "RESULT"
        out = PhantomBuilder(orig_fade_ms=800, tts_fade_ms=600,
                             virtual_fade=True).build("SEGS", "/o", max_workers=1)
        m_build.assert_called_once_with(
            "SEGS", "/o",
            delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
            sync_mode="follow_original", follow_original_rate=1.15, max_workers=1,
            orig_fade_ms=800, tts_fade_ms=600, virtual_fade=True)
        self.assertEqual(out, "RESULT")

    @mock.patch("modules.mode_builders.build_dual_segments")
    def test_config_not_mutated_across_build_calls(self, m_build):
        b = DualBuilder(gap_ms=200)
        b.build(["a"], "/x", max_workers=2)
        b.build(["b"], "/y", max_workers=3)
        self.assertEqual(b.gap_ms, 200)
        self.assertEqual(m_build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
