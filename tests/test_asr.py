"""Tests for modules.asr — the ASR adapter factory.

Run from the project root:  python -m unittest tests.test_asr
Fully offline.
"""

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import modules.asr.factory as factory
from modules.asr import (
    ASRAdapter,
    ChickenRiceASRAdapter,
    CommonASRAdapter,
    FishAudioASRAdapter,
)
from modules.asr.factory import create_asr, get_available_asr_backends, register_asr_backend


class FactoryTest(unittest.TestCase):
    def setUp(self):
        self._registry = dict(factory._registry)

    def tearDown(self):
        factory._registry.clear()
        factory._registry.update(self._registry)

    def test_create_known_backend(self):
        adapter = create_asr("fish-asr", api_key="k")
        self.assertIsInstance(adapter, FishAudioASRAdapter)
        self.assertEqual(adapter.name, "fish-asr")

    def test_create_default_backend(self):
        self.assertIsInstance(create_asr(), FishAudioASRAdapter)

    def test_create_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_asr("does-not-exist")

    def test_register_and_create(self):
        class FakeASR(ASRAdapter):
            @property
            def name(self):
                return "fake-asr"

            @property
            def timeout(self):
                return 30.0

            async def transcribe(self, audio_path, language=None, timestamps=True, client=None):
                return "hello"

        register_asr_backend("fake-asr", FakeASR)
        self.assertIn("fake-asr", get_available_asr_backends())
        adapter = create_asr("fake-asr")
        self.assertIsInstance(adapter, FakeASR)

    def test_available_backends_contains_fish(self):
        self.assertIn("fish-asr", get_available_asr_backends())

    def test_create_chickenrice_backend(self):
        self.assertIsInstance(create_asr("chickenrice"), ChickenRiceASRAdapter)

    def test_create_common_asr_backend(self):
        self.assertIsInstance(create_asr("common-asr"), CommonASRAdapter)


if __name__ == "__main__":
    unittest.main()
