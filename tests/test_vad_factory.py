"""VAD factory contracts and compatibility with the original engine API."""

import unittest
from unittest.mock import Mock, patch

from modules.vad import NoOpVAD
from modules.vad.plugin import ASMRPluginVAD
from modules.vad.engine import create_vad as legacy_create_vad
from modules.vad.factory import (
    VAD_ADAPTERS, VadAdapterRegistration, create_vad, resolve_vad_config,
)


class VadFactoryTests(unittest.TestCase):
    def test_new_registration_owns_configuration_and_construction(self):
        adapter = NoOpVAD("test adapter")
        factory = Mock(return_value=adapter)

        def configure(config):
            config["effective"] = 123
            return config

        original = {"backend": "test-vad"}
        with patch.dict(VAD_ADAPTERS, {"test-vad": VadAdapterRegistration(factory, configure)}):
            self.assertEqual(resolve_vad_config(original), {**original, "effective": 123})
            self.assertIs(create_vad(original), adapter)
        factory.assert_called_once_with({**original, "effective": 123})
        self.assertEqual(original, {"backend": "test-vad"})

    def test_original_engine_entry_point_uses_shared_factory(self):
        self.assertIs(legacy_create_vad, create_vad)
        self.assertIsInstance(create_vad({"backend": "asmr-vad"}), ASMRPluginVAD)

    def test_unknown_backend_is_rejected_by_both_entry_points(self):
        for operation in (create_vad, resolve_vad_config):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "Unknown VAD backend"):
                operation({"backend": "missing"})

    def test_disabled_asmr_does_not_construct_plugin(self):
        with patch("modules.vad.plugin.LocalVADClient", side_effect=AssertionError("plugin loaded")):
            self.assertIsInstance(create_vad({"enabled": False, "backend": "asmr-vad"}), NoOpVAD)

    def test_constructor_key_error_is_not_misreported_as_unknown_backend(self):
        factory = Mock(side_effect=KeyError("model setting"))
        with patch.dict(VAD_ADAPTERS, {"test-vad": VadAdapterRegistration(factory)}):
            with self.assertRaisesRegex(KeyError, "model setting"):
                create_vad({"backend": "test-vad"})
