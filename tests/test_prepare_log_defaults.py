import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_log_defaults import prepare_log_defaults


class PrepareLogDefaultsTest(unittest.TestCase):
    def test_missing_values_are_empty_and_other_credentials_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "_log_upload_defaults.json"
            prepare_log_defaults(output, {"FISH_API_KEY": "must-not-ship"})
            self.assertEqual(json.loads(output.read_text()), {"ASMR_LOG_UPLOAD_URL": ""})

    def test_release_requires_endpoint_and_validation_does_not_expose_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "_log_upload_defaults.json"
            for env in [{"ASMR_REQUIRE_LOG_UPLOAD": "1"},
                        {"ASMR_REQUIRE_LOG_UPLOAD": "1", "ASMR_LOG_UPLOAD_URL": "https://logs.example.test/log", "ASMR_DISABLE_LOG_UPLOAD": "1"},
                        {"ASMR_LOG_UPLOAD_URL": "ftp://private-value.test/log"},
                        {"ASMR_LOG_UPLOAD_URL": "https://logs.example.test/private-value\nsecond-line"}]:
                with self.subTest(env=env), self.assertRaises(ValueError) as caught:
                    prepare_log_defaults(output, env)
                self.assertNotIn("private-value", str(caught.exception))
                self.assertFalse(output.exists())
