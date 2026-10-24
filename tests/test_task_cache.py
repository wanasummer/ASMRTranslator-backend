import json
import os
import tempfile
import unittest
from unittest import mock

from modules.task_cache import JsonResultCache, cache_key, public_settings


class TaskCacheTest(unittest.TestCase):
    def test_atomic_success_survives_failed_write(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResultCache(directory)
            key = cache_key({"text": "你好"})
            cache.save(key, {"text": "你好"})
            with mock.patch("modules.task_cache.os.replace", side_effect=OSError("disk")):
                with self.assertRaises(OSError):
                    cache.save(key, {"text": "new"})
            self.assertEqual(cache.load(key), {"text": "你好"})
            self.assertEqual(len(os.listdir(directory)), 1)

    def test_failure_is_audited_but_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = JsonResultCache(directory)
            cache.fail("unit", RuntimeError("token=secret"))
            self.assertIsNone(cache.load("unit"))
            with open(os.path.join(directory, "unit.json"), encoding="utf-8") as source:
                record = json.load(source)
            self.assertEqual(record["status"], "failed")
            self.assertNotIn("secret", record["detail"])
            cache.save("unit", "recovered")
            self.assertEqual(cache.load("unit"), "recovered")

    def test_reference_content_invalidates_key_and_credentials_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, "voice.wav")
            with open(reference, "wb") as target:
                target.write(b"old")
            settings = {"reference_audio": reference, "api_key": "secret"}
            first = cache_key(public_settings(settings))
            with open(reference, "wb") as target:
                target.write(b"new")
            self.assertNotEqual(first, cache_key(public_settings(settings)))
            self.assertNotIn("api_key", public_settings(settings))
