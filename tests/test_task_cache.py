import json
import tempfile
import unittest
from pathlib import Path
from modules.task_cache import SqliteResultCache, cache_key, public_settings
from modules.storage import StorageError


class TaskCacheTest(unittest.TestCase):
    def test_atomic_success_survives_failed_write(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResultCache(directory)
            key = cache_key({"text": "你好"})
            cache.save(key, {"text": "你好"})
            with cache.store.connect() as connection:
                connection.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON cache_entries BEGIN SELECT RAISE(ABORT,'disk'); END")
            with self.assertRaises(StorageError):
                cache.save(key, {"text": "new"})
            self.assertEqual(cache.load(key), {"text": "你好"})
            self.assertEqual([path.name for path in Path(directory).iterdir()], ['project.sqlite3'])

    def test_failure_is_audited_but_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SqliteResultCache(directory)
            cache.fail("unit", RuntimeError("token=secret"))
            self.assertIsNone(cache.load("unit"))
            with cache.store.connect() as connection:
                status, payload = connection.execute("SELECT status,payload FROM cache_entries").fetchone()
            self.assertEqual(status, "failed")
            self.assertNotIn("secret", json.loads(payload)["detail"])
            cache.save("unit", "recovered")
            self.assertEqual(cache.load("unit"), "recovered")

    def test_reference_content_invalidates_key_and_credentials_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'voice.wav'
            reference.write_bytes(b'old')
            settings = {"reference_audio": str(reference), "api_key": "secret"}
            first = cache_key(public_settings(settings))
            reference.write_bytes(b'new')
            self.assertNotEqual(first, cache_key(public_settings(settings)))
            self.assertNotIn('api_key', public_settings(settings))
