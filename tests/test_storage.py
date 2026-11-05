"""Project persistence contracts: one database, scoped data, short atomic writes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from modules.storage import (address, initialize_project, save_json, read_json,
                             save_text, read_text, DatabaseLog, read_log, StorageError)
from modules.task_cache import SqliteResultCache


class ProjectStorageTests(unittest.TestCase):
    def test_assets_stages_and_unicode_share_one_database_without_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'generated'
            for asset in ('first', 'second'):
                work = root / 'assets' / asset
                initialize_project(work)
                reference = save_json(work / 'detect_output/alignment.json', {'text': '声優：こんにちは😀', 'id': asset})
                self.assertEqual(read_json(reference)['id'], asset)
                cache = SqliteResultCache(str(work), 'asr')
                cache.save('same-key', asset)
                with DatabaseLog(work / 'attempt.log') as stream:
                    stream.write(asset + '\n')
                    self.assertEqual(read_log(stream.name), asset + '\n')
            self.assertEqual(SqliteResultCache(str(root/'assets/first'), 'asr').load('same-key'), 'first')
            self.assertEqual(SqliteResultCache(str(root/'assets/second'), 'asr').load('same-key'), 'second')
            self.assertEqual([path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()], ['project.sqlite3'])

    def test_parallel_workers_commit_isolated_units_and_rollback_failed_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize_project(directory)
            def write(index):
                return save_json(Path(directory) / f'unit-{index}', {'index': index})
            with ThreadPoolExecutor(max_workers=8) as pool:
                references = list(pool.map(write, range(32)))
            self.assertEqual([read_json(ref)['index'] for ref in references], list(range(32)))
            target = address(references[0])
            with target.store.connect() as connection:
                connection.execute("CREATE TRIGGER reject_update BEFORE UPDATE ON documents BEGIN SELECT RAISE(ABORT,'full'); END")
            with self.assertRaises(StorageError):
                save_json(references[0], {'index': -1})
            self.assertEqual(read_json(references[0]), {'index': 0})

    def test_scope_and_document_names_roundtrip_spaces_plus_and_unicode(self):
        with tempfile.TemporaryDirectory(prefix='storage + 中文 ') as directory:
            work = Path(directory) / 'generated/assets/音轨 + 一'
            initialize_project(work)
            reference = save_text(work/'字幕 + 最终.vtt', 'WEBVTT\n你好')
            self.assertEqual(read_text(reference), 'WEBVTT\n你好')

    def test_unsupported_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = initialize_project(directory)
            with store.connect() as connection:
                connection.execute('UPDATE schema_info SET version=999')
            with self.assertRaises(StorageError):
                store.get('', 'anything')

    def test_moved_database_retains_scope_content_without_original_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory)/'before/generated/assets/track'
            initialize_project(before)
            save_json(before/'arrangement/draft.json', {'text':'保留编辑'})
            shutil.move(str(Path(directory)/'before'), str(Path(directory)/'after'))
            self.assertEqual(read_json(Path(directory)/'after/generated/assets/track/arrangement/draft.json'), {'text':'保留编辑'})
