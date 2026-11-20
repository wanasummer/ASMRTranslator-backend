import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from modules.run_results import record_run
from modules.storage import initialize_project, save_json, save_text, delete_document, address, StorageError


def latest(work):
    target = address(Path(work) / 'run')
    with target.store.connect() as connection:
        return json.loads(connection.execute("SELECT payload FROM runs WHERE scope=? ORDER BY updated_at DESC,rowid DESC LIMIT 1", (target.scope,)).fetchone()[0])


class RunResultAssetIdentityTest(unittest.TestCase):
    def test_subtitles_only_record_requires_no_audio_or_tts_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.wav'
            source.write_bytes(b'audio')
            work = Path(directory) / 'generated/assets/subtitles'
            initialize_project(work)
            save_text(work / 'llm_subtitles/final.vtt', 'WEBVTT\n')
            record_run(str(work), 'succeeded', str(source), output_mode='subtitles-only')
            self.assertEqual(latest(work)['files'], {'finalVtt':'llm_subtitles/final.vtt'})
            with self.assertRaises(StorageError):
                record_run(str(work), 'succeeded', str(source))
            delete_document(work / 'llm_subtitles/final.vtt')
            with self.assertRaises(StorageError):
                record_run(str(work), 'succeeded', str(source), output_mode='subtitles-only')

    def test_asset_identity_and_attempt_history_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.wav'
            source.write_bytes(b'audio')
            work = Path(directory) / 'generated/assets/stable'
            work.mkdir(parents=True)
            initialize_project(work)
            (work / 'output.mp3').write_bytes(b'mp3')
            save_json(work / 'debug/_parsed_vtt.json', [])
            save_text(work / 'llm_subtitles/final.vtt', 'WEBVTT\n')
            with patch.dict(os.environ, {'ASMR_ASSET_ID':'asset-stable-1'}):
                record_run(str(work), 'running', str(source))
                record_run(str(work), 'succeeded', str(source))
            self.assertEqual(latest(work)['assetId'], 'asset-stable-1')
            with address(work / 'run').store.connect() as connection:
                self.assertEqual(connection.execute('SELECT count(*) FROM runs').fetchone()[0], 2)
