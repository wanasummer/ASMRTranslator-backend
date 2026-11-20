from pathlib import Path
import tempfile
import unittest
from modules.pipeline import _clean_previous_run, _load_cached_asr
from modules.task_cache import SqliteResultCache
from modules.storage import initialize_project, save_text, contains, document_digest, DatabaseLog, read_log


class PipelineArtifactCleanupTest(unittest.TestCase):
    def test_whole_asr_cache_requires_matching_input_and_intact_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize_project(directory)
            path = Path(directory) / 'asr_subtitles/cleaned.vtt'
            save_text(path, 'WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nテスト\n')
            cache = SqliteResultCache(directory, 'asr_stage_results')
            cache.save('matching', {'sha256':document_digest(path)})
            self.assertIsNotNone(_load_cached_asr(directory, {}, 'matching'))
            self.assertIsNone(_load_cached_asr(directory, {}, 'changed-settings'))
            save_text(path, 'modified')
            self.assertIsNone(_load_cached_asr(directory, {}, 'matching'))

    def test_repeat_run_preserves_logs_and_cache_but_invalidates_published_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize_project(directory)
            with DatabaseLog(Path(directory) / 'pipeline.log') as log:
                log.write('previous attempt\n')
                reference = log.name
            save_text(Path(directory) / 'asr_subtitles/cleaned.vtt', 'cache')
            save_text(Path(directory) / 'llm_subtitles/final.vtt', 'old final')
            audio = Path(directory) / 'output.mp3'
            audio.write_bytes(b'audio')
            self.assertEqual(_clean_previous_run(directory), 1)
            self.assertEqual(read_log(reference), 'previous attempt\n')
            self.assertTrue(contains(Path(directory) / 'asr_subtitles/cleaned.vtt'))
            self.assertFalse(contains(Path(directory) / 'llm_subtitles/final.vtt'))
            self.assertFalse(audio.exists())

    def test_repeat_run_drops_previous_voice_assets_and_tempo_records(self):
        with tempfile.TemporaryDirectory() as directory:
            initialize_project(directory)
            tts = Path(directory) / 'chunk_0/02_tts'
            tts.mkdir(parents=True)
            voice, raw = tts / 'voice_0003.wav', tts / 'tts_0003.mp3'
            voice.write_bytes(b'voice')
            raw.write_bytes(b'raw')
            save_text(tts / 'tts_tempo_0003.json', '{"rate": 1.2}')
            self.assertEqual(_clean_previous_run(directory), 1)
            self.assertFalse(voice.exists())
            self.assertTrue(raw.exists())
            self.assertFalse(contains(tts / 'tts_tempo_0003.json'))

    def test_language_specific_stage_boundaries_are_loaded_without_subtitle_files(self):
        for language, name, text in [('ja','cleaned.vtt','テスト'),('zh','raw.vtt','测试')]:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as directory:
                initialize_project(directory)
                path = Path(directory) / 'asr_subtitles' / name
                save_text(path, f'WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n{text}\n')
                blocks, detected, _ = _load_cached_asr(directory, {'output_language':language})
                self.assertEqual((detected,blocks[0].text), (language,text))
                self.assertFalse(path.exists())
