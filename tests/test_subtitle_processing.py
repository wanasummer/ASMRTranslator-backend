"""Migrated behavioral coverage for separated LLM runtime and capabilities."""
import asyncio
import contextlib
import io
import json
import unittest
from unittest import mock
from modules.llm import LLMConfig, LLMRuntime, is_configured
from modules.llm import runtime as llm_runtime
from modules.llm.factory import create_capabilities
from modules.llm.providers import parsing as provider_parsing
from modules.llm.capabilities.capPolish import parsing as polish_parsing
from modules.llm.capabilities.capTranslation import parsing as translation_parsing
from modules.llm import tts_style
from modules import text_normalization as normalization
from modules.subtitle_processing import timeline
from modules.text_normalization import strip_emotion_tags
from modules.error import LLMError, PipelineError
from modules.tts_engine import prepare_text_for_synthesis
from modules.vtt_parser import VTTBlock
from tests.llm_support import FakeResponse as _FakeResponse, FakeHttpx, process_subtitles, process_asr


def _blocks(*texts):
    return [VTTBlock(index=i, start_ms=i * 1000, end_ms=i * 1000 + 500, text=t, speaker='遥' if i % 2 == 0 else None) for i, t in enumerate(texts)]

def _openai_resp(content):
    return _FakeResponse(200, {'choices': [{'message': {'content': content}}]})

def _anthropic_resp(text):
    return _FakeResponse(200, {'content': [{'text': text}]})

def _cfg(**overrides):
    base = {'provider': 'openai', 'base_url': 'https://api.openai.com/v1', 'api_key': 'sk-test', 'model': 'gpt-test', 'max_retries': 0}
    base.update(overrides)
    return base

def _slices():
    return [VTTBlock(index=0, start_ms=0, end_ms=5000, text='原转写一'), VTTBlock(index=1, start_ms=5000, end_ms=10000, text='原转写二')]

class EnhanceTest(unittest.TestCase):

    def test_content_policy_error_is_logged_and_still_stops_conversion(self):
        from modules.error_reporter import build_error_report
        marker = 'diagnostic-tail-marker'
        body = json.dumps({'error': {'message': '当前请求无权执行', 'type': 'content_policy_violation', 'code': None}, 'api_key': 'echoed-secret', 'extra': 'x' * 700 + marker}, ensure_ascii=False)
        response = _FakeResponse(403, json.loads(body), body)
        response.headers = {'x-request-id': 'request-123', 'Authorization': 'Bearer must-not-log'}
        fake = FakeHttpx([response])
        output = io.StringIO()
        blocks = _blocks('original')
        with mock.patch.object(llm_runtime, 'httpx', fake), contextlib.redirect_stdout(output):
            with self.assertRaises(LLMError) as caught:
                blocks, _ = process_subtitles(blocks, _cfg(base_url='https://name:url-password@example.test/v1?token=url-secret'))
        error = caught.exception
        self.assertEqual(error.descriptor.code, 'LLM_CONTENT_REJECTED')
        self.assertIn('内容策略', error.descriptor.user_message)
        self.assertIn('当前请求无权执行', error.to_payload()['message'])
        self.assertNotIn('API Key', error.descriptor.user_message)
        self.assertEqual(blocks[0].text, 'original')
        log = output.getvalue()
        record = json.loads(next((line.split('[llm-http-error] ', 1)[1] for line in log.splitlines() if '[llm-http-error]' in line)))
        self.assertEqual(record['request_ids'], {'x-request-id': 'request-123'})
        self.assertEqual(record['status'], 403)
        self.assertEqual(record['model'], 'gpt-test')
        self.assertIn(marker, record['response_body'])
        for secret in ('echoed-secret', 'url-password', 'url-secret', 'must-not-log', 'sk-test'):
            self.assertNotIn(secret, log)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pipeline.log'
            path.write_text(log, encoding='utf-8')
            error.log_path = str(path)
            report = build_error_report(error, '', environ={})
        self.assertIn(marker, report)
        self.assertIn('request-123', report)

    def test_403_access_error_and_401_auth_error_are_distinct(self):
        for status, body, expected in [(403, '{"error":{"message":"access denied"}}', 'LLM_ACCESS_DENIED'), (401, '{"error":{"message":"invalid key"}}', 'LLM_AUTH_INVALID')]:
            with self.subTest(status=status):
                fake = FakeHttpx([_FakeResponse(status, json.loads(body), body)])
                with mock.patch.object(llm_runtime, 'httpx', fake):
                    with self.assertRaises(LLMError) as caught:
                        _, (_, _) = process_subtitles(_blocks('original'), _cfg())
                self.assertEqual(caught.exception.descriptor.code, expected)

    def test_optimization_batch_preserves_policy_error(self):
        body = '{"error":{"type":"content_policy_violation","message":"denied"}}'
        fake = FakeHttpx([_openai_resp('summary'), _FakeResponse(403, json.loads(body), body)])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(LLMError) as caught:
                _, (_, _) = process_subtitles(_blocks('original'), _cfg())
        self.assertEqual(caught.exception.descriptor.code, 'LLM_CONTENT_REJECTED')

    def test_optimization_batch_preserves_gemini_rate_limit(self):
        body = '[{"error":{"code":429,"message":"Quota exceeded for this model"}}]'
        fake = FakeHttpx([_openai_resp('summary'), _FakeResponse(429, json.loads(body), body)])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(LLMError) as caught:
                _, (_, _) = process_subtitles(_blocks('original'), _cfg())
        self.assertEqual(caught.exception.descriptor.code, 'LLM_RATE_LIMITED')
        self.assertIn('Quota exceeded', caught.exception.detail)

    def test_large_policy_response_is_classified_before_log_truncation(self):
        data = {'padding': 'x' * 33000, 'error': {'code': 'content_policy_violation', 'message': 'policy denied'}}
        response = _FakeResponse(400, data, json.dumps(data))
        fake = FakeHttpx([response])
        output = io.StringIO()
        with mock.patch.object(llm_runtime, 'httpx', fake), contextlib.redirect_stdout(output):
            with self.assertRaises(LLMError) as caught:
                _, (_, _) = process_subtitles(_blocks('original'), _cfg())
        self.assertEqual(caught.exception.descriptor.code, 'LLM_CONTENT_REJECTED')
        self.assertEqual(caught.exception.to_payload()['context']['status'], 400)
        record = json.loads(next((line.split('[llm-http-error] ', 1)[1] for line in output.getvalue().splitlines() if '[llm-http-error]' in line)))
        self.assertTrue(record['response_body_truncated'])
        self.assertLessEqual(len(record['response_body']), 32768)
        self.assertEqual(record['error_code'], 'content_policy_violation')

    def test_unparsable_line_fails_incomplete_contract(self):
        blocks = _blocks('原句一', '原句二', '原句三')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t优化一\n不是合法行\n2\t优化三')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                blocks, _ = process_subtitles(blocks, _cfg())

    def test_missing_rows_use_targeted_batch_then_single_recovery(self):
        blocks = _blocks('原句零', '原句一', '原句二', '原句三')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t优化零\n3\t优化三'), _openai_resp('1\t优化一'), _openai_resp('优化二')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg(max_retries=1))
        self.assertEqual(stats['rewritten'], 4)
        self.assertEqual([block.text for block in blocks], ['优化零', '优化一', '优化二', '优化三'])
        self.assertEqual(fake.calls[2]['json']['messages'][-1]['content'], '1\t原句一\n2\t遥\t原句二')
        self.assertEqual(fake.calls[3]['json']['messages'][-1]['content'], '原句二')

    def test_single_missing_row_is_recovered_without_renumbering(self):
        blocks = _blocks('原句零', '原句一')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t优化零'), _openai_resp('优化一')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, _ = process_subtitles(blocks, _cfg(max_retries=1))
        self.assertEqual([block.text for block in blocks], ['优化零', '优化一'])
        self.assertEqual(fake.calls[2]['json']['messages'][-1]['content'], '原句一')

    def test_invalid_whole_batch_is_recovered(self):
        blocks = _blocks('原句零', '原句一')
        fake = FakeHttpx([_openai_resp('摘要'), _FakeResponse(200, {'content': []}), _openai_resp('0\t优化零\n1\t优化一')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg(max_retries=1))
        self.assertEqual(stats['rewritten'], 2)
        self.assertEqual([block.text for block in blocks], ['优化零', '优化一'])
        self.assertEqual(fake.calls[2]['json']['messages'][-1]['content'], '0\t遥\t原句零\n1\t原句一')

    def test_long_missing_run_is_retried_in_small_batches(self):
        blocks = _blocks(*[f'原句{i}' for i in range(20)])
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t优化0\n19\t优化19'), *[_openai_resp('\n'.join((f'{i}\t优化{i}' for i in indices))) for indices in (range(1, 9), range(9, 17), range(17, 19))]])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg(max_retries=1))
        self.assertEqual(stats['rewritten'], 20)
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual([block.text for block in blocks], [f'优化{i}' for i in range(20)])

    def test_unrecoverable_missing_row_keeps_all_original_text(self):
        blocks = _blocks('原句零', '原句一', '原句二')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t优化零'), _openai_resp('无编号的批次回复'), _openai_resp('```'), _openai_resp('```')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError) as caught:
                blocks, _ = process_subtitles(blocks, _cfg(max_retries=1))
        self.assertEqual(caught.exception.descriptor.code, 'LLM_OPTIMIZE_INCOMPLETE')
        self.assertIn('missing optimized rows after retries', caught.exception.detail)
        self.assertEqual([block.text for block in blocks], ['原句零', '原句一', '原句二'])

    def test_batching_over_batch_size(self):
        blocks = _blocks(*[f'原句{i}' for i in range(90)])
        p1 = _openai_resp('摘要')
        lines = '\n'.join((f'{i}\t优化{i}' for i in range(90)))
        batch1 = '\n'.join((f'{i}\t优化{i}' for i in range(0, 80)))
        batch2 = '\n'.join((f'{i}\t优化{i}' for i in range(80, 90)))
        fake = FakeHttpx([p1, _openai_resp(batch1), _openai_resp(batch2)])
        cfg = _cfg(batch_size=80)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, cfg)
        self.assertEqual(stats['rewritten'], 90)
        self.assertEqual(len(fake.calls), 3)
        self.assertTrue(all((b.text == f'优化{i}' for i, b in enumerate(blocks))))

    def test_phase1_http_error_still_attempts_optimization(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_FakeResponse(500, {}, 'boom')])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                blocks, _ = process_subtitles(blocks, cfg)
        self.assertEqual(blocks[0].text, '原句')
        self.assertEqual(len(fake.calls), 2)

    def test_phase2_batch_failure_fails_fast(self):
        blocks = _blocks(*['原句'] * 2)
        fake = FakeHttpx([_openai_resp('摘要'), _FakeResponse(500, {}, 'boom')])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                blocks, _ = process_subtitles(blocks, cfg)
        self.assertTrue(all((b.text == '原句' for b in blocks)))

    def test_not_configured_skips_network(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, {})
        self.assertFalse(stats['enabled'])
        self.assertEqual(fake.calls, [])
        self.assertEqual(blocks[0].text, '原句')

    def test_is_configured(self):
        self.assertTrue(is_configured(LLMConfig(provider='openai', api_key='k', model='m')))
        self.assertTrue(is_configured(LLMConfig(provider='anthropic', api_key='k', model='m')))
        self.assertFalse(is_configured(LLMConfig(provider='openai', api_key='', model='m')))
        self.assertFalse(is_configured(LLMConfig(provider='gemini', api_key='k', model='m')))
        self.assertFalse(is_configured(LLMConfig()))

class BuildSentenceBlocksTest(unittest.TestCase):

    def test_char_proportional_splitting(self):
        blocks = _slices()
        per_slice = {0: ['你好。', '世界。'], 1: ['结束。']}
        out = timeline._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([(b.index, b.start_ms, b.end_ms, b.text) for b in out], [(0, 0, 2500, '你好。'), (1, 2500, 5000, '世界。'), (2, 5000, 10000, '结束。')])

    def test_last_sentence_absorbs_remainder(self):
        blocks = [VTTBlock(index=0, start_ms=0, end_ms=6000, text='x')]
        per_slice = {0: ['短', '一个很长很长的句子占据了大部分内容']}
        out = timeline._build_sentence_blocks(blocks, per_slice)
        self.assertEqual(out[-1].end_ms, 6000)

    def test_slice_without_sentences_is_discarded(self):
        blocks = _slices()
        out = timeline._build_sentence_blocks(blocks, {})
        self.assertEqual(out, [])

    def test_upstream_original_only_slice_is_preserved_without_translation(self):
        blocks = _slices()
        blocks[0].text = ''
        blocks[0].skip_tts = True
        out = timeline._build_sentence_blocks(blocks, {1: ['再见。']})
        self.assertEqual([(b.index, b.text, b.start_ms, b.end_ms, b.skip_tts) for b in out], [(0, '', 0, 5000, True), (1, '再见。', 5000, 10000, False)])

    def test_partial_translation_discards_only_missing(self):
        blocks = _slices()
        per_slice = {1: ['再见。']}
        out = timeline._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([(b.index, b.text, b.start_ms, b.end_ms) for b in out], [(0, '再见。', 5000, 10000)])

    def test_indices_are_global_and_ordered(self):
        blocks = _slices()
        per_slice = {0: ['甲', '乙'], 1: ['丙']}
        out = timeline._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([b.index for b in out], [0, 1, 2])

    def test_detected_japanese_timeline_is_not_split_by_chinese_sentences(self):
        blocks = _slices()
        per_slice = {0: ['今天休息。', '好开心啊。'], 1: ['明天见。']}
        out = timeline._build_sentence_blocks(blocks, per_slice, preserve_source_timeline=True)
        self.assertEqual([(b.start_ms, b.end_ms, b.text) for b in out], [(0, 5000, '今天休息。好开心啊。'), (5000, 10000, '明天见。')])

class EnhanceAsrTest(unittest.TestCase):

    def test_end_to_end_translate_then_polish(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0.1\t你好。\n0.2\t世界。\n1.1\t再见。'), _openai_resp('0\t[whispering] 你好。\n1\t世界……\n2\t再见！\n')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, _cfg())
        self.assertTrue(stats['enabled'])
        self.assertTrue(stats['summary_ok'])
        self.assertEqual(stats['translated'], 3)
        self.assertEqual(stats['rewritten'], 3)
        self.assertEqual(stats['total'], 3)
        self.assertEqual([b.text for b in out], ['[whispering] 你好。', '世界……', '再见！'])
        self.assertEqual([b.start_ms for b in out], [0, 2500, 5000])
        self.assertEqual([b.end_ms for b in out], [2500, 5000, 10000])
        self.assertEqual(len(fake.calls), 3)

    def test_translate_only_skips_polish_and_emotion_tags(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('[{"slice_id":0,"sentences":["你好。","世界。"]},{"slice_id":1,"sentences":["再见。"]}]')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, _cfg(optimize_enabled=False))
        self.assertEqual([b.text for b in out], ['你好。', '世界。', '再见。'])
        self.assertEqual(stats['translated'], 3)
        self.assertEqual(stats['rewritten'], 0)
        self.assertEqual(len(fake.calls), 2)
        request = json.loads(fake.calls[1]['json']['messages'][-1]['content'])
        self.assertEqual(request, [{'slice_id': 0, 'source_text': '原转写一'}, {'slice_id': 1, 'source_text': '原转写二'}])

    def test_detected_source_boundaries_survive_translation(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('[{"slice_id":0,"sentences":["你好。","世界。"]},{"slice_id":1,"sentences":["再见。"]}]')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, _ = process_asr(blocks, _cfg(optimize_enabled=False), preserve_source_timeline=True)
        self.assertEqual([(block.start_ms, block.end_ms, block.text) for block in out], [(0, 5000, '你好。世界。'), (5000, 10000, '再见。')])

    def test_cleaned_empty_slice_never_enters_summary_translation_or_polish(self):
        blocks = _slices()
        blocks[0].text = ''
        blocks[0].skip_tts = True
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('[{"slice_id":1,"sentences":["有效译文。"]}]'), _openai_resp('1\t润色后的有效译文。')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, _cfg())
        summary_input = fake.calls[0]['json']['messages'][-1]['content']
        translate_input = fake.calls[1]['json']['messages'][-1]['content']
        polish_input = fake.calls[2]['json']['messages'][-1]['content']
        self.assertNotIn('原转写一', summary_input)
        self.assertIn('原转写二', summary_input)
        self.assertNotIn('"slice_id": 0', translate_input)
        self.assertNotIn('0\t', polish_input)
        self.assertEqual([block.skip_tts for block in out], [True, False])
        self.assertEqual(out[1].text, '润色后的有效译文。')
        self.assertEqual(stats['untranslated'], 0)

    def test_all_cleaned_empty_slices_skip_every_llm_request(self):
        blocks = _slices()
        for block in blocks:
            block.text = ''
            block.skip_tts = True
        fake = FakeHttpx([])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, _cfg())
        self.assertEqual(fake.calls, [])
        self.assertTrue(all((block.skip_tts for block in out)))
        self.assertEqual(stats['translated'], 0)

    def test_existing_subtitles_do_nothing_when_optimization_is_disabled(self):
        blocks = _blocks('已有中文字幕。')
        fake = FakeHttpx([])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg(optimize_enabled=False))
        self.assertFalse(stats['enabled'])
        self.assertEqual(blocks[0].text, '已有中文字幕。')
        self.assertEqual(fake.calls, [])

    def test_not_configured_returns_original(self):
        blocks = _slices()
        fake = FakeHttpx([])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, {})
        self.assertFalse(stats['enabled'])
        self.assertEqual([b.text for b in out], ['原转写一', '原转写二'])
        self.assertEqual(fake.calls, [])

    def test_phase1_failure_still_attempts_translation(self):
        blocks = _slices()
        fake = FakeHttpx([_FakeResponse(500, {}, 'boom')])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                process_asr(blocks, cfg)
        self.assertEqual(len(fake.calls), 2)
