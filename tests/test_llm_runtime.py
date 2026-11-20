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

class EndpointTest(unittest.TestCase):

    def test_openai_endpoint_variants(self):
        for base, expected in [('https://api.openai.com/v1', 'https://api.openai.com/v1/chat/completions'), ('https://api.openai.com', 'https://api.openai.com/chat/completions'), ('https://api.openai.com/v1/', 'https://api.openai.com/v1/chat/completions'), ('https://x.com/v1/chat/completions', 'https://x.com/v1/chat/completions')]:
            c = create_capabilities(LLMConfig(provider='openai', base_url=base))
            self.assertEqual(c.runtime.provider.endpoint(), expected, base)

    def test_openai_default_base(self):
        c = create_capabilities(LLMConfig(provider='openai', api_key='k', model='m'))
        self.assertEqual(c.runtime.provider.endpoint(), 'https://api.openai.com/v1/chat/completions')

    def test_anthropic_endpoint_variants(self):
        for base, expected in [('https://api.anthropic.com', 'https://api.anthropic.com/v1/messages'), ('https://api.anthropic.com/v1', 'https://api.anthropic.com/v1/messages'), ('https://api.anthropic.com/v1/messages', 'https://api.anthropic.com/v1/messages')]:
            c = create_capabilities(LLMConfig(provider='anthropic', base_url=base))
            self.assertEqual(c.runtime.provider.endpoint(), expected, base)

    def test_anthropic_default_base(self):
        c = create_capabilities(LLMConfig(provider='anthropic', api_key='k', model='m'))
        self.assertEqual(c.runtime.provider.endpoint(), 'https://api.anthropic.com/v1/messages')

class RequestFormatTest(unittest.TestCase):

    def test_official_deepseek_openai_disables_thinking_for_every_request(self):
        for base in ('https://api.deepseek.com', 'https://api.deepseek.com/v1', 'https://api.deepseek.com/chat/completions'):
            with self.subTest(base=base):
                fake = FakeHttpx([_openai_resp('好'), _openai_resp('摘要'), _openai_resp('0\t译文。'), _openai_resp('0.1\t译文。'), _openai_resp('译文。')])
                client = create_capabilities(LLMConfig(**_cfg(base_url=base, model='deepseek-flash')))
                with mock.patch.object(llm_runtime, 'httpx', fake):
                    asyncio.run(client.runtime.check_connection())
                    asyncio.run(client.summary.summarize(_blocks('原句')))
                    asyncio.run(client.polish.polish(_blocks('原句'), '摘要'))
                    self.assertEqual(asyncio.run(client.translation.translate(_blocks('原句'), '摘要')), {0: ['译文。']})
                    self.assertEqual(asyncio.run(client.translation._translate_single(_blocks('原句')[0], '摘要')), ['译文。'])
                self.assertEqual(len(fake.calls), 5)
                for call in fake.calls:
                    self.assertEqual(call['json']['thinking'], {'type': 'disabled'})

    def test_other_openai_endpoints_do_not_receive_deepseek_extension(self):
        for base in ('https://api.openai.com/v1', 'https://proxy.example/v1', 'https://api.deepseek.com.example/v1', 'https://api.deepseek.com@proxy.example/v1'):
            with self.subTest(base=base):
                client = create_capabilities(LLMConfig(**_cfg(base_url=base, model='deepseek-flash')))
                self.assertNotIn('thinking', client.runtime.provider.payload('system', 'user'))

    def test_openai_request_format(self):
        blocks = _blocks('原句一', '原句二')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t你好，世界。\n1\t[whispering] 小声一点。')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg())
        self.assertEqual(stats['rewritten'], 2)
        self.assertEqual(blocks[0].text, '你好，世界。')
        self.assertEqual(blocks[1].text, '[whispering] 小声一点。')
        p1, p2 = (fake.calls[0], fake.calls[1])
        self.assertEqual(p1['url'], 'https://api.openai.com/v1/chat/completions')
        self.assertEqual(p1['headers']['Authorization'], 'Bearer sk-test')
        self.assertEqual(p1['json']['model'], 'gpt-test')
        self.assertEqual(p1['json']['messages'][0]['role'], 'system')
        self.assertEqual(p1['json']['messages'][1]['role'], 'user')
        self.assertIn('序号', p1['json']['messages'][0]['content'])
        self.assertIn('temperature', p1['json'])

    def test_gpt_5_6_luna_uses_max_completion_tokens(self):
        client = create_capabilities(LLMConfig(provider='openai', api_key='sk-test', model='gpt-5.6-luna', max_tokens=2048))
        payload = client.runtime.provider.payload('system', 'user')
        self.assertEqual(payload['max_completion_tokens'], 2048)
        self.assertNotIn('max_tokens', payload)

    def test_anthropic_request_format(self):
        blocks = _blocks('原句一')
        fake = FakeHttpx([_anthropic_resp('摘要'), _anthropic_resp('0\t你好，世界。')])
        cfg = _cfg(provider='anthropic', base_url='https://api.anthropic.com')
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, _ = process_subtitles(blocks, cfg)
        p1, p2 = (fake.calls[0], fake.calls[1])
        self.assertEqual(p1['url'], 'https://api.anthropic.com/v1/messages')
        self.assertEqual(p1['headers']['x-api-key'], 'sk-test')
        self.assertEqual(p1['headers']['anthropic-version'], '2023-06-01')
        self.assertEqual(p1['json']['max_tokens'], 1024)
        self.assertIn('system', p1['json'])
        self.assertEqual(p1['json']['thinking'], {'type': 'disabled'})
        self.assertEqual(p1['json']['messages'], [{'role': 'user', 'content': p1['json']['messages'][0]['content']}])
        self.assertEqual(blocks[0].text, '你好，世界。')

    def test_response_parse_failure_raises(self):
        with self.assertRaises(LLMError):
            provider_parsing.parse_response('openai', {'no': 'choices'})
        with self.assertRaises(LLMError):
            provider_parsing.parse_response('anthropic', {'content': []})

class ResponseParseRobustnessTest(unittest.TestCase):
    """Real-world edge cases from DeepSeek's Anthropic-compatible endpoint."""

    def test_thinking_block_before_text(self):
        resp = {'content': [{'type': 'thinking', 'thinking': '推理过程'}, {'type': 'text', 'text': '正式摘要'}]}
        self.assertEqual(provider_parsing.parse_response('anthropic', resp), '正式摘要')

    def test_content_as_plain_string(self):
        self.assertEqual(provider_parsing.parse_response('anthropic', {'content': '直接文本'}), '直接文本')

    def test_openai_empty_content_reports_token_limit(self):
        response = {'choices': [{'message': {'role': 'assistant'}, 'finish_reason': 'length'}]}
        with self.assertRaisesRegex(LLMError, '输出 token 上限可能过低'):
            provider_parsing.parse_response('openai', response)

    @staticmethod
    def _gemini_cfg(**overrides):
        return _cfg(model='gemini-3.1-flash-lite', base_url='https://generativelanguage.googleapis.com/v1beta/openai/', **overrides)

    def test_gemini_payload_keeps_model_defaults(self):
        client = create_capabilities(LLMConfig(**self._gemini_cfg()))
        self.assertNotIn('reasoning_effort', client.runtime.provider.payload('system', 'user'))

    def test_empty_openai_text_is_retried_and_recovered(self):
        empty = _FakeResponse(200, {'choices': [{'message': {'role': 'assistant'}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 0}})
        fake = FakeHttpx([empty, _openai_resp('摘要')])
        client = create_capabilities(LLMConfig(**self._gemini_cfg(max_retries=1)))
        log = io.StringIO()
        with mock.patch.object(llm_runtime, 'httpx', fake), mock.patch.object(llm_runtime.asyncio, 'sleep', new_callable=mock.AsyncMock), contextlib.redirect_stdout(log):
            result = asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(result, '摘要')
        self.assertEqual(len(fake.calls), 2)
        self.assertIn('[llm-empty-response]', log.getvalue())
        self.assertNotIn('user', log.getvalue())

    def test_persistent_empty_openai_text_reports_metadata(self):
        empty = _FakeResponse(200, {'choices': [{'message': {'role': 'assistant'}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 0}})
        fake = FakeHttpx([empty, empty])
        client = create_capabilities(LLMConfig(**self._gemini_cfg(max_retries=1)))
        log = io.StringIO()
        with mock.patch.object(llm_runtime, 'httpx', fake), mock.patch.object(llm_runtime.asyncio, 'sleep', new_callable=mock.AsyncMock), contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(LLMError, '大模型未返回文本'):
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("completion_tokens': 0", log.getvalue())

    def test_other_openai_empty_response_is_not_retried(self):
        empty = _FakeResponse(200, {'choices': [{'message': {'role': 'assistant'}}]})
        fake = FakeHttpx([empty, _openai_resp('later')])
        client = create_capabilities(LLMConfig(**_cfg(max_retries=2)))
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaisesRegex(LLMError, '大模型未返回文本'):
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)

    def test_gemini_429_retries_then_recovers(self):
        limited = _FakeResponse(429, {'error': {'code': 429, 'message': 'rate limit'}})
        limited.headers = {'retry-after': '12'}
        fake = FakeHttpx([limited, limited, _openai_resp('摘要')])
        client = create_capabilities(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_runtime, 'httpx', fake), mock.patch.object(llm_runtime.asyncio, 'sleep', new_callable=mock.AsyncMock) as sleep, mock.patch.object(llm_runtime.random, 'uniform', return_value=0):
            self.assertEqual(asyncio.run(client.runtime.complete('system', 'user')), '摘要')
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [12, 12])

    def test_gemini_503_exhaustion_is_retryable_for_manual_resume(self):
        busy = _FakeResponse(503, {'error': {'code': 503, 'message': 'busy'}})
        fake = FakeHttpx([busy] * 6)
        client = create_capabilities(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_runtime, 'httpx', fake), mock.patch.object(llm_runtime.asyncio, 'sleep', new_callable=mock.AsyncMock), mock.patch.object(llm_runtime.random, 'uniform', return_value=0):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(caught.exception.descriptor.code, 'LLM_SERVICE_BUSY')
        self.assertTrue(caught.exception.descriptor.retryable)

    def test_gemini_429_exhaustion_preserves_rate_limit_error(self):
        limited = _FakeResponse(429, {'error': {'code': 429, 'message': 'rate limit'}})
        fake = FakeHttpx([limited] * 6)
        blocks = _slices()
        with mock.patch.object(llm_runtime, 'httpx', fake), mock.patch.object(llm_runtime.asyncio, 'sleep', new_callable=mock.AsyncMock), mock.patch.object(llm_runtime.random, 'uniform', return_value=0):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(create_capabilities(LLMConfig(**self._gemini_cfg())).runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(caught.exception.descriptor.code, 'LLM_GEMINI_RATE_LIMITED')
        self.assertTrue(caught.exception.descriptor.retryable)
        self.assertEqual([block.text for block in blocks], ['原转写一', '原转写二'])

    def test_other_openai_429_still_fails_without_retry(self):
        limited = _FakeResponse(429, {'error': {'code': 429, 'message': 'rate limit'}})
        fake = FakeHttpx([limited, _openai_resp('later')])
        client = create_capabilities(LLMConfig(**_cfg(max_retries=2)))
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(caught.exception.descriptor.code, 'LLM_RATE_LIMITED')

    def test_gemini_model_on_custom_openai_endpoint_keeps_original_429_behavior(self):
        limited = _FakeResponse(429, {'error': {'code': 429, 'message': 'rate limit'}})
        fake = FakeHttpx([limited, _openai_resp('later')])
        client = create_capabilities(LLMConfig(**_cfg(model='gemini-3.1-flash-lite', max_retries=2)))
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(LLMError):
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)

    def test_gemini_400_is_not_retried(self):
        bad_request = _FakeResponse(400, {'error': {'code': 400, 'message': 'invalid request'}})
        fake = FakeHttpx([bad_request, _openai_resp('later')])
        client = create_capabilities(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(LLMError):
                asyncio.run(client.runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)

    def test_connection_probe_allows_reasoning_tokens(self):
        fake = FakeHttpx([_openai_resp('好')])
        client = create_capabilities(LLMConfig(**_cfg(model='gemini-3.6-flash')))
        with mock.patch.object(llm_runtime, 'httpx', fake):
            self.assertEqual(asyncio.run(client.runtime.check_connection()), '好')
        self.assertEqual(fake.calls[0]['json']['max_tokens'], 256)

    def test_anthropic_openai_shape_fallback(self):
        resp = {'choices': [{'message': {'content': 'openai-style'}}]}
        self.assertEqual(provider_parsing.parse_response('anthropic', resp), 'openai-style')

    def test_error_field_raises(self):
        with self.assertRaises(LLMError):
            provider_parsing.parse_response('anthropic', {'error': {'message': 'boom'}})

    def test_no_text_block_raises_with_body(self):
        with self.assertRaises(LLMError) as ctx:
            provider_parsing.parse_response('anthropic', {'content': [{'type': 'thinking'}]})
        self.assertIn('没有可解析的文本', str(ctx.exception))

    def test_4xx_not_retried(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_FakeResponse(404, {}, 'model not found')])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                asyncio.run(create_capabilities(LLMConfig(**cfg)).runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)

    def test_parse_failure_not_retried(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_FakeResponse(200, {'content': [{'type': 'thinking'}]})])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                asyncio.run(create_capabilities(LLMConfig(**cfg)).runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)

    def test_5xx_still_retried(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_FakeResponse(500, {}, 'boom'), _FakeResponse(500, {}, 'boom')])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                asyncio.run(create_capabilities(LLMConfig(**cfg)).runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 2)

    def test_non_dict_body_raises_llm_error(self):
        for bad in ([], 'oops', None):
            with self.assertRaises(LLMError):
                provider_parsing.parse_response('anthropic', bad)
            with self.assertRaises(LLMError):
                provider_parsing.parse_response('openai', bad)

    def test_non_dict_body_not_retried(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_FakeResponse(200, [{'x': 1}])])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                asyncio.run(create_capabilities(LLMConfig(**cfg)).runtime.complete('system', 'user'))
        self.assertEqual(len(fake.calls), 1)
