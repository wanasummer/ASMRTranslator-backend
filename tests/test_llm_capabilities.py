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

class ParseLinesTest(unittest.TestCase):

    def test_tab_format_and_cleanup(self):
        raw = '```\n0\t你好。\n1\t[whispering] 嘘…\n```'
        out = polish_parsing._parse_lines(raw, {0, 1, 2})
        self.assertEqual(out, {0: '你好。', 1: '[whispering] 嘘…'})

    def test_fallback_period_format(self):
        out = polish_parsing._parse_lines('3. 这句话没tab。\n4： 这句用冒号。', {3, 4})
        self.assertEqual(out, {3: '这句话没tab。', 4: '这句用冒号。'})

    def test_dirty_rows_dropped(self):
        out = polish_parsing._parse_lines('0\t好\n垃圾行\nx\t坏\n1\t', {0, 1})
        self.assertEqual(out, {0: '好'})

    def test_unwanted_index_dropped(self):
        out = polish_parsing._parse_lines('9\t不属于这组', {0, 1})
        self.assertEqual(out, {})

class MaxTokensBudgetTest(unittest.TestCase):

    def test_phase2_uses_dynamic_budget(self):
        blocks = _blocks(*[f'原句{i}' for i in range(90)])
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t好。'), _openai_resp('80\t好。')])
        cfg = _cfg(batch_size=80, max_tokens=1024)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                blocks, _ = process_subtitles(blocks, cfg)
        p1, p2, p3 = (fake.calls[0], fake.calls[1], fake.calls[2])
        self.assertEqual(p1['json']['max_tokens'], 1024)
        self.assertEqual(p2['json']['max_tokens'], 512 + 80 * 96)
        self.assertEqual(p3['json']['max_tokens'], 512 + 10 * 96)

    def test_small_batch_never_below_default(self):
        blocks = _blocks('原句')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t好。')])
        cfg = _cfg(max_tokens=4096)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, _ = process_subtitles(blocks, cfg)
        self.assertEqual(fake.calls[1]['json']['max_tokens'], 4096)

class StripTagsTest(unittest.TestCase):

    def test_strip_emotion_tags(self):
        self.assertEqual(strip_emotion_tags('[whispering] 你好世界'), '你好世界')
        self.assertEqual(strip_emotion_tags('[angry][sad] 你好'), '你好')
        self.assertEqual(strip_emotion_tags('[pause] 嗯… 好的'), '嗯… 好的')
        self.assertEqual(strip_emotion_tags('没有标签的句子'), '没有标签的句子')
        self.assertEqual(strip_emotion_tags('[中文标签] 你好'), '[中文标签] 你好')

    def test_strip_bare_word_tag(self):
        self.assertEqual(strip_emotion_tags('whispering 你好世界'), '你好世界')
        self.assertEqual(strip_emotion_tags('whispering    这里是和耳大社。'), '这里是和耳大社。')
        self.assertEqual(strip_emotion_tags('softly 小声一点'), '小声一点')

    def test_strip_keeps_non_whitelist_english(self):
        self.assertEqual(strip_emotion_tags('OK 好的'), 'OK 好的')
        self.assertEqual(strip_emotion_tags('hey 你好'), 'hey 你好')

    def test_prepare_text_for_synthesis(self):

        class _Fish:
            name = 'fish-audio'

        class _Edge:
            name = 'edge-tts'

        class _MiniMax28:
            name = 'minimax-tts'
            _model = 'speech-2.8-hd'

        class _MiniMax26:
            name = 'minimax-tts'
            _model = 'speech-2.6-hd'
        text = '[whispering] 你好世界'
        self.assertEqual(prepare_text_for_synthesis(_Fish(), text), text)
        self.assertEqual(prepare_text_for_synthesis(_Edge(), text), '你好世界')
        mm28 = '[whispering] 哈哈哈(laughs) 你好(sighs)'
        self.assertEqual(prepare_text_for_synthesis(_MiniMax28(), mm28), '哈哈哈(laughs) 你好(sighs)')
        self.assertEqual(prepare_text_for_synthesis(_MiniMax26(), mm28), '哈哈哈 你好')

    def test_prepare_text_for_synthesis_strips_bare_tag(self):

        class _Fish:
            name = 'fish-audio'

        class _Edge:
            name = 'edge-tts'
        self.assertEqual(prepare_text_for_synthesis(_Fish(), 'whispering 你好'), 'whispering 你好')
        self.assertEqual(prepare_text_for_synthesis(_Edge(), 'whispering 你好'), '你好')

    def test_build_polish_system_by_backend(self):
        fish = tts_style.build_polish_system('摘要', 'fish-audio', '')
        self.assertIn('[whispering]', fish)
        self.assertNotIn('(laughs)', fish)
        mm28 = tts_style.build_polish_system('摘要', 'minimax-tts', 'speech-2.8-turbo')
        self.assertIn('(laughs)（笑声）', mm28)
        self.assertNotIn('[whispering]', mm28)
        mm26 = tts_style.build_polish_system('摘要', 'minimax-tts', 'speech-2.6-hd')
        self.assertIn('不支持任何情绪/语气标签', mm26)
        edge = tts_style.build_polish_system('摘要', 'edge-tts', '')
        self.assertIn('不支持任何情绪/语气标签', edge)
        self.assertNotIn('(laughs)', edge)

class NormalizeTagsTest(unittest.TestCase):

    def test_bare_tag_to_bracket(self):
        self.assertEqual(normalization._normalize_tags('whispering 文本'), '[whispering] 文本')
        self.assertEqual(normalization._normalize_tags('whispering\t这里是和耳大社。'), '[whispering] 这里是和耳大社。')
        self.assertEqual(normalization._normalize_tags('whispering    多空格。'), '[whispering] 多空格。')

    def test_already_bracketed_unchanged(self):
        self.assertEqual(normalization._normalize_tags('[whispering] 文本'), '[whispering] 文本')

    def test_non_whitelist_word_unchanged(self):
        self.assertEqual(normalization._normalize_tags('OK 好的'), 'OK 好的')
        self.assertEqual(normalization._normalize_tags('你好世界'), '你好世界')

    def test_bare_tag_normalized_in_writeback(self):
        blocks = _blocks('原句一')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\twhispering\t这里是和耳大社。')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, _ = process_subtitles(blocks, _cfg())
        self.assertEqual(blocks[0].text, '[whispering] 这里是和耳大社。')

class SanitizeSymbolsTest(unittest.TestCase):
    """LLM-optimized text must be stripped of symbol chars (♡, ★) that TTS
    reads out as noise; wave-dash variants are normalized to ASCII '~'."""

    def test_heart_symbol_removed(self):
        self.assertEqual(normalization._sanitize_symbols('看来你也变成男人了呢...♡'), '看来你也变成男人了呢...')

    def test_wave_dash_variants_normalized_to_ascii(self):
        self.assertEqual(normalization._sanitize_symbols('我都这么大了吗～'), '我都这么大了吗~')
        self.assertEqual(normalization._sanitize_symbols('我都这么大了吗〜'), '我都这么大了吗~')
        self.assertEqual(normalization._sanitize_symbols('我都这么大了吗∼'), '我都这么大了吗~')
        self.assertEqual(normalization._sanitize_symbols('都这么大了吗~'), '都这么大了吗~')

    def test_normal_punctuation_kept(self):
        text = '「喂」——她说："该走了"？对，就是这样……；真好啊！'
        self.assertEqual(normalization._sanitize_symbols(text), text)

    def test_emotion_tag_kept(self):
        self.assertEqual(normalization._sanitize_symbols('[teasing] 你一直在偷瞄吧？♡'), '[teasing] 你一直在偷瞄吧？')

    def test_whitespace_collapsed(self):
        self.assertEqual(normalization._sanitize_symbols('嘛 ♡  之后呢'), '嘛 之后呢')

    def test_symbols_only_becomes_empty(self):
        self.assertEqual(normalization._sanitize_symbols('♡♡'), '')

    def test_heart_removed_wave_dash_normalized_at_parse_time(self):
        raw = '0\t真是的♡\n1\t[affectionate] 当然...我也一样哦～\n'
        out = polish_parsing._parse_lines(raw, {0, 1})
        self.assertEqual(out, {0: '真是的', 1: '[affectionate] 当然...我也一样哦~'})

    def test_heart_removed_wave_dash_normalized_in_translate_split(self):
        raw = '0.1\t一起玩吧♡\n1.1\t再见～\n'
        out = translation_parsing._parse_split_lines(raw, {0, 1})
        self.assertEqual(out, {0: ['一起玩吧'], 1: ['再见~']})

    def test_heart_removed_wave_dash_normalized_end_to_end(self):
        blocks = _blocks('原句一')
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0\t一起玩吧♡～')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            blocks, stats = process_subtitles(blocks, _cfg())
        self.assertEqual(stats['rewritten'], 1)
        self.assertEqual(blocks[0].text, '一起玩吧~')

class ParseSplitLinesTest(unittest.TestCase):

    def test_dotted_indices(self):
        raw = '3.1\t第一句\n3.2\t第二句\n5.1\t别的\n'
        out = translation_parsing._parse_split_lines(raw, {3, 5})
        self.assertEqual(out, {3: ['第一句', '第二句'], 5: ['别的']})

    def test_bare_index_maps_to_single(self):
        out = translation_parsing._parse_split_lines('4\t单句\n', {4})
        self.assertEqual(out, {4: ['单句']})

    def test_dirty_and_unwanted_dropped(self):
        out = translation_parsing._parse_split_lines('3.1\t好\n垃圾\n9.1\t越界\n3.x\t坏\n3.1\t', {3})
        self.assertEqual(out, {3: ['好']})

    def test_code_fence_stripped(self):
        raw = '```\n0.1\t你好。\n0.2\t再见。\n```'
        out = translation_parsing._parse_split_lines(raw, {0})
        self.assertEqual(out, {0: ['你好。', '再见。']})

    def test_sentences_kept_in_order(self):
        raw = '2.3\t三\n2.1\t一\n2.2\t二\n'
        out = translation_parsing._parse_split_lines(raw, {2})
        self.assertEqual(out, {2: ['一', '二', '三']})

    def test_structured_translation_response(self):
        raw = '[{"slice_id":0,"sentences":["自然一点。","慢慢说。"]},{"slice_id":1,"sentences":["下一段。"]}]'
        out = translation_parsing._parse_translation_response(raw, {0, 1})
        self.assertEqual(out, {0: ['自然一点。', '慢慢说。'], 1: ['下一段。']})

    def test_structured_translation_rejects_duplicate_slice_id(self):
        raw = '[{"slice_id":0,"sentences":["本段。"]},{"slice_id":0,"sentences":["借来的下一段。"]}]'
        out = translation_parsing._parse_translation_response(raw, {0})
        self.assertEqual(out, {0: ['本段。']})

class TranslateSplitRetryTest(unittest.TestCase):
    """translate_split must retry slices the LLM omitted, not silently drop them."""

    def test_missing_slice_is_retried_and_recovered(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('0.1\t你好。\n0.2\t世界。'), _openai_resp('1.1\t再见。')])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            client = create_capabilities(LLMConfig(**cfg))
            per_slice = asyncio.run(client.translation.translate(blocks, '摘要'))
        self.assertEqual(per_slice, {0: ['你好。', '世界。'], 1: ['再见。']})
        self.assertEqual(len(fake.calls), 2)
        retry_text = fake.calls[1]['json']['messages'][-1]['content']
        self.assertIn('原转写二', retry_text)
        self.assertNotIn('原转写一', retry_text)

    def test_no_retry_when_every_slice_translated(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('0.1\t你好。\n1.1\t再见。')])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            client = create_capabilities(LLMConfig(**cfg))
            per_slice = asyncio.run(client.translation.translate(blocks, '摘要'))
        self.assertEqual(per_slice, {0: ['你好。'], 1: ['再见。']})
        self.assertEqual(len(fake.calls), 1)

    def test_suspicious_cross_slice_duplicate_is_verified_individually(self):
        blocks = [VTTBlock(index=0, start_ms=0, end_ms=5000, text='どうせ彼女いないなら昔みたいに遊ぼうよ'), VTTBlock(index=1, start_ms=5000, end_ms=10000, text='お医者さんごっこでもいいよ、せっかく帰ってきたんだし')]
        fake = FakeHttpx([_openai_resp('[{"slice_id":0,"sentences":["玩医生游戏也行啊，难得回来，不好好玩怎么行。"]},{"slice_id":1,"sentences":["玩医生游戏也行啊，难得回来一趟，好好玩玩吧。"]}]'), _openai_resp('反正你也没有女朋友，就像以前那样一起玩吧。'), _openai_resp('玩医生游戏也行啊，难得回来一趟，好好玩玩吧。')])
        with mock.patch.object(llm_runtime, 'httpx', fake):
            client = create_capabilities(LLMConfig(**_cfg()))
            per_slice = asyncio.run(client.translation.translate(blocks, '摘要'))
        self.assertEqual(per_slice, {0: ['反正你也没有女朋友，就像以前那样一起玩吧。'], 1: ['玩医生游戏也行啊，难得回来一趟，好好玩玩吧。']})
        self.assertEqual(len(fake.calls), 3)

    def test_exhausted_retries_fail_fast(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('0.1\t你好。'), _openai_resp('0.2\t还是漏了。')])
        cfg = _cfg(max_retries=1)
        buf = io.StringIO()
        with mock.patch.object(llm_runtime, 'httpx', fake), contextlib.redirect_stdout(buf):
            client = create_capabilities(LLMConfig(**cfg))
            with self.assertRaises(PipelineError):
                asyncio.run(client.translation.translate(blocks, '摘要'))
        self.assertEqual(len(fake.calls), 2)

    def test_untranslated_slice_fails_fast(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0.1\t你好。'), _openai_resp('0.2\t还是没翻出来。'), _openai_resp('0\t[whispering] 你好。')])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            with self.assertRaises(PipelineError):
                process_asr(blocks, cfg)
        self.assertEqual(len(fake.calls), 3)

    def test_end_to_end_recovers_last_slice(self):
        blocks = _slices() + [VTTBlock(index=2, start_ms=10000, end_ms=15000, text='一緒に気持ち良くなりますよね。')]
        fake = FakeHttpx([_openai_resp('摘要'), _openai_resp('0.1\t你好。\n1.1\t再见。'), _openai_resp('2.1\t一起变得更舒服吧。'), _openai_resp('0\t你好。\n1\t再见。\n2\t一起变得更舒服吧。')])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_runtime, 'httpx', fake):
            out, stats = process_asr(blocks, cfg)
        self.assertEqual(stats['translated'], 3)
        self.assertEqual(stats['untranslated'], 0)
        self.assertEqual([b.text for b in out], ['你好。', '再见。', '一起变得更舒服吧。'])
        self.assertEqual(len(fake.calls), 4)
