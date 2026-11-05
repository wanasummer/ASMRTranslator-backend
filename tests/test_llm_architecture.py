from modules.storage import address, read_text
"""LLM extension boundaries, persistence and immutable subtitle handoff."""
import asyncio
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from modules.error import LLMError
from modules.llm import LLMConfig, LLMRuntime
from modules.llm.factory import CAPABILITIES, create_capabilities, create_capability
from modules.llm.capabilities.capSummary import SummaryCapability
from modules.subtitle_processing import SubtitleStageInput
from modules.subtitle_processing.stage import LLMSubtitleStage
from modules.vtt_parser import VTTBlock
from tests.llm_support import FakeHttpx, FakeResponse


class LLMArchitectureTests(unittest.TestCase):
    def test_capabilities_share_injected_runtime_and_registry_is_extensible(self):
        runtime = LLMRuntime(LLMConfig())
        capabilities = create_capabilities(LLMConfig(), runtime=runtime)
        for capability in (capabilities.summary, capabilities.translation, capabilities.polish):
            self.assertIs(capability.runtime, runtime)
        with patch.dict(CAPABILITIES, {'custom': SummaryCapability}):
            self.assertIsInstance(create_capability('custom', runtime, LLMConfig()), SummaryCapability)
        with self.assertRaises(ValueError):
            create_capability('missing', runtime, LLMConfig())

    def test_summary_can_run_with_only_injected_text_generation_port(self):
        class Generator:
            cache = None
            complete = AsyncMock(return_value='摘要')
        generator = Generator()
        result = asyncio.run(SummaryCapability(generator, LLMConfig()).summarize([VTTBlock(7, 0, 1, '原文')]))
        self.assertEqual(result, '摘要')
        self.assertIn('原文', generator.complete.call_args.args[1])

    def test_stage_returns_new_blocks_and_preserves_input_metadata(self):
        config = LLMConfig(provider='openai', api_key='secret', model='test')
        capabilities = create_capabilities(config)
        capabilities.summary.summarize = AsyncMock(return_value='摘要')
        capabilities.polish.polish = AsyncMock(return_value={7: '新文本'})
        original = VTTBlock(7, 123, 987, '旧文本', speaker='A', source_text='日语')
        result = asyncio.run(LLMSubtitleStage(config, capabilities).run_async(SubtitleStageInput((original,))))
        self.assertEqual(original.text, '旧文本')
        self.assertIsNot(result.blocks[0], original)
        self.assertEqual((result.blocks[0].index, result.blocks[0].start_ms, result.blocks[0].end_ms,
                          result.blocks[0].speaker, result.blocks[0].source_text), (7, 123, 987, 'A', '日语'))
        self.assertEqual(result.blocks[0].text, '新文本')

    def test_raw_response_saved_before_parse_failure_and_secrets_redacted(self):
        secret = 'api-secret-value'
        config = LLMConfig(provider='openai', api_key=secret, model='test', max_retries=0)
        response = FakeResponse(200, {'unrecognized': '原始响应 ' + secret})
        with tempfile.TemporaryDirectory() as directory:
            runtime = LLMRuntime(config, directory)
            with patch('modules.llm.runtime.httpx', FakeHttpx([response])):
                with self.assertRaises(LLMError):
                    asyncio.run(runtime.complete('提示词', '原文'))
            store = address(Path(directory) / 'storage').store
            files = [Path(directory) / key for key in store.keys('', 'llm_results/requests/')]
            self.assertEqual(len(files), 2)
            for path in files:
                text = read_text(path)
                self.assertNotIn(secret, text)
                json.loads(text)
            raw = next(path for path in files if path.name.endswith('.response.json'))
            self.assertIn('原始响应', read_text(raw))

    def test_translation_cache_includes_summary_context(self):
        with tempfile.TemporaryDirectory() as directory:
            capability = create_capabilities(LLMConfig(), directory).translation
            capability._translate_split_once = AsyncMock(return_value={7: ['译文']})
            blocks = [VTTBlock(7, 0, 1000, '原文')]
            for summary in ['场景A', '场景A', '场景B']:
                self.assertEqual(asyncio.run(capability.translate(blocks, summary)), {7: ['译文']})
            self.assertEqual(capability._translate_split_once.await_count, 2)

    def test_llm_package_does_not_import_downstream_domains(self):
        directory = Path(__file__).resolve().parents[1] / 'modules' / 'llm'
        forbidden = {'subtitle_processing', 'vtt_parser', 'pipeline', 'detect', 'asr', 'tts_engine', 'arrangement'}
        for path in directory.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(forbidden.intersection(node.module.split('.')), str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(forbidden.intersection(alias.name.split('.')), str(path))

    def test_invalid_configuration_is_rejected_at_boundary(self):
        for settings in ({'batch_size': 0}, {'timeout': -1}, {'max_tokens': 0}, {'max_retries': -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                LLMConfig(**settings)


if __name__ == '__main__':
    unittest.main()
