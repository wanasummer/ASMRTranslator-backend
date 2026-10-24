"""Contract tests for the synchronous common ASR service adapter."""

import unittest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, mock_open, patch

from modules.asr.common_asr_adapter import CommonASRAdapter
from modules.error import ASRError
from modules.vad.models import SpeechRegion


class CommonASRAdapterTest(unittest.TestCase):
    def test_restores_speech_only_after_validating_padded_response(self):
        from modules.vad import optimize_asmr_segments
        slices = optimize_asmr_segments([SpeechRegion(10, 15)])
        client = AsyncMock()
        response = Mock(is_success=True)
        response.json.return_value = {
            'contract_version': 1, 'stage': 'asr', 'language': 'ja',
            'blocks': [{'index': 0, 'start_ms': 9600, 'end_ms': 15400, 'text': '日本語'}],
        }
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch('modules.asr.common_asr_adapter.httpx.AsyncClient', return_value=context), \
             patch('builtins.open', mock_open(read_data=b'audio')):
            result = asyncio.run(CommonASRAdapter().transcribe_document('audio.wav', slices))
        sent = json.loads(client.post.call_args.kwargs['data']['segments'])
        self.assertEqual((sent[0]['start_ms'], sent[0]['end_ms']), (9600, 15400))
        self.assertEqual((result.blocks[0].start_ms, result.blocks[0].end_ms), (10000, 15000))
        self.assertEqual(response.json.return_value['blocks'][0]['start_ms'], 9600)

    def test_external_segments_trim_overlap(self):
        segments = CommonASRAdapter._external_segments([
            SpeechRegion(1.0, 3.0),
            SpeechRegion(2.5, 4.0),
        ])
        self.assertEqual(segments, [
            {"index": 0, "start_ms": 1000, "end_ms": 3000},
            {"index": 1, "start_ms": 3000, "end_ms": 4000},
        ])

    def test_parses_direct_downstream_contract(self):
        adapter = CommonASRAdapter(output_language="ja")
        requested = [{"index": 0, "start_ms": 1200, "end_ms": 4600}]
        result = adapter._parse_result({
            "contract_version": 1,
            "stage": "asr",
            "language": "ja",
            "source": "parakeet-asr",
            "timeline_source": "external_segments",
            "blocks": [{
                "index": 0,
                "start_ms": 1200,
                "end_ms": 4600,
                "text": "今日もお疲れさまでした。",
                "skip_tts": False,
            }],
        }, requested)
        self.assertEqual(result.language, "ja")
        self.assertEqual(result.source, "parakeet-asr")
        self.assertEqual(result.blocks[0].text, "今日もお疲れさまでした。")

    def test_rejects_changed_external_timeline(self):
        adapter = CommonASRAdapter(output_language="ja")
        requested = [{"index": 0, "start_ms": 1200, "end_ms": 4600}]
        with self.assertRaises(ASRError):
            adapter._parse_result({
                "contract_version": 1,
                "stage": "asr",
                "language": "ja",
                "blocks": [{
                    "index": 0,
                    "start_ms": 1000,
                    "end_ms": 4600,
                    "text": "text",
                }],
            }, requested)

    def test_loopback_client_ignores_process_proxy(self):
        from modules.vad import optimize_asmr_segments
        slices = optimize_asmr_segments([SpeechRegion(10, 15)])
        client = AsyncMock()
        response = Mock(is_success=True)
        response.json.return_value = {
            "contract_version": 1, "stage": "asr", "language": "ja",
            "blocks": [{"index": 0, "start_ms": 9600, "end_ms": 15400, "text": "日本語"}],
        }
        client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch("modules.asr.common_asr_adapter.httpx.AsyncClient", return_value=context) as factory, \
                patch("builtins.open", mock_open(read_data=b"audio")):
            asyncio.run(CommonASRAdapter("http://127.0.0.1:8001").transcribe_document("audio.wav", slices))
        self.assertFalse(factory.call_args.kwargs["trust_env"])
        transport = factory.call_args.kwargs["transport"]
        transport.policy.proxy_provider = lambda: {"http": "http://proxy.test:8080"}
        self.assertEqual(transport.policy.routes("http://127.0.0.1:8001"), [None])

    def test_explicit_no_speech_contract_returns_empty_document(self):
        adapter = CommonASRAdapter(output_language="ja")
        result = adapter._parse_result({
            "contract_version": 1,
            "stage": "asr",
            "language": "ja",
            "code": "NO_SPEECH_DETECTED",
            "blocks": [],
        }, [{"index": 0, "start_ms": 0, "end_ms": 1000}])

        self.assertEqual(result.language, "ja")
        self.assertEqual(result.blocks, [])


if __name__ == "__main__":
    unittest.main()
