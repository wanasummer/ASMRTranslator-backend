"""Contract tests for the ChickenRice whole-audio ASR adapter."""

import unittest

from modules.asr.chickenrice_adapter import ChickenRiceASRAdapter
from modules.error import ASRError
from modules.vad.models import SpeechRegion


class ChickenRiceAdapterTest(unittest.TestCase):
    def test_external_segments_trim_overlap_and_never_use_none(self):
        segments = ChickenRiceASRAdapter._external_segments([
            SpeechRegion(1.0, 3.0),
            SpeechRegion(2.5, 4.0),
        ])
        self.assertEqual(segments, [
            {"index": 0, "start_ms": 1000, "end_ms": 3000},
            {"index": 1, "start_ms": 3000, "end_ms": 4000},
        ])

    def test_parses_structured_blocks_and_language(self):
        adapter = ChickenRiceASRAdapter(output_language="zh")
        result = adapter._parse_result({
            "language": "zh",
            "source": "chickenrice-zh-translate",
            "timeline_source": "chickenrice_vad",
            "blocks": [{
                "index": 0, "start_ms": 1200, "end_ms": 4600,
                "text": "今天辛苦了。", "skip_tts": False,
            }],
        })
        self.assertEqual(result.language, "zh")
        self.assertEqual(result.blocks[0].text, "今天辛苦了。")

    def test_rejects_unexpected_response_language(self):
        adapter = ChickenRiceASRAdapter(output_language="zh")
        with self.assertRaises(ASRError):
            adapter._parse_result({"language": "ja", "blocks": []})

    def test_explicit_no_speech_contract_returns_empty_document(self):
        adapter = ChickenRiceASRAdapter(output_language="zh")
        result = adapter._parse_result({
            "language": "zh",
            "code": "NO_SPEECH_DETECTED",
        })

        self.assertEqual(result.language, "zh")
        self.assertEqual(result.blocks, [])


if __name__ == "__main__":
    unittest.main()
