import asyncio
import json
import os
import tempfile
import unittest
import httpx
from unittest import mock

from modules.asr.models import ASRResult, ASRSegment, AudioChunk
from modules.asr_engine import _transcribe_chunks
from modules.error import ASRError, LLMError, PipelineError, TTSError
from modules.llm_enhancer import LLMClient, LLMConfig
from modules.task_cache import JsonResultCache
from modules.tts_engine import generate_tts
from modules.vtt_parser import VTTBlock


class SliceResumeTest(unittest.TestCase):
    def test_asr_network_retry_releases_slot_for_later_slice(self):
        class Adapter:
            name = "fake-asr"
            calls = []
            async def transcribe(self, path):
                self.calls.append(path)
                if path == "first" and self.calls.count(path) == 1:
                    raise httpx.ConnectError("temporary network failure")
                return ASRResult("文", 1, "ja", [ASRSegment("文", 0, 1)])
        adapter = Adapter()
        chunks = [AudioChunk("0", 0, 1, "first"), AudioChunk("1", 1, 2, "later")]
        async def release_slot(*args):
            await original_sleep(0)
        original_sleep = asyncio.sleep
        with mock.patch("modules.asr_engine.asyncio.sleep", side_effect=release_slot):
            result = asyncio.run(_transcribe_chunks(chunks, adapter, 1, lambda *args: None))
        self.assertEqual(adapter.calls, ["first", "later", "first"])
        self.assertEqual(len(result), 2)

    def test_content_rejection_is_distinct_from_transient_failure(self):
        for error in (ASRError("content_policy_violation"), TTSError("content_filter")):
            self.assertFalse(error.descriptor.retryable)
            self.assertTrue(error.descriptor.code.endswith("CONTENT_REJECTED"))

    def test_asr_attempts_every_slice_and_only_retries_failed_slice(self):
        class Adapter:
            name = "fake-asr"
            calls = []
            fail = True
            async def transcribe(self, path):
                self.calls.append(path)
                if path.endswith("1.wav") and self.fail:
                    raise ASRError("rejected")
                return ASRResult("你好", 1, "ja", [ASRSegment("你好", 0, 1)])
        with tempfile.TemporaryDirectory() as directory:
            chunks = []
            for i in range(3):
                path = os.path.join(directory, f"{i}.wav")
                with open(path, "wb") as target:
                    target.write(bytes([i]))
                chunks.append(AudioChunk(str(i), i, i + 1, path))
            cache = JsonResultCache(os.path.join(directory, "results"))
            adapter = Adapter()
            progress = []
            with self.assertRaises(PipelineError):
                asyncio.run(_transcribe_chunks(chunks, adapter, 1,
                    lambda done, total: progress.append(done), cache, {"language": "ja"}))
            self.assertEqual(adapter.calls, [c.path for c in chunks])
            self.assertEqual(progress, [1, 2, 3])
            adapter.fail = False
            adapter.calls.clear()
            pairs = asyncio.run(_transcribe_chunks(chunks, adapter, 1,
                lambda *args: None, cache, {"language": "ja"}))
            self.assertEqual(adapter.calls, [chunks[1].path])
            self.assertEqual(len(pairs), 3)
            adapter.calls.clear()
            asyncio.run(_transcribe_chunks(chunks, adapter, 1,
                lambda *args: None, cache, {"language": "zh"}))
            self.assertEqual(len(adapter.calls), 3)

    def test_tts_partial_output_is_not_cached_and_later_slices_run(self):
        class Adapter:
            name = "fake-tts"
            output_extension = ".mp3"
            calls = []
            fail = True
            def get_voice_for_speaker(self, speaker): return "voice"
            async def synthesize(self, text, voice, path, **kwargs):
                self.calls.append(text)
                with open(path, "wb") as target:
                    target.write(b"audio" * 100)
                if text == "bad" and self.fail:
                    raise RuntimeError("rejected after partial audio")
        adapter = Adapter()
        segments = [{"index": i, "text": text} for i, text in enumerate(["ok", "bad", "later"])]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                "modules.tts_engine._get_adapter", return_value=adapter):
            with self.assertRaises(PipelineError):
                generate_tts(segments, directory, max_concurrent=1)
            self.assertEqual(adapter.calls, ["ok", "bad", "later"])
            self.assertFalse(os.path.exists(os.path.join(directory, "tts_0001.mp3")))
            adapter.fail = False
            adapter.calls.clear()
            generate_tts(segments, directory, max_concurrent=1)
            self.assertEqual(adapter.calls, ["bad"])
            adapter.calls.clear()
            segments[0]["text"] = "changed"
            generate_tts(segments, directory, max_concurrent=1)
            self.assertEqual(adapter.calls, ["changed"])

    def test_llm_rejected_batch_does_not_stop_later_batch_and_resume_reuses_rows(self):
        blocks = [VTTBlock(index=i, start_ms=i * 1000, end_ms=(i + 1) * 1000,
                           text=f"原文{i}") for i in range(3)]
        cfg = LLMConfig(provider="openai", model="test", batch_size=1, max_retries=0)
        calls = []
        fail = True
        async def respond(system, user, max_tokens=None):
            calls.append(user)
            if "原文1" in user and fail:
                raise LLMError("content_policy_violation")
            return json.dumps([{"slice_id": i, "sentences": [f"翻译{i}"]}
                               for i in range(3) if f"原文{i}" in user])
        with tempfile.TemporaryDirectory() as directory:
            client = LLMClient(cfg, directory)
            with mock.patch.object(client, "_post", side_effect=respond):
                with self.assertRaises(PipelineError):
                    asyncio.run(client.translate_split(blocks, "摘要"))
                self.assertEqual(len(calls), 3)
                fail = False
                calls.clear()
                result = asyncio.run(client.translate_split(blocks, "摘要"))
                self.assertEqual(len(calls), 1)
                self.assertIn("原文1", calls[0])
                self.assertEqual(set(result), {0, 1, 2})

    def test_llm_request_cache_survives_new_client(self):
        cfg = LLMConfig(provider="openai", model="test")
        with tempfile.TemporaryDirectory() as directory:
            client = LLMClient(cfg, directory)
            with mock.patch.object(client, "_post_uncached", new=mock.AsyncMock(return_value="结果")) as post:
                self.assertEqual(asyncio.run(client._post("system", "user")), "结果")
                post.assert_awaited_once()
            restored = LLMClient(cfg, directory)
            with mock.patch.object(restored, "_post_uncached", new=mock.AsyncMock()) as post:
                self.assertEqual(asyncio.run(restored._post("system", "user")), "结果")
                post.assert_not_awaited()

    def test_rejected_llm_slice_is_not_retried_repeatedly(self):
        cfg = LLMConfig(provider="openai", model="test", max_retries=3)
        blocks = [VTTBlock(0, 0, 1000, "拒绝片段")]
        client = LLMClient(cfg)
        with mock.patch.object(client, "_translate_split_once", new=mock.AsyncMock(return_value={})), \
             mock.patch.object(client, "_translate_single", new=mock.AsyncMock(
                 side_effect=LLMError("content_policy_violation"))) as single:
            with self.assertRaises(PipelineError) as caught:
                asyncio.run(client.translate_split(blocks, "摘要"))
            self.assertEqual(caught.exception.descriptor.code, "LLM_CONTENT_REJECTED")
            single.assert_awaited_once()

    def test_invalid_llm_rows_are_requested_again_on_resume(self):
        cfg = LLMConfig(provider="openai", model="test", max_retries=0)
        blocks = [VTTBlock(0, 0, 1000, "原文")]
        with tempfile.TemporaryDirectory() as directory:
            client = LLMClient(cfg, directory)
            with mock.patch.object(client, "_post_uncached", new=mock.AsyncMock(return_value="malformed")):
                with self.assertRaises(PipelineError):
                    asyncio.run(client.translate_split(blocks, "摘要"))
            restored = LLMClient(cfg, directory)
            with mock.patch.object(restored, "_post_uncached", new=mock.AsyncMock(
                    return_value='[{"slice_id":0,"sentences":["你好"]}]')) as post:
                self.assertEqual(asyncio.run(restored.translate_split(blocks, "摘要")), {0: ["你好"]})
                post.assert_awaited_once()
