"""Tests for modules.llm_enhancer (LLM subtitle enhancement).

Run from the project root:  python -m unittest tests.test_llm_enhancer
Fully offline — httpx is stubbed with a fake async client; no network is touched.
"""

import asyncio
import contextlib
import io
import json
import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules import llm_enhancer
from modules.error import LLMError, PipelineError
from modules.llm_enhancer import (
    LLMConfig,
    LLMClient,
    enhance_blocks,
    is_configured,
    strip_emotion_tags,
)
from modules.tts_engine import prepare_text_for_synthesis
from modules.vtt_parser import VTTBlock


class _FakeResponse:
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


class _FakeACM:
    """Async context manager standing in for httpx.AsyncClient."""

    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers, json):
        self.owner.calls.append({"url": url, "headers": headers, "json": json})
        if self.owner.responses:
            return self.owner.responses.pop(0)
        return _FakeResponse(500, {}, "no response queued")


class FakeHttpx:
    """Stand-in for the httpx module inside llm_enhancer (isolated via patch)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def AsyncClient(self, **kwargs):
        return _FakeACM(self)


def _blocks(*texts):
    return [VTTBlock(index=i, start_ms=i * 1000, end_ms=i * 1000 + 500,
                     text=t, speaker=("遥" if i % 2 == 0 else None))
            for i, t in enumerate(texts)]


def _openai_resp(content):
    return _FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def _anthropic_resp(text):
    return _FakeResponse(200, {"content": [{"text": text}]})


def _cfg(**overrides):
    base = {"provider": "openai", "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test", "model": "gpt-test", "max_retries": 0}
    base.update(overrides)
    return base


class EndpointTest(unittest.TestCase):
    def test_openai_endpoint_variants(self):
        for base, expected in [
            ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
            ("https://api.openai.com", "https://api.openai.com/chat/completions"),
            ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
            ("https://x.com/v1/chat/completions", "https://x.com/v1/chat/completions"),
        ]:
            c = LLMClient(LLMConfig(provider="openai", base_url=base))
            self.assertEqual(c._endpoint(), expected, base)

    def test_openai_default_base(self):
        c = LLMClient(LLMConfig(provider="openai", api_key="k", model="m"))
        self.assertEqual(c._endpoint(), "https://api.openai.com/v1/chat/completions")

    def test_anthropic_endpoint_variants(self):
        for base, expected in [
            ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
            ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
            ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"),
        ]:
            c = LLMClient(LLMConfig(provider="anthropic", base_url=base))
            self.assertEqual(c._endpoint(), expected, base)

    def test_anthropic_default_base(self):
        c = LLMClient(LLMConfig(provider="anthropic", api_key="k", model="m"))
        self.assertEqual(c._endpoint(), "https://api.anthropic.com/v1/messages")


class RequestFormatTest(unittest.TestCase):
    def test_official_deepseek_openai_disables_thinking_for_every_request(self):
        for base in ("https://api.deepseek.com", "https://api.deepseek.com/v1",
                     "https://api.deepseek.com/chat/completions"):
            with self.subTest(base=base):
                fake = FakeHttpx([_openai_resp("好"), _openai_resp("摘要"),
                                  _openai_resp("0\t译文。"),
                                  _openai_resp("0.1\t译文。"),
                                  _openai_resp("译文。")])
                client = LLMClient(LLMConfig(**_cfg(
                    base_url=base, model="deepseek-flash")))
                with mock.patch.object(llm_enhancer, "httpx", fake):
                    asyncio.run(client.check_connection())
                    asyncio.run(client.phase1_summarize(_blocks("原句")))
                    asyncio.run(client.phase2_optimize(_blocks("原句"), "摘要"))
                    self.assertEqual(
                        asyncio.run(client.translate_split(_blocks("原句"), "摘要")),
                        {0: ["译文。"]})
                    self.assertEqual(
                        asyncio.run(client._translate_single(_blocks("原句")[0], "摘要")),
                        ["译文。"])
                self.assertEqual(len(fake.calls), 5)
                for call in fake.calls:
                    self.assertEqual(call["json"]["thinking"], {"type": "disabled"})

    def test_other_openai_endpoints_do_not_receive_deepseek_extension(self):
        for base in ("https://api.openai.com/v1", "https://proxy.example/v1",
                     "https://api.deepseek.com.example/v1",
                     "https://api.deepseek.com@proxy.example/v1"):
            with self.subTest(base=base):
                client = LLMClient(LLMConfig(**_cfg(
                    base_url=base, model="deepseek-flash")))
                self.assertNotIn("thinking", client._payload("system", "user"))

    def test_openai_request_format(self):
        blocks = _blocks("原句一", "原句二")
        fake = FakeHttpx([_openai_resp("摘要"), _openai_resp("0\t你好，世界。\n1\t[whispering] 小声一点。")])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg())
        self.assertEqual(stats["rewritten"], 2)
        self.assertEqual(blocks[0].text, "你好，世界。")
        self.assertEqual(blocks[1].text, "[whispering] 小声一点。")

        p1, p2 = fake.calls[0], fake.calls[1]
        self.assertEqual(p1["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(p1["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(p1["json"]["model"], "gpt-test")
        self.assertEqual(p1["json"]["messages"][0]["role"], "system")
        self.assertEqual(p1["json"]["messages"][1]["role"], "user")
        self.assertIn("序号", p1["json"]["messages"][0]["content"])  # prompt is Chinese
        self.assertIn("temperature", p1["json"])

    def test_gpt_5_6_luna_uses_max_completion_tokens(self):
        client = LLMClient(LLMConfig(
            provider="openai",
            api_key="sk-test",
            model="gpt-5.6-luna",
            max_tokens=2048,
        ))

        payload = client._payload("system", "user")

        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", payload)

    def test_anthropic_request_format(self):
        blocks = _blocks("原句一")
        fake = FakeHttpx([_anthropic_resp("摘要"), _anthropic_resp("0\t你好，世界。")])
        cfg = _cfg(provider="anthropic", base_url="https://api.anthropic.com")
        with mock.patch.object(llm_enhancer, "httpx", fake):
            enhance_blocks(blocks, cfg)
        p1, p2 = fake.calls[0], fake.calls[1]
        self.assertEqual(p1["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(p1["headers"]["x-api-key"], "sk-test")
        self.assertEqual(p1["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(p1["json"]["max_tokens"], 1024)  # anthropic requires max_tokens
        self.assertIn("system", p1["json"])  # top-level, not in messages
        # DeepSeek's Anthropic-compatible endpoint defaults to thinking mode;
        # it must be explicitly disabled or a long reasoning block eats the
        # max_tokens budget and the text block never appears.
        self.assertEqual(p1["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(p1["json"]["messages"], [{"role": "user", "content": p1["json"]["messages"][0]["content"]}])
        self.assertEqual(blocks[0].text, "你好，世界。")

    def test_response_parse_failure_raises(self):
        with self.assertRaises(LLMError):
            LLMClient._parse_response("openai", {"no": "choices"})
        with self.assertRaises(LLMError):
            LLMClient._parse_response("anthropic", {"content": []})


class ParseLinesTest(unittest.TestCase):
    def test_tab_format_and_cleanup(self):
        raw = "```\n0\t你好。\n1\t[whispering] 嘘…\n```"
        out = llm_enhancer._parse_lines(raw, {0, 1, 2})
        self.assertEqual(out, {0: "你好。", 1: "[whispering] 嘘…"})

    def test_fallback_period_format(self):
        out = llm_enhancer._parse_lines("3. 这句话没tab。\n4： 这句用冒号。", {3, 4})
        self.assertEqual(out, {3: "这句话没tab。", 4: "这句用冒号。"})

    def test_dirty_rows_dropped(self):
        out = llm_enhancer._parse_lines("0\t好\n垃圾行\nx\t坏\n1\t", {0, 1})
        self.assertEqual(out, {0: "好"})

    def test_unwanted_index_dropped(self):
        out = llm_enhancer._parse_lines("9\t不属于这组", {0, 1})
        self.assertEqual(out, {})


class EnhanceTest(unittest.TestCase):
    def test_content_policy_error_is_logged_and_still_stops_conversion(self):
        from modules.error_reporter import build_error_report
        marker = "diagnostic-tail-marker"
        body = json.dumps({"error": {"message": "当前请求无权执行", "type": "content_policy_violation", "code": None},
                           "api_key": "echoed-secret", "extra": "x" * 700 + marker}, ensure_ascii=False)
        response = _FakeResponse(403, json.loads(body), body)
        response.headers = {"x-request-id": "request-123", "Authorization": "Bearer must-not-log"}
        fake = FakeHttpx([response])
        output = io.StringIO()
        blocks = _blocks("original")
        with mock.patch.object(llm_enhancer, "httpx", fake), contextlib.redirect_stdout(output):
            with self.assertRaises(LLMError) as caught:
                enhance_blocks(blocks, _cfg(base_url="https://name:url-password@example.test/v1?token=url-secret"))
        error = caught.exception
        self.assertEqual(error.descriptor.code, "LLM_CONTENT_REJECTED")
        self.assertIn("内容策略", error.descriptor.user_message)
        self.assertIn("当前请求无权执行", error.to_payload()["message"])
        self.assertNotIn("API Key", error.descriptor.user_message)
        self.assertEqual(blocks[0].text, "original")
        log = output.getvalue()
        record = json.loads(next(line.split("[llm-http-error] ", 1)[1] for line in log.splitlines() if "[llm-http-error]" in line))
        self.assertEqual(record["request_ids"], {"x-request-id": "request-123"})
        self.assertEqual(record["status"], 403)
        self.assertEqual(record["model"], "gpt-test")
        self.assertIn(marker, record["response_body"])
        for secret in ("echoed-secret", "url-password", "url-secret", "must-not-log", "sk-test"):
            self.assertNotIn(secret, log)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.log"
            path.write_text(log, encoding="utf-8")
            error.log_path = str(path)
            report = build_error_report(error, "", environ={})
        self.assertIn(marker, report)
        self.assertIn("request-123", report)

    def test_403_access_error_and_401_auth_error_are_distinct(self):
        for status, body, expected in [(403, '{"error":{"message":"access denied"}}', "LLM_ACCESS_DENIED"),
                                       (401, '{"error":{"message":"invalid key"}}', "LLM_AUTH_INVALID")]:
            with self.subTest(status=status):
                fake = FakeHttpx([_FakeResponse(status, json.loads(body), body)])
                with mock.patch.object(llm_enhancer, "httpx", fake):
                    with self.assertRaises(LLMError) as caught:
                        enhance_blocks(_blocks("original"), _cfg())
                self.assertEqual(caught.exception.descriptor.code, expected)

    def test_optimization_batch_preserves_policy_error(self):
        body = '{"error":{"type":"content_policy_violation","message":"denied"}}'
        fake = FakeHttpx([_openai_resp("summary"), _FakeResponse(403, json.loads(body), body)])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(LLMError) as caught:
                enhance_blocks(_blocks("original"), _cfg())
        self.assertEqual(caught.exception.descriptor.code, "LLM_CONTENT_REJECTED")

    def test_optimization_batch_preserves_gemini_rate_limit(self):
        body = '[{"error":{"code":429,"message":"Quota exceeded for this model"}}]'
        fake = FakeHttpx([_openai_resp("summary"), _FakeResponse(429, json.loads(body), body)])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(LLMError) as caught:
                enhance_blocks(_blocks("original"), _cfg())
        self.assertEqual(caught.exception.descriptor.code, "LLM_RATE_LIMITED")
        self.assertIn("Quota exceeded", caught.exception.detail)

    def test_large_policy_response_is_classified_before_log_truncation(self):
        data = {"padding": "x" * 33000, "error": {"code": "content_policy_violation", "message": "policy denied"}}
        response = _FakeResponse(400, data, json.dumps(data))
        fake = FakeHttpx([response])
        output = io.StringIO()
        with mock.patch.object(llm_enhancer, "httpx", fake), contextlib.redirect_stdout(output):
            with self.assertRaises(LLMError) as caught:
                enhance_blocks(_blocks("original"), _cfg())
        self.assertEqual(caught.exception.descriptor.code, "LLM_CONTENT_REJECTED")
        self.assertEqual(caught.exception.to_payload()["context"]["status"], 400)
        record = json.loads(next(line.split("[llm-http-error] ", 1)[1] for line in output.getvalue().splitlines() if "[llm-http-error]" in line))
        self.assertTrue(record["response_body_truncated"])
        self.assertLessEqual(len(record["response_body"]), 32768)
        self.assertEqual(record["error_code"], "content_policy_violation")

    def test_unparsable_line_fails_incomplete_contract(self):
        blocks = _blocks("原句一", "原句二", "原句三")
        fake = FakeHttpx([_openai_resp("摘要"),
                          _openai_resp("0\t优化一\n不是合法行\n2\t优化三")])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                enhance_blocks(blocks, _cfg())

    def test_missing_rows_use_targeted_batch_then_single_recovery(self):
        blocks = _blocks("原句零", "原句一", "原句二", "原句三")
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp("0\t优化零\n3\t优化三"),
            _openai_resp("1\t优化一"),
            _openai_resp("优化二"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg(max_retries=1))

        self.assertEqual(stats["rewritten"], 4)
        self.assertEqual([block.text for block in blocks],
                         ["优化零", "优化一", "优化二", "优化三"])
        self.assertEqual(fake.calls[2]["json"]["messages"][-1]["content"],
                         "1\t原句一\n2\t遥\t原句二")
        self.assertEqual(fake.calls[3]["json"]["messages"][-1]["content"],
                         "原句二")

    def test_single_missing_row_is_recovered_without_renumbering(self):
        blocks = _blocks("原句零", "原句一")
        fake = FakeHttpx([
            _openai_resp("摘要"), _openai_resp("0\t优化零"),
            _openai_resp("优化一"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            enhance_blocks(blocks, _cfg(max_retries=1))
        self.assertEqual([block.text for block in blocks], ["优化零", "优化一"])
        self.assertEqual(fake.calls[2]["json"]["messages"][-1]["content"],
                         "原句一")

    def test_invalid_whole_batch_is_recovered(self):
        blocks = _blocks("原句零", "原句一")
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _FakeResponse(200, {"content": []}),
            _openai_resp("0\t优化零\n1\t优化一"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg(max_retries=1))
        self.assertEqual(stats["rewritten"], 2)
        self.assertEqual([block.text for block in blocks], ["优化零", "优化一"])
        self.assertEqual(fake.calls[2]["json"]["messages"][-1]["content"],
                         "0\t遥\t原句零\n1\t原句一")

    def test_long_missing_run_is_retried_in_small_batches(self):
        blocks = _blocks(*[f"原句{i}" for i in range(20)])
        fake = FakeHttpx([
            _openai_resp("摘要"), _openai_resp("0\t优化0\n19\t优化19"),
            *[_openai_resp("\n".join(f"{i}\t优化{i}" for i in indices))
              for indices in (range(1, 9), range(9, 17), range(17, 19))],
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg(max_retries=1))
        self.assertEqual(stats["rewritten"], 20)
        self.assertEqual(len(fake.calls), 5)
        self.assertEqual([block.text for block in blocks],
                         [f"优化{i}" for i in range(20)])

    def test_unrecoverable_missing_row_keeps_all_original_text(self):
        blocks = _blocks("原句零", "原句一", "原句二")
        fake = FakeHttpx([
            _openai_resp("摘要"), _openai_resp("0\t优化零"),
            _openai_resp("无编号的批次回复"), _openai_resp("```"),
            _openai_resp("```"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError) as caught:
                enhance_blocks(blocks, _cfg(max_retries=1))
        self.assertEqual(caught.exception.descriptor.code, "LLM_OPTIMIZE_INCOMPLETE")
        self.assertIn("missing optimized rows after retries", caught.exception.detail)
        self.assertEqual([block.text for block in blocks],
                         ["原句零", "原句一", "原句二"])

    def test_batching_over_batch_size(self):
        blocks = _blocks(*[f"原句{i}" for i in range(90)])
        p1 = _openai_resp("摘要")
        lines = "\n".join(f"{i}\t优化{i}" for i in range(90))
        batch1 = "\n".join(f"{i}\t优化{i}" for i in range(0, 80))
        batch2 = "\n".join(f"{i}\t优化{i}" for i in range(80, 90))
        fake = FakeHttpx([p1, _openai_resp(batch1), _openai_resp(batch2)])
        cfg = _cfg(batch_size=80)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, cfg)
        self.assertEqual(stats["rewritten"], 90)
        self.assertEqual(len(fake.calls), 3)  # 1 summary + 2 phase-2 batches
        self.assertTrue(all(b.text == f"优化{i}" for i, b in enumerate(blocks)))

    def test_phase1_http_error_still_attempts_optimization(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_FakeResponse(500, {}, "boom")])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                enhance_blocks(blocks, cfg)
        self.assertEqual(blocks[0].text, "原句")
        self.assertEqual(len(fake.calls), 2)

    def test_phase2_batch_failure_fails_fast(self):
        blocks = _blocks(*["原句"] * 2)
        fake = FakeHttpx([_openai_resp("摘要"), _FakeResponse(500, {}, "boom")])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                enhance_blocks(blocks, cfg)
        self.assertTrue(all(b.text == "原句" for b in blocks))

    def test_not_configured_skips_network(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, {})  # no provider/key/model
        self.assertFalse(stats["enabled"])
        self.assertEqual(fake.calls, [])
        self.assertEqual(blocks[0].text, "原句")

    def test_is_configured(self):
        self.assertTrue(is_configured(LLMConfig(provider="openai", api_key="k", model="m")))
        self.assertTrue(is_configured(LLMConfig(provider="anthropic", api_key="k", model="m")))
        self.assertFalse(is_configured(LLMConfig(provider="openai", api_key="", model="m")))
        self.assertFalse(is_configured(LLMConfig(provider="gemini", api_key="k", model="m")))
        self.assertFalse(is_configured(LLMConfig()))


class ResponseParseRobustnessTest(unittest.TestCase):
    """Real-world edge cases from DeepSeek's Anthropic-compatible endpoint."""

    def test_thinking_block_before_text(self):
        resp = {"content": [
            {"type": "thinking", "thinking": "推理过程"},
            {"type": "text", "text": "正式摘要"},
        ]}
        self.assertEqual(LLMClient._parse_response("anthropic", resp), "正式摘要")

    def test_content_as_plain_string(self):
        self.assertEqual(LLMClient._parse_response("anthropic", {"content": "直接文本"}), "直接文本")

    def test_openai_empty_content_reports_token_limit(self):
        response = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}]}
        with self.assertRaisesRegex(LLMError, "输出 token 上限可能过低"):
            LLMClient._parse_response("openai", response)

    @staticmethod
    def _gemini_cfg(**overrides):
        return _cfg(model="gemini-3.1-flash-lite",
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    **overrides)

    def test_gemini_payload_keeps_model_defaults(self):
        client = LLMClient(LLMConfig(**self._gemini_cfg()))
        self.assertNotIn("reasoning_effort", client._payload("system", "user"))

    def test_empty_openai_text_is_retried_and_recovered(self):
        empty = _FakeResponse(200, {"choices": [{"message": {"role": "assistant"}}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 0}})
        fake = FakeHttpx([empty, _openai_resp("摘要")])
        client = LLMClient(LLMConfig(**self._gemini_cfg(max_retries=1)))
        log = io.StringIO()
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             mock.patch.object(llm_enhancer.asyncio, "sleep", new_callable=mock.AsyncMock), \
             contextlib.redirect_stdout(log):
            result = asyncio.run(client._post("system", "user"))
        self.assertEqual(result, "摘要")
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("[llm-empty-response]", log.getvalue())
        self.assertNotIn("user", log.getvalue())

    def test_persistent_empty_openai_text_reports_metadata(self):
        empty = _FakeResponse(200, {"choices": [{"message": {"role": "assistant"}}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 0}})
        fake = FakeHttpx([empty, empty])
        client = LLMClient(LLMConfig(**self._gemini_cfg(max_retries=1)))
        log = io.StringIO()
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             mock.patch.object(llm_enhancer.asyncio, "sleep", new_callable=mock.AsyncMock), \
             contextlib.redirect_stdout(log):
            with self.assertRaisesRegex(LLMError, "大模型未返回文本"):
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("completion_tokens': 0", log.getvalue())

    def test_other_openai_empty_response_is_not_retried(self):
        empty = _FakeResponse(200, {"choices": [{"message": {"role": "assistant"}}]})
        fake = FakeHttpx([empty, _openai_resp("later")])
        client = LLMClient(LLMConfig(**_cfg(max_retries=2)))
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaisesRegex(LLMError, "大模型未返回文本"):
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)

    def test_gemini_429_retries_then_recovers(self):
        limited = _FakeResponse(429, {"error": {"code": 429, "message": "rate limit"}})
        limited.headers = {"retry-after": "12"}
        fake = FakeHttpx([limited, limited, _openai_resp("摘要")])
        client = LLMClient(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             mock.patch.object(llm_enhancer.asyncio, "sleep", new_callable=mock.AsyncMock) as sleep, \
             mock.patch.object(llm_enhancer.random, "uniform", return_value=0):
            self.assertEqual(asyncio.run(client._post("system", "user")), "摘要")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [12, 12])

    def test_gemini_503_exhaustion_is_retryable_for_manual_resume(self):
        busy = _FakeResponse(503, {"error": {"code": 503, "message": "busy"}})
        fake = FakeHttpx([busy] * 6)
        client = LLMClient(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             mock.patch.object(llm_enhancer.asyncio, "sleep", new_callable=mock.AsyncMock), \
             mock.patch.object(llm_enhancer.random, "uniform", return_value=0):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(caught.exception.descriptor.code, "LLM_SERVICE_BUSY")
        self.assertTrue(caught.exception.descriptor.retryable)

    def test_gemini_429_exhaustion_preserves_rate_limit_error(self):
        limited = _FakeResponse(429, {"error": {"code": 429, "message": "rate limit"}})
        fake = FakeHttpx([limited] * 6)
        blocks = _slices()
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             mock.patch.object(llm_enhancer.asyncio, "sleep", new_callable=mock.AsyncMock), \
             mock.patch.object(llm_enhancer.random, "uniform", return_value=0):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(LLMClient(LLMConfig(**self._gemini_cfg()))._post("system", "user"))
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(caught.exception.descriptor.code, "LLM_GEMINI_RATE_LIMITED")
        self.assertTrue(caught.exception.descriptor.retryable)
        self.assertEqual([block.text for block in blocks], ["原转写一", "原转写二"])

    def test_other_openai_429_still_fails_without_retry(self):
        limited = _FakeResponse(429, {"error": {"code": 429, "message": "rate limit"}})
        fake = FakeHttpx([limited, _openai_resp("later")])
        client = LLMClient(LLMConfig(**_cfg(max_retries=2)))
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(LLMError) as caught:
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(caught.exception.descriptor.code, "LLM_RATE_LIMITED")

    def test_gemini_model_on_custom_openai_endpoint_keeps_original_429_behavior(self):
        limited = _FakeResponse(429, {"error": {"code": 429, "message": "rate limit"}})
        fake = FakeHttpx([limited, _openai_resp("later")])
        client = LLMClient(LLMConfig(**_cfg(model="gemini-3.1-flash-lite", max_retries=2)))
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(LLMError):
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)

    def test_gemini_400_is_not_retried(self):
        bad_request = _FakeResponse(400, {"error": {"code": 400, "message": "invalid request"}})
        fake = FakeHttpx([bad_request, _openai_resp("later")])
        client = LLMClient(LLMConfig(**self._gemini_cfg()))
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(LLMError):
                asyncio.run(client._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)

    def test_connection_probe_allows_reasoning_tokens(self):
        fake = FakeHttpx([_openai_resp("好")])
        client = LLMClient(LLMConfig(**_cfg(model="gemini-3.6-flash")))
        with mock.patch.object(llm_enhancer, "httpx", fake):
            self.assertEqual(asyncio.run(client.check_connection()), "好")
        self.assertEqual(fake.calls[0]["json"]["max_tokens"], 256)

    def test_anthropic_openai_shape_fallback(self):
        resp = {"choices": [{"message": {"content": "openai-style"}}]}
        self.assertEqual(LLMClient._parse_response("anthropic", resp), "openai-style")

    def test_error_field_raises(self):
        with self.assertRaises(LLMError):
            LLMClient._parse_response("anthropic", {"error": {"message": "boom"}})

    def test_no_text_block_raises_with_body(self):
        with self.assertRaises(LLMError) as ctx:
            LLMClient._parse_response("anthropic", {"content": [{"type": "thinking"}]})
        self.assertIn("没有可解析的文本", str(ctx.exception))

    def test_4xx_not_retried(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_FakeResponse(404, {}, "model not found")])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                asyncio.run(LLMClient(LLMConfig(**cfg))._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)  # 4xx is deterministic — single attempt

    def test_parse_failure_not_retried(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_FakeResponse(200, {"content": [{"type": "thinking"}]})])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                asyncio.run(LLMClient(LLMConfig(**cfg))._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)

    def test_5xx_still_retried(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_FakeResponse(500, {}, "boom"), _FakeResponse(500, {}, "boom")])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                asyncio.run(LLMClient(LLMConfig(**cfg))._post("system", "user"))
        self.assertEqual(len(fake.calls), 2)

    def test_non_dict_body_raises_llm_error(self):
        for bad in ([], "oops", None):
            with self.assertRaises(LLMError):
                LLMClient._parse_response("anthropic", bad)
            with self.assertRaises(LLMError):
                LLMClient._parse_response("openai", bad)

    def test_non_dict_body_not_retried(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_FakeResponse(200, [{"x": 1}])])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                asyncio.run(LLMClient(LLMConfig(**cfg))._post("system", "user"))
        self.assertEqual(len(fake.calls), 1)  # non-dict body is deterministic — no retry


class MaxTokensBudgetTest(unittest.TestCase):
    def test_phase2_uses_dynamic_budget(self):
        blocks = _blocks(*[f"原句{i}" for i in range(90)])
        fake = FakeHttpx([_openai_resp("摘要"),
                          _openai_resp("0\t好。"), _openai_resp("80\t好。")])
        cfg = _cfg(batch_size=80, max_tokens=1024)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                enhance_blocks(blocks, cfg)
        p1, p2, p3 = fake.calls[0], fake.calls[1], fake.calls[2]
        # phase1 keeps the configured default; phase2 is sized from the batch.
        self.assertEqual(p1["json"]["max_tokens"], 1024)
        self.assertEqual(p2["json"]["max_tokens"], 512 + 80 * 96)   # 80-row batch
        self.assertEqual(p3["json"]["max_tokens"], 512 + 10 * 96)   # 10-row batch

    def test_small_batch_never_below_default(self):
        blocks = _blocks("原句")
        fake = FakeHttpx([_openai_resp("摘要"), _openai_resp("0\t好。")])
        cfg = _cfg(max_tokens=4096)  # user raised the default above the formula
        with mock.patch.object(llm_enhancer, "httpx", fake):
            enhance_blocks(blocks, cfg)
        self.assertEqual(fake.calls[1]["json"]["max_tokens"], 4096)


class StripTagsTest(unittest.TestCase):
    def test_strip_emotion_tags(self):
        self.assertEqual(strip_emotion_tags("[whispering] 你好世界"), "你好世界")
        self.assertEqual(strip_emotion_tags("[angry][sad] 你好"), "你好")
        self.assertEqual(strip_emotion_tags("[pause] 嗯… 好的"), "嗯… 好的")
        self.assertEqual(strip_emotion_tags("没有标签的句子"), "没有标签的句子")
        # Non-ASCII (Chinese) brackets are NOT tags — preserved.
        self.assertEqual(strip_emotion_tags("[中文标签] 你好"), "[中文标签] 你好")

    def test_strip_bare_word_tag(self):
        # The model sometimes emits the tag without brackets ("whispering<TAB>文本").
        self.assertEqual(strip_emotion_tags("whispering 你好世界"), "你好世界")
        self.assertEqual(strip_emotion_tags("whispering    这里是和耳大社。"), "这里是和耳大社。")
        self.assertEqual(strip_emotion_tags("softly 小声一点"), "小声一点")

    def test_strip_keeps_non_whitelist_english(self):
        # Real English at the start of a line is NOT a tag — preserved.
        self.assertEqual(strip_emotion_tags("OK 好的"), "OK 好的")
        self.assertEqual(strip_emotion_tags("hey 你好"), "hey 你好")

    def test_prepare_text_for_synthesis(self):
        class _Fish:
            name = "fish-audio"

        class _Edge:
            name = "edge-tts"

        class _MiniMax28:
            name = "minimax-tts"
            _model = "speech-2.8-hd"

        class _MiniMax26:
            name = "minimax-tts"
            _model = "speech-2.6-hd"

        text = "[whispering] 你好世界"
        self.assertEqual(prepare_text_for_synthesis(_Fish(), text), text)
        self.assertEqual(prepare_text_for_synthesis(_Edge(), text), "你好世界")

        # minimax 2.8 保留 MiniMax 语气词标签，剥离方括号残留
        mm28 = "[whispering] 哈哈哈(laughs) 你好(sighs)"
        self.assertEqual(prepare_text_for_synthesis(_MiniMax28(), mm28),
                         "哈哈哈(laughs) 你好(sighs)")
        # minimax 2.6 不支持标签：方括号和语气词全部剥离
        self.assertEqual(prepare_text_for_synthesis(_MiniMax26(), mm28),
                         "哈哈哈 你好")

    def test_prepare_text_for_synthesis_strips_bare_tag(self):
        class _Fish:
            name = "fish-audio"

        class _Edge:
            name = "edge-tts"

        self.assertEqual(prepare_text_for_synthesis(_Fish(), "whispering 你好"), "whispering 你好")
        self.assertEqual(prepare_text_for_synthesis(_Edge(), "whispering 你好"), "你好")

    def test_build_phase2_system_by_backend(self):
        # fish → 方括号情绪标签；minimax 2.8 → 语气词标签；其他 → 无标签
        fish = llm_enhancer.build_phase2_system("摘要", "fish-audio", "")
        self.assertIn("[whispering]", fish)
        self.assertNotIn("(laughs)", fish)
        mm28 = llm_enhancer.build_phase2_system("摘要", "minimax-tts", "speech-2.8-turbo")
        self.assertIn("(laughs)（笑声）", mm28)
        self.assertNotIn("[whispering]", mm28)
        mm26 = llm_enhancer.build_phase2_system("摘要", "minimax-tts", "speech-2.6-hd")
        self.assertIn("不支持任何情绪/语气标签", mm26)
        edge = llm_enhancer.build_phase2_system("摘要", "edge-tts", "")
        self.assertIn("不支持任何情绪/语气标签", edge)
        self.assertNotIn("(laughs)", edge)


class NormalizeTagsTest(unittest.TestCase):
    def test_bare_tag_to_bracket(self):
        self.assertEqual(llm_enhancer._normalize_tags("whispering 文本"), "[whispering] 文本")
        # Tab between tag and text (the exact shape seen in real model output).
        self.assertEqual(llm_enhancer._normalize_tags("whispering\t这里是和耳大社。"), "[whispering] 这里是和耳大社。")
        self.assertEqual(llm_enhancer._normalize_tags("whispering    多空格。"), "[whispering] 多空格。")

    def test_already_bracketed_unchanged(self):
        self.assertEqual(llm_enhancer._normalize_tags("[whispering] 文本"), "[whispering] 文本")

    def test_non_whitelist_word_unchanged(self):
        self.assertEqual(llm_enhancer._normalize_tags("OK 好的"), "OK 好的")
        self.assertEqual(llm_enhancer._normalize_tags("你好世界"), "你好世界")

    def test_bare_tag_normalized_in_writeback(self):
        blocks = _blocks("原句一")
        fake = FakeHttpx([_openai_resp("摘要"), _openai_resp("0\twhispering\t这里是和耳大社。")])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            enhance_blocks(blocks, _cfg())
        self.assertEqual(blocks[0].text, "[whispering] 这里是和耳大社。")


def _slices():
    return [
        VTTBlock(index=0, start_ms=0, end_ms=5000, text="原转写一"),
        VTTBlock(index=1, start_ms=5000, end_ms=10000, text="原转写二"),
    ]


class SanitizeSymbolsTest(unittest.TestCase):
    """LLM-optimized text must be stripped of symbol chars (♡, ★) that TTS
    reads out as noise; wave-dash variants are normalized to ASCII '~'."""

    def test_heart_symbol_removed(self):
        self.assertEqual(llm_enhancer._sanitize_symbols("看来你也变成男人了呢...♡"),
                         "看来你也变成男人了呢...")

    def test_wave_dash_variants_normalized_to_ascii(self):
        self.assertEqual(llm_enhancer._sanitize_symbols("我都这么大了吗～"),
                         "我都这么大了吗~")
        self.assertEqual(llm_enhancer._sanitize_symbols("我都这么大了吗\u301c"),
                         "我都这么大了吗~")
        self.assertEqual(llm_enhancer._sanitize_symbols("我都这么大了吗\u223c"),
                         "我都这么大了吗~")
        self.assertEqual(llm_enhancer._sanitize_symbols("都这么大了吗~"),
                         "都这么大了吗~")

    def test_normal_punctuation_kept(self):
        text = "「喂」——她说：\"该走了\"？对，就是这样……；真好啊！"
        self.assertEqual(llm_enhancer._sanitize_symbols(text), text)

    def test_emotion_tag_kept(self):
        self.assertEqual(llm_enhancer._sanitize_symbols("[teasing] 你一直在偷瞄吧？♡"),
                         "[teasing] 你一直在偷瞄吧？")

    def test_whitespace_collapsed(self):
        self.assertEqual(llm_enhancer._sanitize_symbols("嘛 ♡  之后呢"),
                         "嘛 之后呢")

    def test_symbols_only_becomes_empty(self):
        self.assertEqual(llm_enhancer._sanitize_symbols("♡♡"), "")

    def test_heart_removed_wave_dash_normalized_at_parse_time(self):
        raw = "0\t真是的♡\n1\t[affectionate] 当然...我也一样哦～\n"
        out = llm_enhancer._parse_lines(raw, {0, 1})
        self.assertEqual(out, {0: "真是的", 1: "[affectionate] 当然...我也一样哦~"})

    def test_heart_removed_wave_dash_normalized_in_translate_split(self):
        raw = "0.1\t一起玩吧♡\n1.1\t再见～\n"
        out = llm_enhancer._parse_split_lines(raw, {0, 1})
        self.assertEqual(out, {0: ["一起玩吧"], 1: ["再见~"]})

    def test_heart_removed_wave_dash_normalized_end_to_end(self):
        blocks = _blocks("原句一")
        fake = FakeHttpx([_openai_resp("摘要"), _openai_resp("0\t一起玩吧♡～")])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg())
        self.assertEqual(stats["rewritten"], 1)
        self.assertEqual(blocks[0].text, "一起玩吧~")


class ParseSplitLinesTest(unittest.TestCase):
    def test_dotted_indices(self):
        raw = "3.1\t第一句\n3.2\t第二句\n5.1\t别的\n"
        out = llm_enhancer._parse_split_lines(raw, {3, 5})
        self.assertEqual(out, {3: ["第一句", "第二句"], 5: ["别的"]})

    def test_bare_index_maps_to_single(self):
        out = llm_enhancer._parse_split_lines("4\t单句\n", {4})
        self.assertEqual(out, {4: ["单句"]})

    def test_dirty_and_unwanted_dropped(self):
        out = llm_enhancer._parse_split_lines(
            "3.1\t好\n垃圾\n9.1\t越界\n3.x\t坏\n3.1\t", {3}
        )
        self.assertEqual(out, {3: ["好"]})

    def test_code_fence_stripped(self):
        raw = "```\n0.1\t你好。\n0.2\t再见。\n```"
        out = llm_enhancer._parse_split_lines(raw, {0})
        self.assertEqual(out, {0: ["你好。", "再见。"]})

    def test_sentences_kept_in_order(self):
        raw = "2.3\t三\n2.1\t一\n2.2\t二\n"
        out = llm_enhancer._parse_split_lines(raw, {2})
        self.assertEqual(out, {2: ["一", "二", "三"]})

    def test_structured_translation_response(self):
        raw = ('[{"slice_id":0,"sentences":["自然一点。","慢慢说。"]},'
               '{"slice_id":1,"sentences":["下一段。"]}]')
        out = llm_enhancer._parse_translation_response(raw, {0, 1})
        self.assertEqual(out, {
            0: ["自然一点。", "慢慢说。"],
            1: ["下一段。"],
        })

    def test_structured_translation_rejects_duplicate_slice_id(self):
        raw = ('[{"slice_id":0,"sentences":["本段。"]},'
               '{"slice_id":0,"sentences":["借来的下一段。"]}]')
        out = llm_enhancer._parse_translation_response(raw, {0})
        self.assertEqual(out, {0: ["本段。"]})


class BuildSentenceBlocksTest(unittest.TestCase):
    def test_char_proportional_splitting(self):
        blocks = _slices()
        per_slice = {0: ["你好。", "世界。"], 1: ["结束。"]}
        out = llm_enhancer._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([(b.index, b.start_ms, b.end_ms, b.text) for b in out], [
            (0, 0, 2500, "你好。"),    # 3 chars of 6 → half of [0,5000]
            (1, 2500, 5000, "世界。"),
            (2, 5000, 10000, "结束。"),
        ])

    def test_last_sentence_absorbs_remainder(self):
        blocks = [VTTBlock(index=0, start_ms=0, end_ms=6000, text="x")]
        per_slice = {0: ["短", "一个很长很长的句子占据了大部分内容"]}
        out = llm_enhancer._build_sentence_blocks(blocks, per_slice)
        self.assertEqual(out[-1].end_ms, 6000)  # last sentence ends at slice end

    def test_slice_without_sentences_is_discarded(self):
        # Untranslated slices must NOT fall back to the raw (often Japanese)
        # transcript — they are dropped outright.
        blocks = _slices()
        out = llm_enhancer._build_sentence_blocks(blocks, {})
        self.assertEqual(out, [])

    def test_upstream_original_only_slice_is_preserved_without_translation(self):
        blocks = _slices()
        blocks[0].text = ""
        blocks[0].skip_tts = True
        out = llm_enhancer._build_sentence_blocks(blocks, {1: ["再见。"]})
        self.assertEqual(
            [(b.index, b.text, b.start_ms, b.end_ms, b.skip_tts) for b in out],
            [(0, "", 0, 5000, True), (1, "再见。", 5000, 10000, False)],
        )

    def test_partial_translation_discards_only_missing(self):
        blocks = _slices()
        per_slice = {1: ["再见。"]}  # slice 0 untranslated
        out = llm_enhancer._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([(b.index, b.text, b.start_ms, b.end_ms) for b in out],
                         [(0, "再见。", 5000, 10000)])

    def test_indices_are_global_and_ordered(self):
        blocks = _slices()
        per_slice = {0: ["甲", "乙"], 1: ["丙"]}
        out = llm_enhancer._build_sentence_blocks(blocks, per_slice)
        self.assertEqual([b.index for b in out], [0, 1, 2])

    def test_detected_japanese_timeline_is_not_split_by_chinese_sentences(self):
        blocks = _slices()
        per_slice = {0: ["今天休息。", "好开心啊。"], 1: ["明天见。"]}
        out = llm_enhancer._build_sentence_blocks(
            blocks,
            per_slice,
            preserve_source_timeline=True,
        )
        self.assertEqual(
            [(b.start_ms, b.end_ms, b.text) for b in out],
            [(0, 5000, "今天休息。好开心啊。"), (5000, 10000, "明天见。")],
        )


class EnhanceAsrTest(unittest.TestCase):
    def test_end_to_end_translate_then_polish(self):
        blocks = _slices()
        fake = FakeHttpx([
            _openai_resp("摘要"),                                    # phase 1
            _openai_resp("0.1\t你好。\n0.2\t世界。\n1.1\t再见。"),   # translate-split
            _openai_resp("0\t[whispering] 你好。\n1\t世界……\n2\t再见！\n"),  # polish
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(blocks, _cfg())

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["summary_ok"])
        self.assertEqual(stats["translated"], 3)
        # Every polish row differs from the sentence text, so all 3 count.
        self.assertEqual(stats["rewritten"], 3)
        self.assertEqual(stats["total"], 3)
        self.assertEqual([b.text for b in out],
                         ["[whispering] 你好。", "世界……", "再见！"])
        # slice 0 (5s, two 3-char sentences) → 2.5s / 2.5s; slice 1 → whole.
        self.assertEqual([b.start_ms for b in out], [0, 2500, 5000])
        self.assertEqual([b.end_ms for b in out], [2500, 5000, 10000])
        self.assertEqual(len(fake.calls), 3)

    def test_translate_only_skips_polish_and_emotion_tags(self):
        blocks = _slices()
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp(
                '[{"slice_id":0,"sentences":["你好。","世界。"]},'
                '{"slice_id":1,"sentences":["再见。"]}]'
            ),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(
                blocks, _cfg(optimize_enabled=False)
            )

        self.assertEqual([b.text for b in out], ["你好。", "世界。", "再见。"])
        self.assertEqual(stats["translated"], 3)
        self.assertEqual(stats["rewritten"], 0)
        self.assertEqual(len(fake.calls), 2)
        request = json.loads(fake.calls[1]["json"]["messages"][-1]["content"])
        self.assertEqual(request, [
            {"slice_id": 0, "source_text": "原转写一"},
            {"slice_id": 1, "source_text": "原转写二"},
        ])

    def test_detected_source_boundaries_survive_translation(self):
        blocks = _slices()
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp(
                '[{"slice_id":0,"sentences":["你好。","世界。"]},'
                '{"slice_id":1,"sentences":["再见。"]}]'
            ),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, _ = llm_enhancer.enhance_asr_blocks(
                blocks,
                _cfg(optimize_enabled=False),
                preserve_source_timeline=True,
            )
        self.assertEqual(
            [(block.start_ms, block.end_ms, block.text) for block in out],
            [(0, 5000, "你好。世界。"), (5000, 10000, "再见。")],
        )

    def test_cleaned_empty_slice_never_enters_summary_translation_or_polish(self):
        blocks = _slices()
        blocks[0].text = ""
        blocks[0].skip_tts = True
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp('[{"slice_id":1,"sentences":["有效译文。"]}]'),
            _openai_resp("1\t润色后的有效译文。"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(blocks, _cfg())

        summary_input = fake.calls[0]["json"]["messages"][-1]["content"]
        translate_input = fake.calls[1]["json"]["messages"][-1]["content"]
        polish_input = fake.calls[2]["json"]["messages"][-1]["content"]
        self.assertNotIn("原转写一", summary_input)
        self.assertIn("原转写二", summary_input)
        self.assertNotIn('"slice_id": 0', translate_input)
        self.assertNotIn("0\t", polish_input)
        self.assertEqual([block.skip_tts for block in out], [True, False])
        self.assertEqual(out[1].text, "润色后的有效译文。")
        self.assertEqual(stats["untranslated"], 0)

    def test_all_cleaned_empty_slices_skip_every_llm_request(self):
        blocks = _slices()
        for block in blocks:
            block.text = ""
            block.skip_tts = True
        fake = FakeHttpx([])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(blocks, _cfg())
        self.assertEqual(fake.calls, [])
        self.assertTrue(all(block.skip_tts for block in out))
        self.assertEqual(stats["translated"], 0)

    def test_existing_subtitles_do_nothing_when_optimization_is_disabled(self):
        blocks = _blocks("已有中文字幕。")
        fake = FakeHttpx([])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            stats = enhance_blocks(blocks, _cfg(optimize_enabled=False))

        self.assertFalse(stats["enabled"])
        self.assertEqual(blocks[0].text, "已有中文字幕。")
        self.assertEqual(fake.calls, [])

    def test_not_configured_returns_original(self):
        blocks = _slices()
        fake = FakeHttpx([])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(blocks, {})
        self.assertFalse(stats["enabled"])
        self.assertEqual([b.text for b in out], ["原转写一", "原转写二"])
        self.assertEqual(fake.calls, [])

    def test_phase1_failure_still_attempts_translation(self):
        blocks = _slices()
        fake = FakeHttpx([_FakeResponse(500, {}, "boom")])
        cfg = _cfg(max_retries=0)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                llm_enhancer.enhance_asr_blocks(blocks, cfg)
        self.assertEqual(len(fake.calls), 2)


class TranslateSplitRetryTest(unittest.TestCase):
    """translate_split must retry slices the LLM omitted, not silently drop them."""

    def test_missing_slice_is_retried_and_recovered(self):
        blocks = _slices()  # indices 0, 1
        fake = FakeHttpx([
            _openai_resp("0.1\t你好。\n0.2\t世界。"),  # initial: slice 1 omitted
            _openai_resp("1.1\t再见。"),               # targeted retry recovers it
        ])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            client = LLMClient(LLMConfig(**cfg))
            per_slice = asyncio.run(client.translate_split(blocks, "摘要"))

        self.assertEqual(per_slice, {0: ["你好。", "世界。"], 1: ["再见。"]})
        self.assertEqual(len(fake.calls), 2)
        # The retry request carries only the missing slice's raw text, unnumbered.
        retry_text = fake.calls[1]["json"]["messages"][-1]["content"]
        self.assertIn("原转写二", retry_text)
        self.assertNotIn("原转写一", retry_text)

    def test_no_retry_when_every_slice_translated(self):
        blocks = _slices()
        fake = FakeHttpx([_openai_resp("0.1\t你好。\n1.1\t再见。")])
        cfg = _cfg(max_retries=2)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            client = LLMClient(LLMConfig(**cfg))
            per_slice = asyncio.run(client.translate_split(blocks, "摘要"))
        self.assertEqual(per_slice, {0: ["你好。"], 1: ["再见。"]})
        self.assertEqual(len(fake.calls), 1)  # no retry needed

    def test_suspicious_cross_slice_duplicate_is_verified_individually(self):
        blocks = [
            VTTBlock(index=0, start_ms=0, end_ms=5000,
                     text="どうせ彼女いないなら昔みたいに遊ぼうよ"),
            VTTBlock(index=1, start_ms=5000, end_ms=10000,
                     text="お医者さんごっこでもいいよ、せっかく帰ってきたんだし"),
        ]
        fake = FakeHttpx([
            _openai_resp(
                '[{"slice_id":0,"sentences":["玩医生游戏也行啊，难得回来，不好好玩怎么行。"]},'
                '{"slice_id":1,"sentences":["玩医生游戏也行啊，难得回来一趟，好好玩玩吧。"]}]'
            ),
            _openai_resp("反正你也没有女朋友，就像以前那样一起玩吧。"),
            _openai_resp("玩医生游戏也行啊，难得回来一趟，好好玩玩吧。"),
        ])
        with mock.patch.object(llm_enhancer, "httpx", fake):
            client = LLMClient(LLMConfig(**_cfg()))
            per_slice = asyncio.run(client.translate_split(blocks, "摘要"))

        self.assertEqual(per_slice, {
            0: ["反正你也没有女朋友，就像以前那样一起玩吧。"],
            1: ["玩医生游戏也行啊，难得回来一趟，好好玩玩吧。"],
        })
        self.assertEqual(len(fake.calls), 3)

    def test_exhausted_retries_fail_fast(self):
        blocks = _slices()
        fake = FakeHttpx([
            _openai_resp("0.1\t你好。"),   # slice 1 omitted initially
            _openai_resp("0.2\t还是漏了。"),  # retry also omits slice 1
        ])
        cfg = _cfg(max_retries=1)
        buf = io.StringIO()
        with mock.patch.object(llm_enhancer, "httpx", fake), \
             contextlib.redirect_stdout(buf):
            client = LLMClient(LLMConfig(**cfg))
            with self.assertRaises(PipelineError):
                asyncio.run(client.translate_split(blocks, "摘要"))
        self.assertEqual(len(fake.calls), 2)

    def test_untranslated_slice_fails_fast(self):
        # With retries exhausted and no translation, the slice's raw (Japanese)
        # text must NOT appear in the output — it is dropped entirely.
        blocks = _slices()
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp("0.1\t你好。"),          # slice 1 omitted
            _openai_resp("0.2\t还是没翻出来。"),  # retry omits slice 1 again
            _openai_resp("0\t[whispering] 你好。"),  # polish only slice 0
        ])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            with self.assertRaises(PipelineError):
                llm_enhancer.enhance_asr_blocks(blocks, cfg)
        self.assertEqual(len(fake.calls), 3)  # summary + translate + retry; no polish

    def test_end_to_end_recovers_last_slice(self):
        # The reported bug: the LLM omitted the final slice's translation, so the
        # last sentence-block fell back to the raw (Japanese) transcript.
        blocks = _slices() + [VTTBlock(index=2, start_ms=10000, end_ms=15000,
                                       text="一緒に気持ち良くなりますよね。")]
        fake = FakeHttpx([
            _openai_resp("摘要"),
            _openai_resp("0.1\t你好。\n1.1\t再见。"),   # slice 2 omitted
            _openai_resp("2.1\t一起变得更舒服吧。"),   # retry recovers it
            _openai_resp("0\t你好。\n1\t再见。\n2\t一起变得更舒服吧。"),
        ])
        cfg = _cfg(max_retries=1)
        with mock.patch.object(llm_enhancer, "httpx", fake):
            out, stats = llm_enhancer.enhance_asr_blocks(blocks, cfg)
        self.assertEqual(stats["translated"], 3)
        self.assertEqual(stats["untranslated"], 0)
        self.assertEqual([b.text for b in out],
                         ["你好。", "再见。", "一起变得更舒服吧。"])
        self.assertEqual(len(fake.calls), 4)  # summary + translate + retry + polish


if __name__ == "__main__":
    unittest.main()
