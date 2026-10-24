import json
import unittest

from modules.error import PipelineError, TTSError, descriptor, safe_detail, unexpected_error


class PipelineErrorContractTest(unittest.TestCase):
    def test_tts_memory_failures_take_priority_over_http_500(self):
        for cause in ("bad allocation", "std::bad_alloc", "out of memory", "MemoryError"):
            with self.subTest(cause=cause):
                payload = TTSError(f"HTTP 500 ONNXRuntimeError: {cause}").to_payload()
                self.assertEqual(payload["code"], "TTS_OUT_OF_MEMORY")
                self.assertIn("释放内存后重试", payload["message"])
                self.assertEqual(payload["action"], "view_log")

    def test_tts_server_errors_are_distinct_from_connection_errors(self):
        self.assertEqual(TTSError("HTTP 500 / SYNTHESIS_FAILED").descriptor.code, "TTS_SERVICE_ERROR")
        self.assertEqual(TTSError("connection refused").descriptor.code, "TTS_UNREACHABLE")

    def test_descriptor_serializes_stable_contract(self):
        error = PipelineError(
            descriptor("TTS_SEGMENT_FAILED", segment=3),
            detail="HTTP 500 Authorization: Bearer top-secret",
            context={"segment": 3, "api_key": "must-not-leak"},
        )
        payload = error.to_payload(log_path="pipeline.log")
        self.assertEqual(payload["contractVersion"], 1)
        self.assertEqual(payload["code"], "TTS_SEGMENT_FAILED")
        self.assertEqual(payload["stage"], "tts")
        self.assertTrue(payload["retryable"])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("must-not-leak", serialized)
        self.assertEqual(payload["context"], {"segment": 3})

    def test_detail_redacts_common_secret_shapes_and_is_bounded(self):
        value = safe_detail("api_key=abc token:xyz https://x.test?a=1&secret=q " + "x" * 600)
        self.assertNotIn("abc", value)
        self.assertNotIn("xyz", value)
        self.assertNotIn("secret=q", value)
        self.assertLessEqual(len(value), 500)

    def test_unexpected_error_has_stable_code(self):
        payload = unexpected_error(RuntimeError("boom")).to_payload()
        self.assertEqual(payload["code"], "PROGRAM_UNEXPECTED")
        self.assertNotIn("errorId", payload)

    def test_edge_endpoint_credentials_are_redacted(self):
        value = safe_detail(
            "wss://api.msedgeservices.com/test?Ocp-Apim-Subscription-Key=client-secret"
            "&Sec-MS-GEC=signature-secret&Sec-MS-GEC-Version=1-143.0.3650.75"
        )
        self.assertNotIn("client-secret", value)
        self.assertNotIn("signature-secret", value)
        self.assertIn("Sec-MS-GEC-Version=1-143.0.3650.75", value)


if __name__ == "__main__":
    unittest.main()
