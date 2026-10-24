"""Model-free tests for schema, per-segment isolation and the HTTP boundary."""

import asyncio
import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

import httpx

from modules.error import PipelineError, TTSError
from modules.tts.common_tts_adapter import CommonTTSAdapter
from modules.tts.factory import create_tts
from modules.tts.plugin_contract import TTSSynthesisResult
from modules.tts.plugin_manifest import read_runtime_configuration, resolve_parameters, validate_manifest
from modules.tts.plugin_writer import write_plugin_result
from modules.tts_engine import generate_tts


def wav_bytes(frames=80):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * frames)
    return data.getvalue()


def descriptor():
    return {"schema_version": 1, "protocol": "common-tts/1", "id": "test-tts", "name": "Test",
        "version": "1", "capabilities": {"required_assets": ["source_speech"], "text_policy": "preserve"},
        "parameters": {"type": "object", "additionalProperties": False, "properties": {
            "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.8, "x-segment-override": True},
            "reference": {"type": "string", "x-control": "audio-file", "x-role": "voice_reference"},
            "sampling": {"type": "object", "default": {}, "additionalProperties": False,
                         "properties": {"max_mel_tokens": {"type": "integer", "default": 1500}}}}}}


class ManifestTests(unittest.TestCase):
    def test_defaults_preserve_nested_configuration_and_validate_overrides(self):
        manifest = validate_manifest(descriptor())
        self.assertEqual(resolve_parameters(manifest, {}, {"temperature": 1.2}),
                         {"temperature": 1.2, "sampling": {"max_mel_tokens": 1500}})
        with self.assertRaises(TTSError):
            resolve_parameters(manifest, {}, {"sampling": {}})
        for parameters in ({"unknown": 1}, {"temperature": "hot"}, {"temperature": 3}):
            with self.assertRaises(TTSError):
                resolve_parameters(manifest, parameters)

    def test_external_schema_refs_and_unsupported_execution_rejected(self):
        for change in ({"parameters": {"type": "object", "additionalProperties": False, "$ref": "https://example.com/schema"}},
                       {"capabilities": {"execution": "conversation"}}, {"output": "pcm"}):
            manifest = {**descriptor(), **change}
            with self.assertRaises(TTSError):
                validate_manifest(manifest)

    def test_parameter_errors_do_not_echo_values(self):
        with self.assertRaises(TTSError) as raised:
            resolve_parameters(validate_manifest(descriptor()), {"temperature": "private-value"})
        self.assertNotIn("private-value", str(raised.exception))


class AdapterTests(unittest.TestCase):
    def test_required_segment_roles_are_checked_after_reference_preflight_before_http(self):
        manifest = descriptor()
        manifest["capabilities"]["required_assets"] = ["source_speech", "source_segment", "voice_reference"]
        requests = []
        def handle(request):
            requests.append(request.read())
            return httpx.Response(200, content=wav_bytes(), headers={"Content-Type": "audio/wav",
                "X-TTS-Contract-Version": "1", "X-Request-ID": request.headers["X-Request-ID"]})

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "speech.wav"
            source.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=manifest, base_url="http://plugin",
                parameters={"reference": str(source)}, transport=httpx.MockTransport(handle))
            adapter.validate_reference_assets()
            base = {"index": 0, "text": "中文"}
            for segment, missing in ((base, "source_speech"),
                    ({**base, "speech_file": str(source)}, "source_segment")):
                with self.subTest(missing=missing), self.assertRaisesRegex(TTSError, missing):
                    bound = adapter.bind_segment(segment)
                    asyncio.run(bound.synthesize("中文", "", str(Path(folder) / "rejected.wav")))
            self.assertEqual(requests, [])
            self.assertFalse((Path(folder) / "rejected.wav").exists())
            bound = adapter.bind_segment({**base, "speech_file": str(source), "segment_file": str(source)})
            target = Path(folder) / "accepted.wav"
            asyncio.run(bound.synthesize("中文", "", str(target)))
            self.assertEqual(target.read_bytes(), wav_bytes())
            self.assertEqual(len(requests), 1)
            for role in manifest["capabilities"]["required_assets"]:
                self.assertIn(f'"role": "{role}"'.encode(), requests[0])
            self.assertEqual(adapter.parameters["reference"], str(source))

    def test_http_failures_produce_safe_frontend_error_payloads(self):
        cases = [
            (500, {"code": "OUT_OF_MEMORY"}, "TTS_OUT_OF_MEMORY"),
            (500, {"code": "SYNTHESIS_FAILED", "message": "ONNXRuntimeError: bad allocation private-text"}, "TTS_OUT_OF_MEMORY"),
            (500, {"code": "MODEL_LOAD_FAILED"}, "TTS_MODEL_LOAD_FAILED"),
            (500, {"code": "SYNTHESIS_FAILED"}, "TTS_SERVICE_ERROR"),
            (503, {"code": "HTTP_ERROR"}, "TTS_SERVICE_ERROR"),
            (401, {"code": "AUTH_INVALID"}, "TTS_AUTH_INVALID"),
            (429, {"code": "RATE_LIMITED"}, "TTS_RATE_LIMITED"),
        ]
        adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test")
        for status, error, expected in cases:
            with self.subTest(status=status, error=error):
                with self.assertRaises(TTSError) as raised:
                    adapter._check(httpx.Response(status, json={"error": error}))
                payload = raised.exception.to_payload()
                self.assertEqual(payload["code"], expected)
                self.assertEqual(payload["stage"], "tts")
                self.assertEqual(payload["context"]["status"], status)
                self.assertNotIn("private-text", json.dumps(payload))

    def test_malformed_http_error_bodies_remain_service_errors(self):
        adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test")
        for body in ([], {"error": None}, {"error": []}, {"error": "private-text"}):
            with self.subTest(body=body):
                with self.assertRaises(TTSError) as raised:
                    adapter._check(httpx.Response(500, json=body))
                self.assertEqual(raised.exception.descriptor.code, "TTS_SERVICE_ERROR")
                self.assertNotIn("private-text", str(raised.exception))
        with self.assertRaises(TTSError) as raised:
            adapter._check(httpx.Response(500, text="private-text bad allocation"))
        self.assertEqual(raised.exception.descriptor.code, "TTS_SERVICE_ERROR")

    def test_pipeline_preserves_memory_error_for_frontend(self):
        def handle(request):
            return httpx.Response(500, json={"error": {
                "code": "SYNTHESIS_FAILED", "message": "Exception during initialization: bad allocation"}})

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test",
                                       transport=httpx.MockTransport(handle))
            with patch("modules.tts_engine._get_adapter", return_value=adapter):
                with self.assertRaises(PipelineError) as raised:
                    generate_tts([{"index": 0, "text": "中文", "speech_file": str(source)}],
                                 folder, max_concurrent=1, backend="common-tts")
            payload = raised.exception.to_payload()
            self.assertEqual(payload["code"], "TTS_OUT_OF_MEMORY")
            self.assertIn("内存分配失败", payload["message"])
            self.assertFalse((Path(folder) / "tts_0000.wav").exists())

    def test_structured_results_publish_word_boundaries(self):
        def handle(request):
            return httpx.Response(200, json={"contract_version": 1, "request_id": request.headers["X-Request-ID"],
                "segment_id": "42", "audio_base64": base64.b64encode(wav_bytes()).decode(),
                "boundaries": {"type": "word", "unit": "ms", "words": [{"text": "中文", "offset_ms": 0, "duration_ms": 3}]}},
                headers={"X-TTS-Contract-Version": "1", "X-Request-ID": request.headers["X-Request-ID"]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.wav"; path.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test", transport=httpx.MockTransport(handle))
            adapter = adapter.bind_segment({"index": 0, "segment_id": "42", "text": "中文", "speech_file": str(path)})
            target = Path(folder) / "tts.wav"
            asyncio.run(adapter.synthesize("中文", "", str(target)))
            report = json.loads(Path(str(target) + ".boundaries.json").read_text(encoding="utf-8"))
            self.assertEqual(report["words"][0]["text"], "中文")

    def test_large_constructor_configuration_can_be_read_from_stdin(self):
        manifest = descriptor()
        manifest["parameters"]["properties"]["description"] = {"type": "string"}
        payload = {"manifest": manifest, "base_url": "http://test", "parameters": {"description": "x" * 40000}}
        self.assertEqual(read_runtime_configuration(io.BytesIO(json.dumps(payload).encode())), payload)
        for raw in (b"[]", b'{"invalid":NaN}', b"x" * (2 * 1024 * 1024 + 1)):
            with self.assertRaises(TTSError):
                read_runtime_configuration(io.BytesIO(raw))

    def test_artifact_regeneration_keeps_the_same_upstream_segment_input(self):
        from modules.audio_handlers import TTSArtifactHandler
        calls = []
        def handle(request):
            calls.append(request.read())
            return httpx.Response(200, content=wav_bytes(), headers={"Content-Type": "audio/wav",
                "X-TTS-Contract-Version": "1", "X-Request-ID": request.headers["X-Request-ID"]})
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"; source.write_bytes(wav_bytes())
            output = Path(folder) / "tts_0003.wav"; output.write_bytes(wav_bytes())
            segment = {"index": 3, "segment_id": "stable-42", "text": "中文", "speech_file": str(source), "tts_file": str(output)}
            kwargs = dict(manifest=descriptor(), base_url="http://test", transport=httpx.MockTransport(handle))
            with patch("modules.audio_handlers.detect_tts_artifact", side_effect=[{"flagged": True}, {"flagged": False}]):
                result = TTSArtifactHandler(backend="common-tts", backend_kwargs=kwargs).process([segment], folder, max_workers=1)
            self.assertIn(b'"id": "stable-42"', calls[0])
            self.assertIn(b'source_speech.wav', calls[0])
            self.assertTrue(Path(result[0]["tts_file"]).is_file())

    def test_pipeline_uploads_each_segment_and_publishes_only_valid_audio(self):
        requests = []
        def handle(request):
            body = request.read()
            requests.append(body)
            self.assertIn('filename="source_speech.wav"'.encode(), body)
            self.assertNotIn(str(source).encode(), body)
            return httpx.Response(200, content=wav_bytes(), headers={"Content-Type": "audio/wav",
                "X-TTS-Contract-Version": "1", "X-Request-ID": request.headers["X-Request-ID"]})
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "private-source.wav"
            source.write_bytes(wav_bytes())
            segments = [{"index": i, "segment_id": f"stable-{i}", "text": f"中文{i}", "speaker": "A",
                         "speech_file": str(source), "source_text": "日本語", "tts_overrides": {"temperature": i + 0.2}}
                        for i in range(2)]
            kwargs = dict(manifest=descriptor(), base_url="http://test", transport=httpx.MockTransport(handle))
            adapter = create_tts("common-tts", **kwargs)
            with patch("modules.tts_engine._get_adapter", return_value=adapter):
                result = generate_tts(segments, folder, max_concurrent=2, backend="common-tts")
            self.assertEqual([s["segment_id"] for s in result], ["stable-0", "stable-1"])
            self.assertTrue(all(Path(s["tts_file"]).is_file() for s in result))
            self.assertIsNone(adapter._request)
            self.assertEqual(len(requests), 2)
            for index, body in enumerate(requests):
                self.assertIn(f'"id": "stable-{index}"'.encode(), body)
                self.assertIn(f'"temperature": {index + 0.2}'.encode(), body)

    def test_reference_contents_and_model_limits_affect_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "speech.wav"
            path.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test")
            one = adapter.bind_segment({"index": 0, "text": "中文", "speech_file": str(path)})
            key = one.cache_inputs()
            path.write_bytes(wav_bytes(160))
            two = adapter.bind_segment({"index": 0, "text": "中文", "speech_file": str(path)})
            self.assertNotEqual(key, two.cache_inputs())
            self.assertEqual(key["parameters"]["sampling"]["max_mel_tokens"], 1500)

    def test_multiple_nested_references_are_uploaded(self):
        manifest = descriptor()
        manifest["parameters"]["properties"]["references"] = {"type": "array", "items": {
            "type": "object", "properties": {"audio": {"type": "string", "x-control": "audio-file"}, "text": {"type": "string"}}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"
            path.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=manifest, base_url="http://test",
                parameters={"references": [{"audio": str(path), "text": "sample text"}]})
            bound = adapter.bind_segment({"index": 0, "text": "中文", "speech_file": str(path)})
            key = bound.cache_inputs()
            self.assertEqual(len(key["assets"]), 2)
            self.assertIn("asset_id", key["parameters"]["references"][0]["audio"])

    def test_timeout_is_not_replayed(self):
        calls = []
        def handle(request):
            calls.append(request)
            raise httpx.ReadTimeout("private request body")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"; path.write_bytes(wav_bytes())
            adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test", transport=httpx.MockTransport(handle))
            adapter = adapter.bind_segment({"index": 0, "text": "中文", "speech_file": str(path)})
            with self.assertRaises(TTSError) as raised:
                asyncio.run(adapter.synthesize("中文", "", str(Path(folder) / "out.wav")))
            self.assertEqual(len(calls), 1)
            self.assertNotIn("private request", str(raised.exception))

    def test_truncated_audio_never_published(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.wav"
            with self.assertRaises(TTSError):
                write_plugin_result(TTSSynthesisResult(wav_bytes()[:-20], "audio/wav"), str(path))
            self.assertFalse(path.exists())

    def test_segment_binding_copies_caller_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"; path.write_bytes(wav_bytes())
            segment = {"index": 0, "text": "中文", "speech_file": str(path), "tts_overrides": {"temperature": 1}}
            original = copy.deepcopy(segment)
            adapter = CommonTTSAdapter(manifest=descriptor(), base_url="http://test").bind_segment(segment)
            segment["tts_overrides"]["temperature"] = 2
            self.assertEqual(adapter._request.overrides, original["tts_overrides"])


if __name__ == "__main__":
    unittest.main()
