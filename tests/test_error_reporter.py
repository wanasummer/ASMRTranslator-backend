import io
import unittest
from unittest import mock

from modules.app_version import APP_VERSION
from modules.error import PipelineError, descriptor
from modules.error_reporter import (
    DEFAULT_LOG_UPLOAD_URL,
    build_error_report,
    redact_log_text,
    report_error_async,
)


class _RecordingUploader:
    def __init__(self, failure=None):
        self.failure = failure
        self.reports = []

    def upload(self, report):
        self.reports.append(report)
        if self.failure:
            raise self.failure


class ErrorReporterTest(unittest.TestCase):
    def test_default_endpoint_is_disabled(self):
        self.assertEqual(DEFAULT_LOG_UPLOAD_URL, "")
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        with mock.patch("modules.error_reporter.HttpLogUploader") as transport:
            self.assertIsNone(report_error_async(error, "trace", environ={}))
        transport.assert_not_called()

    def test_custom_endpoint_enables_upload(self):
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        with mock.patch("modules.error_reporter.HttpLogUploader") as transport:
            worker = report_error_async(error, "trace", environ={
                "ASMR_LOG_UPLOAD_URL": "https://logs.example.test/log",
            })
            self.assertIsNotNone(worker)
            worker.join(timeout=1)
        transport.assert_called_once_with(
            endpoint="https://logs.example.test/log")
        transport.return_value.upload.assert_called_once()


    def test_build_report_includes_contract_and_redacts_secrets(self):
        error = PipelineError(
            descriptor("TTS_SEGMENT_FAILED", segment=2),
            detail="api_key=private-value",
        )
        report = build_error_report(
            error,
            "Authorization: Bearer traceback-secret",
            environ={"ASMR_ASSET_ID": "asset-1"},
        )
        self.assertIn('"code": "TTS_SEGMENT_FAILED"', report)
        self.assertIn(f'"appVersion": "{APP_VERSION}"', report)
        self.assertIn('"assetId": "asset-1"', report)
        self.assertNotIn("private-value", report)
        self.assertNotIn("traceback-secret", report)

    def test_build_report_honors_explicit_application_version(self):
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))

        report = build_error_report(
            error,
            "",
            environ={"ASMR_APP_VERSION": "1.5.4-preview"},
        )

        self.assertIn('"appVersion": "1.5.4-preview"', report)

    def test_async_upload_uses_injected_adapter(self):
        uploader = _RecordingUploader()
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"), detail="boom")
        worker = report_error_async(error, "trace", uploader=uploader, environ={})
        self.assertIsNotNone(worker)
        worker.join(timeout=1)
        self.assertEqual(len(uploader.reports), 1)

    def test_upload_failure_is_silent_in_production(self):
        uploader = _RecordingUploader(RuntimeError("endpoint expired"))
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        stderr = io.StringIO()
        with mock.patch("modules.error_reporter._is_packaged", return_value=True), \
                mock.patch("modules.error_reporter.sys.stderr", stderr):
            worker = report_error_async(error, "", uploader=uploader, environ={})
            worker.join(timeout=1)
        self.assertEqual(stderr.getvalue(), "")

    def test_report_construction_failure_is_also_isolated(self):
        uploader = _RecordingUploader()
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        stderr = io.StringIO()
        with mock.patch(
            "modules.error_reporter.build_error_report", side_effect=RuntimeError("format failed")
        ), mock.patch("modules.error_reporter._is_packaged", return_value=True), mock.patch(
            "modules.error_reporter.sys.stderr", stderr
        ):
            worker = report_error_async(error, "", uploader=uploader, environ={})
            worker.join(timeout=1)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(uploader.reports, [])

    def test_upload_failure_is_visible_in_development(self):
        uploader = _RecordingUploader(RuntimeError("endpoint expired"))
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        stderr = io.StringIO()
        with mock.patch("modules.error_reporter.sys.stderr", stderr):
            worker = report_error_async(
                error, "", uploader=uploader, environ={"ASMR_ENV": "development"}
            )
            worker.join(timeout=1)
        self.assertIn("[log-upload] endpoint expired", stderr.getvalue())

    def test_upload_can_be_disabled(self):
        uploader = _RecordingUploader()
        error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
        worker = report_error_async(
            error, "", uploader=uploader, environ={"ASMR_DISABLE_LOG_UPLOAD": "1"}
        )
        self.assertIsNone(worker)
        self.assertEqual(uploader.reports, [])

    def test_http_upload_sends_utf8_without_authorization(self):
        from modules.error_reporter import HttpLogUploader
        with mock.patch("modules.error_reporter.urlopen") as open_request:
            open_request.return_value.__enter__.return_value.status = 201
            HttpLogUploader("https://logs.example.test/log").upload("诊断日志")
        request = open_request.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.data.decode("utf-8"), "诊断日志")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(open_request.call_args.kwargs["timeout"], 2.5)

    def test_redacts_query_tokens(self):
        self.assertEqual(
            redact_log_text("https://example.test/log?token=secret&x=1"),
            "https://example.test/log?token=[REDACTED]&x=1",
        )


class BuildLogDefaultsTest(unittest.TestCase):
    def test_source_runs_do_not_read_packaged_defaults(self):
        from modules.error_reporter import _build_log_defaults
        with mock.patch("modules.error_reporter._is_packaged", return_value=False), mock.patch("modules.error_reporter.Path.read_text") as reader:
            self.assertEqual(_build_log_defaults(), {})
        reader.assert_not_called()

    def test_packaged_upload_uses_build_values_then_runtime_overrides(self):
        import tempfile
        from pathlib import Path
        from scripts.prepare_log_defaults import prepare_log_defaults
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "error_reporter.py"
            prepare_log_defaults(module.with_name("_log_upload_defaults.json"), {
                "ASMR_LOG_UPLOAD_URL": "https://build.example.test/log",
                "FISH_API_KEY": "do-not-embed",
            })
            error = PipelineError(descriptor("PROGRAM_UNEXPECTED"))
            with mock.patch("modules.error_reporter._is_packaged", return_value=True), mock.patch("modules.error_reporter.__file__", str(module)):
                for env, endpoint in [({}, "https://build.example.test/log"),
                                      ({"ASMR_LOG_UPLOAD_URL": "https://runtime.example.test/log"}, "https://runtime.example.test/log")]:
                    with self.subTest(env=env), mock.patch("modules.error_reporter.HttpLogUploader") as transport:
                        worker = report_error_async(error, "trace", environ=env)
                        self.assertIsNotNone(worker)
                        worker.join(timeout=1)
                        transport.assert_called_once_with(endpoint=endpoint)
                        transport.return_value.upload.assert_called_once()
                with mock.patch("modules.error_reporter.HttpLogUploader") as transport:
                    self.assertIsNone(report_error_async(error, "trace", environ={"ASMR_LOG_UPLOAD_URL": ""}))
                    self.assertIsNone(report_error_async(error, "trace", environ={"ASMR_DISABLE_LOG_UPLOAD": "1"}))
                    transport.assert_not_called()

    def test_missing_or_malformed_build_values_fall_back_to_empty(self):
        from modules.error_reporter import _build_log_defaults
        with mock.patch("modules.error_reporter._is_packaged", return_value=True):
            with mock.patch("modules.error_reporter.Path.read_text", side_effect=FileNotFoundError):
                self.assertEqual(_build_log_defaults(), {})
            for payload in ['[]', 'invalid json', '{"ASMR_LOG_UPLOAD_URL": 12, "OTHER_KEY": "ignored"}']:
                with self.subTest(payload=payload), mock.patch("modules.error_reporter.Path.read_text", return_value=payload):
                    self.assertEqual(_build_log_defaults(), {})


if __name__ == "__main__":
    unittest.main()
