"""Local VAD discovery, isolated transport, restart and failure contracts."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from modules.analysis_plugins import AnalysisPluginError
from modules.vad.local_runtime import (LocalVADClient, LocalVADInstallation,
                                       discover_local_vad, worker_environment)
from modules import preflight


class LocalVADTests(unittest.TestCase):
    def installation(self, directory):
        root = Path(directory) / "plugins/asmr-vad/0.2.0"
        root.mkdir(parents=True)
        for name in ("local_worker.py", "source/engine.py", "source/model/model.onnx"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        (root / "plugin-manifest.json").write_text(json.dumps({"id": "asmr-vad", "apiVersion": "asmr-vad/1",
             "capabilities": ["local-vad-worker-v1"]}), encoding="utf-8")
        key = "a" * 64
        (root / ".python-environment").write_text(key)
        environment = Path(directory) / ".environments/asmr-vad" / key
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.touch()
        (environment / ".ready").touch()
        registry = Path(directory) / "plugins.json"
        registry.write_text(json.dumps({"plugins": {"asmrVad": {
            "id": "asmr-vad", "path": str(root), "enabled": True, "port": 0}}}), encoding="utf-8")
        return root, python, registry

    def test_registry_survives_restart_without_manager_or_port(self):
        with tempfile.TemporaryDirectory(prefix="VAD 安装 ") as directory:
            root, python, registry = self.installation(directory)
            for _ in range(2):
                self.assertEqual(discover_local_vad(registry), LocalVADInstallation(root, python))
            (root / "source/model/model.onnx").unlink()
            with self.assertRaisesRegex(AnalysisPluginError, "文件缺失"):
                discover_local_vad(registry)

    def test_incomplete_disabled_and_untrusted_environment_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, python, registry = self.installation(directory)
            python.unlink()
            with self.assertRaisesRegex(AnalysisPluginError, "环境缺失"):
                discover_local_vad(registry)
            python.touch()
            (root / ".python-environment").write_text("../../outside")
            with self.assertRaisesRegex(AnalysisPluginError, "引用无效"):
                discover_local_vad(registry)
            registry.write_text('{"plugins": {}}')
            with self.assertRaises(AnalysisPluginError):
                discover_local_vad(registry)

    def test_worker_reuses_model_process_closes_and_recovers_after_exit(self):
        with tempfile.TemporaryDirectory(prefix="VAD worker ") as directory:
            root = Path(directory)
            (root / "local_worker.py").write_text('''import json, sys, os
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'exit':
        sys.exit(1)
    print(json.dumps({'protocol': request['protocol'], 'request_id': request['request_id'],
                      'ok': True, 'pid': os.getpid(), 'duration_ms': 1000, 'segments': []}), flush=True)
''', encoding="utf-8")
            client = LocalVADClient(installation=LocalVADInstallation(root, Path(sys.executable)))
            try:
                pid = client.call("check")["pid"]
                self.assertEqual(client.call("check")["pid"], pid)
                with self.assertRaisesRegex(AnalysisPluginError, "意外退出"):
                    client.call("exit")
                self.assertIsNotNone(client.call("check")["pid"])
                process = client._process
            finally:
                client.close()
            self.assertIsNotNone(process.poll())

    def test_protocol_errors_and_timeout_stop_worker(self):
        for code in ("print('{}', flush=True)", "import time; time.sleep(5)"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "local_worker.py").write_text("import sys\nsys.stdin.readline()\n" + code)
                client = LocalVADClient(installation=LocalVADInstallation(root, Path(sys.executable)))
                with self.assertRaises(AnalysisPluginError):
                    client.call("check", timeout=0.2)
                self.assertIsNone(client._process)

    def test_ambient_python_proxy_and_gateway_settings_are_removed(self):
        with patch.dict(os.environ, {"PYTHONPATH": "bad", "HTTPS_PROXY": "http://bad",
                                    "ASMRTRANSTOR_LOG_GATEWAY_URL": "ws://bad/log"}):
            environment = worker_environment()
        for name in ("PYTHONPATH", "HTTPS_PROXY", "ASMRTRANSTOR_LOG_GATEWAY_URL"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")


class LocalPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_check_does_not_contact_http(self):
        with patch("modules.vad.local_runtime.check_local_vad") as check, patch(
                "modules.preflight.async_client", side_effect=AssertionError("HTTP called")):
            result = await preflight.run_preflight_check("vad", {})
            self.assertTrue(result.success, result.message)
            check.assert_called_once()
        with patch("modules.vad.local_runtime.check_local_vad", side_effect=AnalysisPluginError("模型缺失")):
            result = await preflight.run_preflight_check("vad", {})
            self.assertFalse(result.success)
            self.assertIn("模型缺失", result.message)
