"""Discover and call the installed VAD without importing its model dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
from uuid import uuid4
import weakref

from ..analysis_plugins import AnalysisPluginError, normalized_audio

PROTOCOL = "asmr-vad-local/1"


@dataclass(frozen=True)
class LocalVADInstallation:
    root: Path
    python: Path


def discover_local_vad(registry_path: Path | None = None) -> LocalVADInstallation:
    if registry_path is None:
        app_data = os.environ.get("APPDATA")
        if not app_data:
            raise AnalysisPluginError("无法定位本地 ASMR VAD，请通过插件安装器安装")
        registry_path = Path(app_data) / "ASMRTranstor/plugins.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        plugin = next(item for item in registry["plugins"].values()
                      if item.get("id") == "asmr-vad" and item.get("enabled"))
        root = Path(plugin["path"])
        if not root.is_absolute():
            raise ValueError("Installation path must be absolute")
        manifest = json.loads((root / "plugin-manifest.json").read_text(encoding="utf-8-sig"))
        if (manifest.get("id") != "asmr-vad" or manifest.get("apiVersion") != "asmr-vad/1"
                or "local-vad-worker-v1" not in manifest.get("capabilities", [])):
            raise ValueError("请更新 ASMR VAD 至支持本地推理的版本")
        for name in ("local_worker.py", "source/engine.py", "source/model/model.onnx"):
            if not (root / name).is_file():
                raise ValueError(f"插件文件缺失：{name}")
        reference = root / ".python-environment"
        if reference.is_file():
            key = reference.read_text(encoding="utf-8").strip()
            if not re.fullmatch(r"[0-9a-fA-F]{64}", key) or root.parent.name != "asmr-vad":
                raise ValueError("插件 Python 环境引用无效")
            environment = root.parents[2] / ".environments/asmr-vad" / key
            if not (environment / ".ready").is_file():
                raise ValueError("插件 Python 环境未就绪")
        else:
            environment = root / ".venv"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise ValueError("插件 Python 环境缺失")
        return LocalVADInstallation(root, python)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration) as error:
        raise AnalysisPluginError(f"本地 ASMR VAD 不可用，请安装或修复插件：{error}") from error


def worker_environment() -> dict[str, str]:
    # Match the installer's isolation: no ambient Python paths, proxies or IDE DLLs.
    allowed = {"systemroot", "systemdrive", "windir", "temp", "tmp", "userprofile",
               "appdata", "localappdata", "programdata", "homedrive", "homepath",
               "home", "lang", "lc_all", "tmpdir"}
    environment = {key: value for key, value in os.environ.items() if key.lower() in allowed}
    system = Path(os.environ.get("SystemRoot", "C:/Windows"))
    environment["PATH"] = os.pathsep.join(map(str, (system / "System32", system))) if os.name == "nt" else os.defpath
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    return environment


def _stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for stream in (process.stdin, process.stdout):
        if stream:
            stream.close()


def _read_responses(stream, responses):
    try:
        for line in stream:
            responses.put(line)
    finally:
        responses.put(None)


class LocalVADClient:
    """One lazy worker per adapter; repeated crops reuse the loaded model."""

    def __init__(self, *, installation=None):
        self.installation = installation
        self._process = None
        self._cleanup = None
        self._responses = None
        self._lock = threading.Lock()

    def close(self):
        if self._cleanup is not None:
            self._cleanup()
        self._process = self._cleanup = self._responses = None

    def _start(self):
        installation = self.installation or discover_local_vad()
        self._process = subprocess.Popen(
            [str(installation.python), "-I", "-X", "utf8", "-u", str(installation.root / "local_worker.py")],
            cwd=installation.root, env=worker_environment(), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self._cleanup = weakref.finalize(self, _stop_process, self._process)
        self._responses = queue.Queue()
        threading.Thread(target=_read_responses, args=(self._process.stdout, self._responses), daemon=True).start()

    def call(self, operation: str, *, timeout=120, **payload):
        with self._lock:
            try:
                if self._process is None or self._process.poll() is not None:
                    self.close()
                    self._start()
                request_id = uuid4().hex
                self._process.stdin.write(json.dumps({"protocol": PROTOCOL, "request_id": request_id,
                                                     "op": operation, **payload}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                line = self._responses.get(timeout=timeout)
                if line is None:
                    raise AnalysisPluginError("ASMR VAD 本地推理进程意外退出，请修复插件后重试")
                body = json.loads(line)
                if body.get("protocol") != PROTOCOL or body.get("request_id") != request_id:
                    raise AnalysisPluginError("ASMR VAD 本地响应协议不匹配")
                if body.get("ok") is not True:
                    raise AnalysisPluginError(f"ASMR VAD 本地推理失败：{body.get('error', '未知错误')}")
                return body
            except Exception as error:
                self.close()
                if isinstance(error, AnalysisPluginError):
                    raise
                detail = "等待本地推理超时" if isinstance(error, queue.Empty) else str(error)
                raise AnalysisPluginError(f"ASMR VAD 本地调用失败：{detail}") from error

    def request(self, route, audio_path, data, *, start_ms=None, end_ms=None):
        with normalized_audio(audio_path, start_ms, end_ms) as (audio, duration):
            body = self.call("detect", timeout=21600, audio_path=str(audio.resolve()), options=data)
        if type(body.get("duration_ms")) is not int or abs(body["duration_ms"] - duration) > 1:
            raise AnalysisPluginError("ASMR VAD 本地响应音频时长不匹配")
        return body, duration


def check_local_vad():
    client = LocalVADClient()
    try:
        client.call("check")
    finally:
        client.close()
