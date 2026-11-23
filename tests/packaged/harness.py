"""Extensible clean-room harness for testing the packaged sidecar (`main.exe`).

To add a new end-to-end packaged test for a future feature:
1. Create a `test_<feature>.py` file in `models/tests/packaged/` (or add a test
   method to an existing file) inheriting from `PackagedSidecarTestCase`.
2. Use `self.run_sidecar([...])` to invoke the sidecar in a clean-room environment
   (stripped `PATH`, isolated temp `cwd`, non-ASCII user paths).
3. Use `self.stub.add_route("POST", "/custom-path", handler)` if your feature
   needs a custom HTTP endpoint or custom response from the local loopback server.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from mutagen import id3

MODELS_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = MODELS_ROOT.parent


def build_speech_wav_bytes(duration_sec: float, sample_rate: int = 24000) -> bytes:
    """Synthesize a deterministic speech-modulated PCM WAV that triggers FSMN-VAD."""
    num_samples = int(duration_sec * sample_rate)
    silence_lead = min(int(0.2 * sample_rate), int(num_samples * 0.1))
    silence_tail = min(int(0.2 * sample_rate), int(num_samples * 0.1))
    seed = 42
    samples = bytearray(num_samples * 2)

    for i in range(num_samples):
        sample = 0.0
        if silence_lead <= i < num_samples - silence_tail:
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            noise = (seed / 0xFFFFFFFF) * 2.0 - 1.0
            t = i / sample_rate
            carrier = (
                0.55 * math.sin(2.0 * math.pi * 190.0 * t)
                + 0.30 * math.sin(2.0 * math.pi * 380.0 * t)
                + 0.15 * math.sin(2.0 * math.pi * 570.0 * t)
            )
            envelope = 0.5 + 0.5 * math.sin(2.0 * math.pi * 4.0 * t)
            sample = 0.42 * carrier * envelope + 0.12 * noise
        clamped = max(-1.0, min(1.0, sample))
        struct.pack_into("<h", samples, i * 2, round(clamped * 32767))

    data_size = len(samples)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + bytes(samples)


def _extract_multipart_field(raw_text: str, field_name: str) -> str | None:
    pattern = re.compile(
        rf'name="{re.escape(field_name)}"(?:;[^\r\n]*)?\r\n(?:[^\r\n]+\r\n)*\r\n([\s\S]*?)\r\n--'
    )
    match = pattern.search(raw_text)
    return match.group(1).strip() if match else None


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


RouteHandler = Callable[[RecordedRequest], tuple[int, dict[str, str], bytes | str | dict[str, Any]]]


class CleanRoomStubServer:
    """Extensible local loopback HTTP server for LLM, ASR, TTS, and custom routes."""

    def __init__(self, tts_wav_bytes: bytes | None = None) -> None:
        self.tts_wav_bytes = tts_wav_bytes or build_speech_wav_bytes(1.6, 24000)
        self.requests: list[RecordedRequest] = []
        self._custom_routes: dict[tuple[str, str], RouteHandler] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str = ""
        self.port: int = 0

    def add_route(self, method: str, path: str, handler: RouteHandler) -> None:
        """Register or override a custom `(method, path)` route on the stub server."""
        with self._lock:
            self._custom_routes[(method.upper(), path)] = handler

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _dispatch(self, method: str) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(length) if length > 0 else b""
                clean_path = self.path.split("?", 1)[0]
                recorded = RecordedRequest(
                    method=method.upper(),
                    path=clean_path,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=raw_body,
                )
                with outer._lock:
                    outer.requests.append(recorded)
                    custom = outer._custom_routes.get((method.upper(), clean_path))

                if custom is not None:
                    status, headers, payload = custom(recorded)
                    outer._send(self, status, headers, payload)
                    return

                status, headers, payload = outer._default_route(recorded)
                outer._send(self, status, headers, payload)

            def do_GET(self) -> None:
                self._dispatch("GET")

            def do_POST(self) -> None:
                self._dispatch("POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @staticmethod
    def _send(
        handler: BaseHTTPRequestHandler,
        status: int,
        headers: dict[str, str],
        payload: bytes | str | dict[str, Any],
    ) -> None:
        if isinstance(payload, dict):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
            content_type = "text/plain; charset=utf-8"
        else:
            data = payload
            content_type = "application/octet-stream"

        handler.send_response(status)
        if not any(k.lower() == "content-type" for k in headers):
            handler.send_header("Content-Type", content_type)
        for k, v in headers.items():
            handler.send_header(k, v)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _default_route(
        self, req: RecordedRequest
    ) -> tuple[int, dict[str, str], bytes | str | dict[str, Any]]:
        if req.method == "POST" and req.path.endswith("/chat/completions"):
            return 200, {}, self._handle_llm(req.body)

        if req.method == "GET" and req.path == "/asr/health":
            return (
                200,
                {},
                {
                    "status": "ok",
                    "ready": True,
                    "capabilities": {"timestamps": True, "external_segments": True},
                },
            )

        if req.method == "POST" and req.path == "/asr/job":
            return 200, {}, self._handle_asr_job(req.body)

        if req.method == "GET" and req.path == "/tts/health":
            return (
                200,
                {},
                {
                    "contract_version": 1,
                    "ready": True,
                    "plugin_id": "cleanroom-stub-tts",
                    "plugin_version": "1.0.0",
                },
            )

        if req.method == "POST" and req.path == "/tts/job":
            req_id = req.headers.get("x-request-id", "cleanroom-req")
            return (
                200,
                {
                    "Content-Type": "audio/wav",
                    "X-TTS-Contract-Version": "1",
                    "X-Request-ID": req_id,
                },
                self.tts_wav_bytes,
            )

        if req.method == "GET" and "/logs/" in req.path:
            return 200, {}, {"logs": [], "completed": True}

        return 404, {}, {"error": {"code": "NOT_FOUND", "message": req.path}}

    @staticmethod
    def _handle_llm(raw_body: bytes) -> dict[str, Any]:
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            body = {}
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        sys_msg = next((str(m.get("content", "")) for m in messages if m.get("role") == "system"), "")
        usr_msg = next((str(m.get("content", "")) for m in messages if m.get("role") == "user"), "").strip()

        reply = "前辈，今天我们一起回家吧，别想逃走哦。"
        if "只回复：好" in usr_msg:
            reply = "好"
        elif "剧情" in sys_msg or "摘要" in sys_msg:
            reply = "剧情摘要：放学后在社团教室里，后辈与前辈进行日常对话。"
        elif usr_msg.startswith("["):
            try:
                items = json.loads(usr_msg)
                if isinstance(items, list):
                    reply = json.dumps(
                        [
                            {
                                "slice_id": item.get("slice_id"),
                                "sentences": [
                                    "前辈，今天我们一起回家吧，外面已经没人了哦。"
                                    if idx % 2 == 0
                                    else "您已经逃不掉了呢，乖乖留下来陪我聊天吧。"
                                ],
                            }
                            for idx, item in enumerate(items)
                        ],
                        ensure_ascii=False,
                    )
            except Exception:
                pass
        else:
            indexed = []
            for line in usr_msg.splitlines():
                match = re.match(r"^(\d+)\t(.+)$", line)
                if match:
                    idx = int(match.group(1))
                    text = (
                        "前辈，今天我们一起回家吧，外面已经没人了哦。"
                        if idx % 2 == 0
                        else "您已经逃不掉了呢，乖乖留下来陪我聊天吧。"
                    )
                    indexed.append(f"{idx}\t{text}")
            if indexed:
                reply = "\n".join(indexed)

        return {
            "id": "chatcmpl-cleanroom",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
        }

    @staticmethod
    def _handle_asr_job(raw_body: bytes) -> dict[str, Any]:
        text = raw_body.decode("utf-8", errors="replace")
        seg_json = _extract_multipart_field(text, "segments") or "[]"
        lang = _extract_multipart_field(text, "language") or "ja"
        try:
            segments = json.loads(seg_json)
        except Exception:
            segments = []
        blocks = [
            {
                "index": s["index"],
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "text": (
                    "先輩、今日は一緒に帰りましょうか？"
                    if idx % 2 == 0
                    else "もう逃げられませんよ、先輩。"
                ),
            }
            for idx, s in enumerate(segments)
        ]
        return {
            "contract_version": 1,
            "stage": "asr",
            "language": lang,
            "source": "common-asr",
            "timeline_source": "external_segments",
            "blocks": blocks,
        }


@dataclass(frozen=True)
class UserFixtureProject:
    project_dir: Path
    source_mp3: Path
    source_lrc: Path


class PackagedSidecarTestCase(unittest.TestCase):
    """Base test case that executes the sidecar in an isolated clean-room sandbox."""

    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="asmr-cleanroom-ut-")
        self.addCleanup(self._temp_dir.cleanup)
        self.temp_root = Path(self._temp_dir.name).resolve()
        self.isolated_cwd = self.temp_root / "isolated_cwd"
        self.isolated_cwd.mkdir(parents=True, exist_ok=True)

        self.stub = CleanRoomStubServer()
        self.stub.start()
        self.addCleanup(self.stub.close)

        self.executable, self.script_prefix, self.ffmpeg_dir, self.is_packaged = (
            self._prepare_target()
        )

    def _prepare_target(self) -> tuple[Path, list[str], Path, bool]:
        packaged_env = os.environ.get("ASMR_PACKAGED_RESOURCES", "").strip()
        if packaged_env:
            resources_dir = Path(packaged_env).resolve()
            src_exe = resources_dir / "main.exe"
            if not src_exe.is_file():
                raise RuntimeError(f"ASMR_PACKAGED_RESOURCES does not contain main.exe: {src_exe}")
            staged = self.temp_root / "installed_app" / "resources"
            staged_ffmpeg = staged / "ffmpeg_bin"
            staged_ffmpeg.mkdir(parents=True, exist_ok=True)
            target_exe = staged / "main.exe"
            shutil.copy2(src_exe, target_exe)

            src_ffmpeg_dir = (
                resources_dir / "ffmpeg_bin"
                if (resources_dir / "ffmpeg_bin" / "ffmpeg.exe").is_file()
                else MODELS_ROOT / "ffmpeg_bin"
            )
            for name in ("ffmpeg.exe", "ffprobe.exe"):
                candidate = src_ffmpeg_dir / name
                if not candidate.is_file():
                    raise RuntimeError(f"Bundled {name} missing from {src_ffmpeg_dir}")
                shutil.copy2(candidate, staged_ffmpeg / name)
            return target_exe, [], staged_ffmpeg, True

        return Path(sys.executable), [str(MODELS_ROOT / "main.py")], MODELS_ROOT / "ffmpeg_bin", False

    @staticmethod
    def _minimal_system_path() -> str:
        if os.name == "nt":
            sys_root = os.environ.get("SystemRoot", r"C:\Windows")
            return os.pathsep.join([str(Path(sys_root) / "System32"), sys_root])
        return "/usr/bin:/bin"

    def run_sidecar(
        self,
        args: list[str],
        *,
        stdin: str | bytes | dict[str, Any] | None = None,
        inject_env: bool = False,
        extra_env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Spawn the sidecar in a clean-room environment with stripped PATH and temp cwd."""
        base_path = self._minimal_system_path()
        env = {
            "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
            "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
            "TEMP": str(self.temp_root),
            "TMP": str(self.temp_root),
            "USERPROFILE": str(self.temp_root),
            "HOME": str(self.temp_root),
            "LOCALAPPDATA": str(self.temp_root / "LocalAppData"),
            "APPDATA": str(self.temp_root / "AppData"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "ASMR_DISABLE_LOG_UPLOAD": "1",
            "PATH": (
                f"{self.ffmpeg_dir}{os.pathsep}{base_path}" if inject_env else base_path
            ),
        }
        if inject_env:
            env["ASMR_FFMPEG_DIR"] = str(self.ffmpeg_dir)
        if extra_env:
            env.update(extra_env)

        if isinstance(stdin, dict):
            input_text: str | None = json.dumps(stdin, ensure_ascii=False)
        elif isinstance(stdin, bytes):
            input_text = stdin.decode("utf-8")
        else:
            input_text = stdin

        cmd = [str(self.executable), *self.script_prefix, *[str(a) for a in args]]
        result = subprocess.run(
            cmd,
            cwd=str(self.isolated_cwd),
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if check and result.returncode != 0:
            self.fail(
                f"Sidecar exited with code {result.returncode} for args={args}\n"
                f"--- STDOUT ---\n{result.stdout[-3000:]}\n"
                f"--- STDERR ---\n{result.stderr[-3000:]}"
            )
        return result

    @staticmethod
    def extract_pipeline_result(stdout: str) -> dict[str, Any]:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("[PIPELINE_RESULT]"):
                return json.loads(line[len("[PIPELINE_RESULT]") :])
        raise AssertionError(f"No [PIPELINE_RESULT] found in stdout:\n{stdout[-2000:]}")

    def common_tts_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "protocol": "common-tts/1",
            "id": "cleanroom-stub-tts",
            "name": "Cleanroom Stub TTS",
            "version": "1.0.0",
            "capabilities": {"rate": True, "pitch": True, "voices": False},
            "parameters": {"type": "object", "additionalProperties": False},
            "execution": {"max_concurrent": 4, "timeout_seconds": 30},
        }

    def common_tts_kwargs(self) -> dict[str, Any]:
        return {
            "manifest": self.common_tts_manifest(),
            "base_url": f"{self.stub.base_url}/tts",
            "api_key": "cleanroom-tts-key",
            "parameters": {},
            "task_id": "cleanroom-task-1",
        }

    def llm_kwargs(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "base_url": f"{self.stub.base_url}/v1",
            "api_key": "cleanroom-llm-key",
            "model": "gpt-4o-mini",
        }

    def create_user_project(
        self,
        folder_name: str = "RJ01689510_テスト作品 [CV.テスト]/mp3",
        track_stem: str = "01導入 (Track 1)",
    ) -> UserFixtureProject:
        """Create a realistic non-ASCII project with ID3+cover MP3 and UTF-16 LE BOM LRC."""
        project_dir = self.temp_root / Path(folder_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        raw_wav = self.temp_root / "raw_source.wav"
        raw_wav.write_bytes(build_speech_wav_bytes(2.5, 24000))

        ffmpeg_bin = self.ffmpeg_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        source_mp3 = project_dir / f"{track_stem}.mp3"
        subprocess.run(
            [
                str(ffmpeg_bin),
                "-y",
                "-v",
                "error",
                "-i",
                str(raw_wav),
                "-ac",
                "2",
                "-ar",
                "44100",
                "-b:a",
                "192k",
                str(source_mp3),
            ],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        tags = id3.ID3()
        tags.add(id3.TIT2(encoding=3, text=["01導入"]))
        tags.add(id3.TPE1(encoding=3, text=["テスト声優一", "テスト声優二"]))
        tags.add(id3.TALB(encoding=3, text=["RJ01689510"]))
        tags.add(
            id3.APIC(
                encoding=3,
                mime="image/png",
                type=3,
                desc="Cover",
                data=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
            )
        )
        tags.save(source_mp3)

        source_lrc = project_dir / f"{track_stem}.lrc"
        lrc_text = (
            "\ufeff[00:00.20]先輩、今日は一緒に帰りましょうか？\r\n"
            "[00:01.00]もう逃げられませんよ、先輩。\r\n"
            "[00:02.20]\r\n"
        )
        source_lrc.write_bytes(lrc_text.encode("utf-16-le"))
        return UserFixtureProject(project_dir=project_dir, source_mp3=source_mp3, source_lrc=source_lrc)
