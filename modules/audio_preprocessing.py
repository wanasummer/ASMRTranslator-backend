"""Optional analysis-only audio preparation; the source timeline stays authoritative."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Protocol
from uuid import uuid4
from contextlib import contextmanager
import os
import subprocess
import threading
import wave

from .analysis_plugins import validate_endpoint
from .audio_splitter import get_ffmpeg_path
from .network import NetworkPolicy, sync_client
from .storage import save_json
from .task_cache import SqliteResultCache, cache_key, file_digest


class AudioPreparationError(RuntimeError):
    """An analysis audio adapter failed or changed the source timeline."""


@dataclass(frozen=True)
class AnalysisAudio:
    path: str
    identity: str


class AudioPreprocessor(Protocol):
    def prepare(self, source: str, output: Path) -> AnalysisAudio: ...


class OriginalAudio:
    def prepare(self, source: str, output: Path) -> AnalysisAudio:
        return AnalysisAudio(source, "original")


class UvrAudio:
    def __init__(self, endpoint: str, *, client_factory=None,
                 progress: Callable[[str, int, int], None] | None = None, poll_interval: float = 1.0):
        self.endpoint = validate_endpoint(endpoint)
        self.client_factory = client_factory
        self.policy = NetworkPolicy()
        self.progress = progress or (lambda label, current, total: None)
        self.poll_interval = poll_interval

    @contextmanager
    def monitor(self, client, request_id: str, total: int, supported: bool) -> Iterator[threading.Event]:
        stop = threading.Event()
        def poll():
            previous = None
            while not stop.is_set():
                try:
                    response = client.get(self.endpoint + "/progress", params={"request_id": request_id}, timeout=5)
                    if response.status_code == 404:
                        stop.wait(self.poll_interval)
                        continue
                    response.raise_for_status()
                    row = response.json()
                    current = row.get("current")
                    if (row.get("service") != "uvr" or row.get("api_version") != "uvr/1"
                            or row.get("request_id") != request_id or type(current) is not int
                            or type(row.get("total")) is not int or not 0 <= current <= row["total"]
                            or row["total"] not in (0, total)):
                        raise ValueError("Invalid UVR progress contract")
                    status = row.get("status")
                    if status == "queued":
                        label = "UVR 服务排队中"
                    elif status in {"running", "completed"}:
                        def stamp(frames):
                            seconds = frames // 44100
                            return f"{seconds // 60:02d}:{seconds % 60:02d}"
                        label = f"UVR 正在分离人声 · {stamp(current)} / {stamp(total)}"
                    elif status == "failed":
                        label = "UVR 分离失败，正在读取服务错误"
                    elif status == "preparing":
                        label = "UVR 正在读取音频"
                    else:
                        raise ValueError("Unknown UVR progress state")
                    event = (label, max(0, min(current, total - 1)), total)
                    if event != previous and not stop.is_set():
                        self.progress(*event)
                        previous = event
                except Exception:
                    event = ("UVR 正在处理，暂时无法获取进度", previous[1], previous[2]) if previous else ("UVR 正在处理，暂时无法获取进度", 0, 0)
                    if event != previous and not stop.is_set():
                        self.progress(*event)
                        previous = event
                stop.wait(self.poll_interval)
        worker = threading.Thread(target=poll, daemon=True, name="uvr-progress") if supported else None
        if worker:
            worker.start()
        try:
            yield stop
        finally:
            stop.set()
            if worker:
                worker.join(timeout=6)

    def prepare(self, source: str, output: Path) -> AnalysisAudio:
        self.progress("正在准备 UVR 分离", 0, 0)
        cache = SqliteResultCache(str(output), "analysis_audio")
        try:
            with sync_client(client_factory=self.client_factory, network_policy=self.policy, timeout=21600) as client:
                health = client.get(self.endpoint + "/health")
                health.raise_for_status()
                info = health.json()
                if (info.get("service") != "uvr" or info.get("api_version") != "uvr/1"
                        or info.get("ready") is not True or not info.get("model_sha256") or not info.get("version")):
                    raise AudioPreparationError("UVR service is not ready or has an incompatible contract")
                identity = cache_key({"policy": 1, "source": file_digest(source), "endpoint": self.endpoint,
                                      "version": info["version"], "model": info["model_sha256"]})
                target = output / "analysis_audio" / f"uvr_{identity}.wav"
                cached = cache.load(identity)
                if cached and target.is_file() and file_digest(target) == cached.get("sha256"):
                    print(f"[audio-preparation] UVR cache: {target}")
                    self.progress("已复用 UVR 人声缓存", 1, 1)
                    return AnalysisAudio(str(target), identity)
                target.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=target.parent) as temporary:
                    normalized, pending = Path(temporary) / "source.wav", Path(temporary) / "vocals.wav"
                    self.progress("正在转换 UVR 输入音频", 0, 0)
                    subprocess.run([get_ffmpeg_path(), "-nostdin", "-v", "error", "-y", "-i", source,
                                    "-vn", "-map_metadata", "-1", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(normalized)],
                                   check=True, capture_output=True, timeout=600,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    with wave.open(str(normalized)) as original:
                        expected = original.getnframes()
                    request_id = uuid4().hex
                    supported = "progress/1" in info.get("capabilities", [])
                    self.progress("正在上传音频并等待 UVR 分离" if supported else "UVR 正在处理（当前服务不提供实时进度）", 0, 0)
                    with self.monitor(client, request_id, expected, supported) as monitor_stop, normalized.open("rb") as stream, client.stream("POST", self.endpoint + "/separate",
                            data={"request_id": request_id}, files={"audio_file": ("audio.wav", stream, "audio/wav")}) as response:
                        response.raise_for_status()
                        if (response.headers.get("X-Request-ID") != request_id or
                                response.headers.get("X-UVR-API") != "uvr/1"):
                            raise AudioPreparationError("UVR response identity mismatch")
                        monitor_stop.set()
                        size = 0
                        self.progress("正在下载 UVR 人声音轨", max(0, expected - 1), expected)
                        with pending.open("wb") as writer:
                            for chunk in response.iter_bytes():
                                size += len(chunk)
                                if size > normalized.stat().st_size + 65536:
                                    raise AudioPreparationError("UVR response exceeds the expected audio size")
                                writer.write(chunk)
                    with wave.open(str(pending)) as vocals:
                        if ((vocals.getframerate(), vocals.getnchannels(), vocals.getsampwidth()) != (44100, 2, 2)
                                or vocals.getnframes() != expected):
                            raise AudioPreparationError("UVR changed the source audio timeline")
                        actual = sum(len(chunk) for chunk in iter(lambda: vocals.readframes(44100), b""))
                        if actual != expected * 4:
                            raise AudioPreparationError("UVR returned truncated audio")
                    os.replace(pending, target)
                cache.save(identity, {"sha256": file_digest(target)})
                save_json(str(output / "analysis_audio/audit.json"), {"policy": 1, "identity": identity,
                          "path": str(target), "source_sha256": file_digest(source), "service": info})
                print(f"[audio-preparation] UVR analysis audio: {target}")
                self.progress("UVR 人声分离完成", 1, 1)
                return AnalysisAudio(str(target), identity)
        except AudioPreparationError:
            raise
        except Exception as error:
            raise AudioPreparationError(f"UVR 分析音轨处理失败（{type(error).__name__}），请检查服务") from error


_PREPROCESSORS = {"original": OriginalAudio, "uvr": UvrAudio}


def create_audio_preprocessor(name: str, **kwargs) -> AudioPreprocessor:
    try:
        adapter = _PREPROCESSORS[name]
    except KeyError as error:
        raise AudioPreparationError(f"Unknown audio preprocessor: {name}") from error
    return adapter(**kwargs)
