"""Normalized WAV transport and strict timeline validation for analysis services."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterator, TypedDict
from urllib.parse import urlsplit
from uuid import uuid4
import wave

import httpx

from .audio_splitter import get_ffmpeg_path
from .network import NetworkPolicy, sync_client


class TimelineSpan(TypedDict):
    start_ms: int
    end_ms: int


class WordSpan(TimelineSpan):
    text: str


class AnalysisPluginError(RuntimeError):
    """An analysis service failed or violated its timeline contract."""

    def __init__(self, message, *, code="ANALYSIS_CONTRACT_INVALID"):
        self.code = code
        super().__init__(message)


def validate_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    url = urlsplit(endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("分析插件地址必须是无凭据的 HTTP(S) 服务地址")
    port = url.port
    if url.hostname == "0.0.0.0":
        return url._replace(netloc="127.0.0.1" + (f":{port}" if port else "")).geturl()
    return endpoint


@contextmanager
def normalized_audio(audio_path: str | Path, start_ms: int | None = None,
                     end_ms: int | None = None) -> Iterator[tuple[Path, int]]:
    """Decode with the same ffmpeg as the pipeline, cropping before upload."""
    with tempfile.TemporaryDirectory(prefix="asmr-analysis-") as directory:
        target = Path(directory) / "audio.wav"
        command = [get_ffmpeg_path(), "-nostdin", "-v", "error", "-y"]
        if start_ms is not None:
            command += ["-ss", str(start_ms / 1000)]
        command += ["-i", str(audio_path)]
        if end_ms is not None:
            if start_ms is None or end_ms <= start_ms:
                raise AnalysisPluginError("Invalid analysis audio crop")
            command += ["-t", str((end_ms - start_ms) / 1000)]
        command += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
        import os
        subprocess.run(command, check=True, capture_output=True, timeout=600,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with wave.open(str(target), "rb") as audio:
            duration_ms = round(audio.getnframes() / 16)
        yield target, duration_ms


class AnalysisPluginClient:
    def __init__(self, endpoint: str, service: str, api_version: str, *, client_factory=None):
        self.endpoint = validate_endpoint(endpoint)
        self.service, self.api_version = service, api_version
        self.client_factory = client_factory
        self.policy = NetworkPolicy()

    def request(self, route: str, audio_path: str | Path, data: dict[str, Any], *,
                start_ms: int | None = None, end_ms: int | None = None) -> tuple[dict[str, Any], int]:
        try:
            with normalized_audio(audio_path, start_ms, end_ms) as (wav, duration):
                request_id = uuid4().hex
                with sync_client(client_factory=self.client_factory, network_policy=self.policy, timeout=21600) as client:
                    with wav.open("rb") as audio:
                        response = client.post(self.endpoint + route, data={**data, "request_id": request_id},
                                               files={"audio_file": ("audio.wav", audio, "audio/wav")})
                response.raise_for_status()
                body = response.json()
                if (body.get("service") != self.service or body.get("api_version") != self.api_version
                        or body.get("request_id") != request_id or type(body.get("duration_ms")) is not int
                        or abs(body["duration_ms"] - duration) > 1):
                    raise AnalysisPluginError("Analysis response identity or duration mismatch")
                return body, duration
        except AnalysisPluginError:
            raise
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            detail = ""
            try:
                payload = error.response.json()
                if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
                    detail = "：" + payload["detail"].replace("\n", " ")[:300]
            except ValueError:
                pass
            raise AnalysisPluginError(f"{self.service} 请求失败（HTTP {status}）{detail}",
                                      code=f"ANALYSIS_HTTP_{status}") from error
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            code = f"ANALYSIS_HTTP_{status}" if status else "ANALYSIS_REQUEST_FAILED"
            raise AnalysisPluginError(f"{self.service} 请求失败（{type(error).__name__}）", code=code) from error


def validate_timeline(body: dict[str, Any], key: str, duration_ms: int) -> list[dict[str, Any]]:
    rows = body.get(key)
    if not isinstance(rows, list):
        raise AnalysisPluginError("Analysis response is missing its timeline")
    previous = 0
    for row in rows:
        if not isinstance(row, dict):
            raise AnalysisPluginError("Invalid analysis timeline row")
        start, end = row.get("start_ms"), row.get("end_ms")
        if type(start) is not int or type(end) is not int or not previous <= start < end <= duration_ms:
            raise AnalysisPluginError("Invalid or overlapping analysis timeline")
        previous = end
    return rows
