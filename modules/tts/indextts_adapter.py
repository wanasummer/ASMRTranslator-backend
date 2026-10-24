"""IndexTTS 2.5 adapter for a remote Gradio service.

The official IndexTTS 2.5 demo exposes ``gen_single`` through Gradio.  A voice
is selected by uploading a reference recording; there is no static voice ID.
"""

import asyncio
import copy
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urljoin, urlparse, urlunparse
from uuid import uuid4

import httpx

from ..network import NetworkPolicy, sync_client

from ..error import TTSError
from ..io import open_write
from .base import TTSAdapter
from .models import VoiceInfo, TTSSegmentContext
from .indextts_options import EMOTION_METHODS, IndexTTSOptions


_request_lock = threading.Lock()
_upload_cache: dict[tuple[str, str, int, int], dict] = {}


class IndexTTSAdapter(TTSAdapter):
    """IndexTTS 2.5 Gradio backend.

    Requests are serialized process-wide because the project processes several
    subtitle chunks in parallel while a typical IndexTTS server has one GPU.
    """

    _DEFAULT_URL = "http://127.0.0.1:7860"
    _SAME_EMOTION = "与音色参考音频相同"

    def __init__(
        self,
        voice_map: Optional[dict[str, str]] = None,
        base_url: str = _DEFAULT_URL,
        reference_audio: Optional[str] = None,
        language: str = "ZH",
        emotion_method: str = _SAME_EMOTION,
        emotion_reference: Optional[str] = None,
        emotion_weight: float = 0.65,
        duration_factor: float = 1.0,
        timeout: float = 1800.0,
        max_retries: int = 1,
        options: Optional[dict] = None,
    ):
        super().__init__(voice_map)
        self._network_policy = NetworkPolicy()
        self._base_url = (base_url or self._DEFAULT_URL).rstrip("/")
        self._generation_endpoint: Optional[str] = None
        self._queue_supported = True
        self._session_hash = uuid4().hex
        self._segment_bound = False
        self._reference_audio = reference_audio or os.environ.get(
            "INDEXTTS_REFERENCE_AUDIO", ""
        )
        self._language = language
        self._emotion_method = emotion_method
        self._emotion_reference = emotion_reference
        self._emotion_weight = max(0.0, min(1.0, float(emotion_weight)))
        self._duration_factor = max(0.5, min(2.0, float(duration_factor)))
        self._timeout = float(timeout)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("IndexTTS timeout 必须为有限正数")
        self._max_retries = max(1, int(max_retries))
        # Preserve the legacy constructor; desktop requests use the typed options.
        self._options = IndexTTSOptions.from_dict(options) if options is not None else None
        if self._options is not None:
            self._emotion_method = EMOTION_METHODS[self._options.emotion_mode]
            if self._options.voice_source == "segment":
                self._reference_audio = ""
                if self._options.emotion_mode == "segment":
                    self._emotion_method = self._SAME_EMOTION
            self._emotion_reference = (
                self._options.emotion_reference if self._options.emotion_mode == "reference" else None
            )
            self._language = self._options.language
            self._duration_factor = self._options.duration_factor
        if not math.isfinite(float(emotion_weight)) or not 0 <= float(emotion_weight) <= 1:
            raise ValueError("IndexTTS 情感权重必须是 0～1 的有限数值")

    def for_segment(self, context: TTSSegmentContext) -> "IndexTTSAdapter":
        if self._options is None or (self._options.emotion_mode != "segment"
                                    and self._options.voice_source != "segment"):
            return self
        if not context.speech_file or not os.path.isfile(context.speech_file):
            raise TTSError(f"IndexTTS 第 {context.index + 1} 段缺少原声语音情感参考")
        try:
            with wave.open(context.speech_file, "rb") as source:
                if source.getnframes() == 0:
                    raise TTSError(f"IndexTTS 第 {context.index + 1} 段原声情感参考为空")
        except (wave.Error, EOFError, OSError) as exc:
            raise TTSError(f"IndexTTS 第 {context.index + 1} 段原声参考 WAV 无效") from exc
        adapter = copy.copy(self)
        adapter._segment_bound = True
        if self._options.voice_source == "segment":
            adapter._reference_audio = context.speech_file
        if self._options.emotion_mode == "segment":
            adapter._emotion_reference = (
                context.speech_file if self._options.voice_source == "reference" else None
            )
        return adapter

    def cache_inputs(self) -> dict:
        from dataclasses import asdict
        generation = asdict(self._options) if self._options else {}
        # Credential redaction excludes keys containing "token". Keep these
        # non-secret model limits in the synthesis cache under explicit aliases.
        for original, alias in (("max_mel_tokens", "mel_limit"),
                                ("max_text_tokens_per_segment", "text_limit")):
            if original in generation:
                generation[alias] = generation.pop(original)
        return {"voice_reference": self._reference_audio,
                "emotion_reference": self._emotion_reference,
                "generation": generation,
                "emotion_method": self._emotion_method,
                "emotion_weight": self._emotion_weight}

    @property
    def name(self) -> str:
        return "indextts-2.5"

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        del voice, pitch  # IndexTTS selects a voice from the reference recording.
        return await asyncio.to_thread(
            self._synthesize_blocking, text, output_path, rate
        )

    def _synthesize_blocking(self, text: str, output_path: str, rate: str) -> str:
        if self._options and (self._options.voice_source == "segment" or self._options.emotion_mode == "segment") and not self._segment_bound:
            raise TTSError("IndexTTS 自动跟随需要分段原声上下文")
        if not self._reference_audio:
            raise TTSError("IndexTTS 2.5 需要配置音色参考音频")
        if not os.path.isfile(self._reference_audio):
            raise TTSError(f"IndexTTS 音色参考音频不存在: {self._reference_audio}")
        if self._options and self._options.emotion_mode == "segment" and self._options.voice_source == "reference" and not self._emotion_reference:
            raise TTSError("IndexTTS 逐段情感模式需要分段原声上下文")
        if self._emotion_reference and not os.path.isfile(self._emotion_reference):
            raise TTSError(f"IndexTTS 情感参考音频不存在: {self._emotion_reference}")
        if self._options:
            print(f"       [indextts] mode={self._options.emotion_mode}, "
                  f"voice={self._reference_audio}, emotion={self._emotion_reference or '音色参考'}")

        last_error: Optional[Exception] = None
        operation = "上传参考音频"
        for attempt in range(self._max_retries):
            try:
                with _request_lock:
                    with sync_client(client_factory=httpx.Client, network_policy=self._network_policy, timeout=self._timeout) as client:
                        operation = "上传参考音频"
                        prompt = self._upload(client, self._reference_audio)
                        emotion_ref = (
                            self._upload(client, self._emotion_reference)
                            if self._emotion_reference else None
                        )
                        if self._options and self._options.emotion_mode == "text":
                            operation = "启用情感描述文本"
                            self._prepare_text_mode(client)
                        operation = "提交/等待合成"
                        result = self._generate(
                            client,
                            self._request_data(text, prompt, emotion_ref, rate),
                        )
                        file_data = (result.get("data") or [None])[0]
                        if isinstance(file_data, dict) and isinstance(
                            file_data.get("value"), dict
                        ):
                            file_data = file_data["value"]
                        if not isinstance(file_data, dict):
                            raise TTSError(
                                f"IndexTTS 未返回音频: {str(result)[:500]}"
                            )
                        audio_url = file_data.get("url")
                        if not audio_url:
                            remote_path = file_data.get("path")
                            if not remote_path:
                                raise TTSError("IndexTTS 返回的音频缺少 url/path")
                            audio_url = (
                                f"{self._base_url}/gradio_api/file={remote_path}"
                            )
                        operation = "下载生成音频"
                        audio = self._download_audio(
                            client, self._asset_url(str(audio_url))
                        )
                        return self._save_audio(audio.content, output_path)
            except httpx.TimeoutException as exc:
                # The server may still be running an accepted generation request.
                raise TTSError(
                    f"IndexTTS 2.5 {operation}超时（任务等待上限 {self._timeout:g} 秒，"
                    f"{type(exc).__name__}，网络无响应可能提前超时）；"
                    "请检查服务端日志、GPU 状态和任务队列"
                ) from exc
            except (httpx.TransportError, TTSError) as exc:
                last_error = exc
                if operation == "提交/等待合成":
                    break  # Do not replay an ambiguously accepted generation.
                if attempt + 1 >= self._max_retries:
                    break
        raise TTSError(f"IndexTTS 2.5 合成失败: {last_error}") from last_error

    def _generate(self, client: httpx.Client, data: list) -> dict:
        """Submit once and wait for Gradio completion; legacy routes use /run."""
        if not self._queue_supported:
            response = self._post_generation(client, data)
            self._raise_for_status(response, "合成")
            return response.json()
        initial_url = self._generation_url().replace("/run/", "/call/", 1)
        response = client.post(initial_url, json=self._payload(data))
        call_url = initial_url
        if response.status_code == 404:
            for candidate in self._discover_generation_urls(client):
                if "/run/" not in candidate:
                    continue
                candidate = candidate.replace("/run/", "/call/", 1)
                if candidate == initial_url:
                    continue
                response = client.post(candidate, json=self._payload(data))
                call_url = candidate
                if response.status_code != 404:
                    break
        if response.status_code == 404:
            self._queue_supported = False
            response = self._post_generation(client, data)
            self._raise_for_status(response, "合成")
            return response.json()
        self._raise_for_status(response, "提交合成任务")
        try:
            event_id = response.json().get("event_id")
        except (ValueError, AttributeError) as exc:
            raise TTSError("IndexTTS 合成任务响应无法识别") from exc
        if not isinstance(event_id, str) or not event_id:
            raise TTSError("IndexTTS 未返回合成任务 ID")
        self._generation_endpoint = call_url.replace("/call/", "/run/", 1)
        return self._wait_generation(client, call_url, event_id)

    def _payload(self, data: list) -> dict:
        payload = {"data": data}
        if self._options and self._options.emotion_mode == "text":
            payload["session_hash"] = self._session_hash
        return payload

    def _prepare_text_mode(self, client: httpx.Client) -> None:
        """Enable the official experimental Radio choice in this Gradio session.

        The UI hides mode 3 by default. Calling gen_single directly with its
        label otherwise fails Radio preprocessing, even with QwenEmotion loaded.
        """
        endpoint = f"{self._base_url}/gradio_api/call/on_experimental_change"
        payload = self._payload([True, self._SAME_EMOTION])
        response = client.post(endpoint, json=payload)
        if response.status_code == 404:
            endpoint = endpoint.replace("/call/", "/run/")
            response = client.post(endpoint, json=payload)
            if response.status_code == 404:
                endpoint = f"{self._base_url}/run/on_experimental_change"
                response = client.post(endpoint, json=payload)
            self._raise_for_status(response, "启用情感描述文本")
            result = response.json()
        else:
            self._raise_for_status(response, "启用情感描述文本")
            event_id = response.json().get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise TTSError("IndexTTS 情感描述控制未返回任务 ID")
            result = self._wait_generation(client, endpoint, event_id)
        outputs = result.get("data") or []
        update = outputs[0] if outputs else None
        choices = update.get("choices", []) if isinstance(update, dict) else []
        if len(choices) < 4:
            raise TTSError("IndexTTS 服务未启用情感描述文本，请在服务端加载 QwenEmotion（--qwen_emo）")

    def _wait_generation(
        self, client: httpx.Client, call_url: str, event_id: str
    ) -> dict:
        """Ignore progress/heartbeats but enforce a finite overall job deadline."""
        deadline = time.monotonic() + self._timeout
        event = ""
        with client.stream(
            "GET", f"{call_url}/{quote(event_id, safe='')}",
            timeout=httpx.Timeout(self._timeout, read=min(60.0, self._timeout)),
        ) as response:
            if response.status_code != 200:
                response.read()
                self._raise_for_status(response, "等待合成任务")
            for line in response.iter_lines():
                if time.monotonic() >= deadline:
                    raise httpx.ReadTimeout("generation deadline exceeded")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event in {"complete", "error"}:
                    if event == "error":
                        raise TTSError(f"IndexTTS 服务端合成失败: {line[5:].strip()[:500]}")
                    try:
                        outputs = json.loads(line[5:].strip())
                    except ValueError as exc:
                        raise TTSError("IndexTTS 合成结果无法识别") from exc
                    if not isinstance(outputs, list):
                        raise TTSError("IndexTTS 合成结果必须为输出列表")
                    return {"data": outputs}
        raise TTSError("IndexTTS 合成结果连接已结束，但未收到完成事件")

    def _generation_url(self) -> str:
        """Return the standard endpoint, or one discovered from Gradio config."""
        return self._generation_endpoint or (
            f"{self._base_url}/gradio_api/run/gen_single"
        )

    def _post_generation(
        self, client: httpx.Client, data: list
    ) -> httpx.Response:
        payload = self._payload(data)
        initial_url = self._generation_url()
        response = client.post(initial_url, json=payload)
        if response.status_code != 404:
            return response

        for candidate in self._discover_generation_urls(client):
            if candidate == initial_url:
                continue
            response = client.post(candidate, json=payload)
            if response.status_code != 404:
                self._generation_endpoint = candidate
                return response
        return response

    def _discover_generation_urls(self, client: httpx.Client) -> list[str]:
        """Find the audio-producing event without exposing Gradio details to users."""
        try:
            response = client.get(f"{self._base_url}/config")
            if response.status_code != 200:
                return []
            config = response.json()
        except (httpx.TransportError, ValueError, TypeError):
            return []
        if not isinstance(config, dict):
            return []

        component_types = {
            component.get("id"): str(component.get("type", "")).lower()
            for component in config.get("components", [])
            if isinstance(component, dict)
        }
        candidates: list[tuple[int, str]] = []
        for dependency in config.get("dependencies", []):
            if not isinstance(dependency, dict):
                continue
            api_name = dependency.get("api_name")
            if not isinstance(api_name, str) or not api_name.strip():
                continue
            outputs = dependency.get("outputs", [])
            if not isinstance(outputs, list):
                continue
            produces_audio = any(
                component_types.get(output_id) == "audio" for output_id in outputs
            )
            normalized = api_name.strip().strip("/")
            score = 100 if normalized == "gen_single" else 0
            score += 50 if produces_audio else 0
            score += 10 if "gen" in normalized.lower() else 0
            if score:
                candidates.append((score, normalized))
        if not candidates:
            return []

        api_name = max(candidates)[1]
        encoded = quote(api_name, safe="")
        api_prefix = str(config.get("api_prefix") or "/gradio_api").rstrip("/")
        if not api_prefix.startswith("/"):
            api_prefix = f"/{api_prefix}"
        return [
            f"{self._base_url}{api_prefix}/run/{encoded}",
            f"{self._base_url}/run/{encoded}",
            f"{self._base_url}/api/{encoded}",
        ]

    def _asset_url(self, asset_url: str) -> str:
        """Keep Gradio file URLs on the configured LAN host."""
        resolved = urljoin(self._base_url + "/", asset_url)
        parsed = urlparse(resolved)
        if parsed.hostname not in {"127.0.0.1", "localhost", "0.0.0.0"}:
            return resolved
        configured = urlparse(self._base_url)
        if configured.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
            return resolved
        return urlunparse(parsed._replace(
            scheme=configured.scheme,
            netloc=configured.netloc,
        ))

    def _download_audio(
        self, client: httpx.Client, audio_url: str
    ) -> httpx.Response:
        """Retry a Gradio file response interrupted by a stale Content-Length."""
        last_error: Optional[Exception] = None
        for _ in range(3):
            try:
                response = client.get(
                    audio_url,
                    headers={"Accept-Encoding": "identity"},
                )
                self._raise_for_status(response, "下载生成音频")
                return response
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as exc:
                last_error = exc
        raise TTSError(f"IndexTTS 下载生成音频失败: {last_error}") from last_error

    def _upload(self, client: httpx.Client, path: str) -> dict:
        resolved = str(Path(path).resolve())
        stat = os.stat(resolved)
        key = (self._base_url, resolved, stat.st_size, stat.st_mtime_ns)
        cached = _upload_cache.get(key)
        if cached:
            return cached

        with open(resolved, "rb") as source:
            response = client.post(
                f"{self._base_url}/gradio_api/upload",
                files={"files": (Path(resolved).name, source, "application/octet-stream")},
            )
        self._raise_for_status(response, "上传参考音频")
        payload = response.json()
        remote_path = payload[0] if isinstance(payload, list) and payload else None
        if isinstance(payload, dict):
            remote_path = (payload.get("files") or [None])[0]
        if not remote_path:
            raise TTSError(f"IndexTTS 上传响应无法识别: {str(payload)[:500]}")
        file_data = {
            "path": remote_path,
            "orig_name": Path(resolved).name,
            "meta": {"_type": "gradio.FileData"},
        }
        _upload_cache[key] = file_data
        return file_data

    def _request_data(
        self, text: str, prompt: dict, emotion_ref: Optional[dict], rate: str
    ) -> list:
        duration = self._duration_factor
        try:
            speed = 1.0 + float(rate.replace("%", "")) / 100.0
            if speed > 0:
                duration = max(0.5, min(2.0, duration / speed))
        except (AttributeError, ValueError):
            pass
        options = self._options or IndexTTSOptions(emotion_mode="same")
        mode = options.emotion_mode
        vector = options.emotion_vector if mode == "vector" else (0.0,) * 8
        weight = 1.0 if self._emotion_method == self._SAME_EMOTION else self._emotion_weight
        return [
            self._emotion_method, prompt, text, self._language, emotion_ref,
            weight, *vector,
            options.emotion_text if mode == "text" else "",
            options.emotion_random if mode in ("vector", "text") else False,
            options.max_text_tokens_per_segment, duration,
            options.do_sample, options.top_p if options.do_sample else 1.0,
            options.top_k if options.do_sample else 0,
            options.temperature if options.do_sample else 1.0,
            options.length_penalty if options.num_beams > 1 else 0.0,
            options.num_beams, options.repetition_penalty, options.max_mel_tokens,
        ]

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.status_code != 200:
            raise TTSError(
                f"IndexTTS {operation}返回 HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

    def _save_audio(self, audio: bytes, output_path: str) -> str:
        if len(audio) < 200:
            raise TTSError(f"IndexTTS 只返回了 {len(audio)} 字节音频")
        is_wav = audio[:4] == b"RIFF" and Path(output_path).suffix.lower() != ".wav"
        if not is_wav:
            with open_write(output_path, "wb") as target:
                target.write(audio)
            return output_path

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open_write(temp_path, "wb") as target:
                target.write(audio)
            ffmpeg = Path(__file__).parent.parent.parent / "ffmpeg_bin" / "ffmpeg.exe"
            executable = str(ffmpeg) if ffmpeg.exists() else "ffmpeg"
            result = subprocess.run(
                [executable, "-y", "-i", temp_path, output_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise TTSError(f"IndexTTS 音频转码失败: {result.stderr[-500:]}")
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return output_path

    def get_available_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(
            id="reference",
            name="参考音频音色",
            language=self._language,
            description="由配置的音色参考音频克隆",
        )]
