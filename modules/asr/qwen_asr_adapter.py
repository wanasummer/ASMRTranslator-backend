"""Alibaba Cloud Bailian Qwen Audio 3.1 ASR Flash and Fun-ASR adapters.

Qwen Audio accepts a Base64 data URL synchronously. Stable Fun-ASR accepts URLs
only, so local VAD slices are uploaded to QwenCloud temporary OSS storage,
submitted as asynchronous transcription jobs, and polled to completion.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import wave
from pathlib import Path
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import ASRError
from .base import ASRAdapter
from .cloud_audio import audio_data_url, wav_duration
from .models import ASRResult, ASRSegment
from .no_speech import ASRNoSpeechError, response_has_no_speech, is_no_speech


_UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


def _workspace_base_url(workspace_id: str) -> str:
    workspace = workspace_id.strip()
    if not workspace:
        return ""
    if any(character in workspace for character in "/\\.:?#"):
        raise ASRError("百炼 Workspace ID 格式无效")
    return f"https://{workspace}.cn-beijing.maas.aliyuncs.com/api/v1"


def _silent_wav_data_url() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class QwenAudio31ASRFlashAdapter(ASRAdapter):
    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        language: str = "ja",
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **_ignored,
    ) -> None:
        self.api_key = (api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
        self.workspace_id = (workspace_id or os.environ.get("DASHSCOPE_WORKSPACE_ID", "")).strip()
        self.language = language or "ja"
        self.base_url = (base_url or _workspace_base_url(self.workspace_id)).rstrip("/")
        self._timeout = float(timeout)
        self._network_policy = NetworkPolicy()

    @property
    def name(self) -> str:
        return "qwen-audio-3.1-asr-flash"

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    async def check_connection(self, client: Optional[httpx.AsyncClient] = None) -> None:
        """Validate credentials and the workspace-specific domestic endpoint."""
        if not self.api_key:
            raise ASRError("千问 ASR API Key 未配置")
        if not self.base_url:
            raise ASRError("百炼 Workspace ID 未配置")
        payload = {
            "model": "qwen-audio-3.1-asr-flash",
            "input": {"messages": [{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": _silent_wav_data_url()},
                }],
            }]},
            "parameters": {"format": "wav", "sample_rate": 16000},
        }
        for attempt in range(2):
            try:
                response = await self._post_probe(payload, client)
                self._raise_for_probe(response)
                return
            except httpx.ConnectTimeout as exc:
                if attempt:
                    raise ASRError(
                        "千问 ASR 连接北京业务空间接口超时（已重试一次）。"
                        "请检查网络或代理连接，稍后再试"
                    ) from exc
                await asyncio.sleep(0.5)

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        if not self.api_key:
            raise ASRError("千问 ASR API Key 未配置")
        if not self.base_url:
            raise ASRError("百炼 Workspace ID 未配置")
        payload = {
            "model": "qwen-audio-3.1-asr-flash",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {"data": audio_data_url(audio_path)},
                    }],
                }],
            },
            "parameters": {"format": "wav", "sample_rate": 16000},
        }
        response = await self._post(payload, client)
        duration = wav_duration(audio_path)
        if response_has_no_speech(response):
            return ASRResult(
                text="",
                duration=duration,
                language_code=language or self.language,
                segments=[],
            )
        text = self._response_text(response.json())
        segments = [ASRSegment(text=text, start=0.0, end=duration)] if text and timestamps else []
        return ASRResult(text=text, duration=duration, language_code=language or self.language, segments=segments)

    async def _post(self, payload: dict, client: Optional[httpx.AsyncClient]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }
        if client is not None:
            response = await client.post(
                f"{self.base_url}/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=payload,
            )
        else:
            async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self._timeout) as own_client:
                response = await own_client.post(
                    f"{self.base_url}/services/aigc/multimodal-generation/generation",
                    headers=headers,
                    json=payload,
                )
        if not response_has_no_speech(response):
            self._raise_for_status(response)
        return response

    async def _post_probe(self, payload: dict, client: Optional[httpx.AsyncClient]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        if client is not None:
            return await client.post(url, headers=headers, json=payload)
        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=httpx.Timeout(self._timeout, connect=min(15.0, self._timeout))) as own_client:
            return await own_client.post(url, headers=headers, json=payload)

    @staticmethod
    def _raise_for_probe(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response_has_no_speech(response):
            return
        QwenAudio31ASRFlashAdapter._raise_for_status(response)

    @staticmethod
    def _is_no_words_response(response: httpx.Response) -> bool:
        return response_has_no_speech(response)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == 401:
            raise ASRError("千问 ASR API Key 无效")
        if response.status_code == 403:
            try:
                error = response.json()
            except ValueError:
                error = {}
            if error.get("code") == "AccessDenied.Unpurchased":
                raise ASRError("百炼 ASR 模型尚未开通，请先在当前工作空间获得模型访问资格")
            raise ASRError(f"百炼 ASR 无访问权限：{error.get('message') or response.text[:300]}")
        raise ASRError(f"千问 ASR 请求失败（HTTP {response.status_code}）：{response.text[:300]}")

    @staticmethod
    def _response_text(payload: dict) -> str:
        output = payload.get("output")
        if not isinstance(output, dict):
            raise ASRError("千问 ASR 响应缺少识别文本")
        text = output.get("text")
        if not text:
            nested = output.get("output") or {}
            sentence = nested.get("sentence") if isinstance(nested, dict) else {}
            text = sentence.get("text") if isinstance(sentence, dict) else ""
        if isinstance(text, str):
            return text.strip()
        raise ASRError("千问 ASR 返回了无效的识别文本")


class FunASRAdapter(ASRAdapter):
    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "ja",
        base_url: str = _DASHSCOPE_BASE_URL,
        upload_policy_url: str = _UPLOAD_POLICY_URL,
        timeout: float = 180.0,
        poll_interval: float = 2.0,
        **_ignored,
    ) -> None:
        self.api_key = (api_key or os.environ.get("DASHSCOPE_API_KEY", "")).strip()
        self.language = language or "ja"
        self.base_url = base_url.rstrip("/")
        self.upload_policy_url = upload_policy_url
        self._timeout = float(timeout)
        self._network_policy = NetworkPolicy()
        self.poll_interval = max(0.1, float(poll_interval))

    @property
    def name(self) -> str:
        return "fun-asr"

    @property
    def timeout(self) -> float:
        return self._timeout

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def check_connection(self, client: Optional[httpx.AsyncClient] = None) -> None:
        """Validate the API key, upload service, and workspace endpoint."""
        if not self.api_key:
            raise ASRError("千问 ASR API Key 未配置")
        if client is not None:
            await self._check_connection(client)
            return
        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self._timeout) as own_client:
            await self._check_connection(own_client)

    async def _check_connection(self, client: httpx.AsyncClient) -> None:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        policy = await client.get(
            self.upload_policy_url,
            headers=headers,
            params={"action": "getPolicy", "model": "fun-asr"},
        )
        self._raise_for_status(policy, "获取上传凭证")
        probe = await client.post(
            f"{self.base_url}/services/audio/asr/transcription",
            headers={
                **headers,
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            json={
                "model": "fun-asr",
                "input": {"file_urls": []},
                "parameters": {"language_hints": [self.language]},
            },
        )
        if probe.is_success:
            return
        try:
            error = probe.json()
        except ValueError:
            error = {}
        if probe.status_code == 400 and error.get("code") == "InvalidParameter":
            return
        self._raise_for_status(probe, "检测服务")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        timestamps: bool = True,
        client: Optional[httpx.AsyncClient] = None,
    ) -> ASRResult:
        if not self.api_key:
            raise ASRError("千问 ASR API Key 未配置")
        result_language = language or self.language
        try:
            if client is not None:
                return await self._transcribe(client, audio_path, result_language, timestamps)
            async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self._timeout) as own_client:
                return await self._transcribe(own_client, audio_path, result_language, timestamps)
        except ASRNoSpeechError:
            return ASRResult("", wav_duration(audio_path), result_language, [])

    async def _transcribe(self, client: httpx.AsyncClient, audio_path: str, language: str, timestamps: bool) -> ASRResult:
        oss_url = await self._upload(client, audio_path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        }
        response = await client.post(
            f"{self.base_url}/services/audio/asr/transcription",
            headers=headers,
            json={
                "model": "fun-asr",
                "input": {"file_urls": [oss_url]},
                "parameters": {"language_hints": [language]},
            },
        )
        self._raise_for_status(response, "提交任务")
        task_id = str((response.json().get("output") or {}).get("task_id") or "")
        if not task_id:
            raise ASRError("Fun-ASR 未返回任务 ID")

        deadline = asyncio.get_running_loop().time() + self._timeout
        while True:
            result_response = await client.get(f"{self.base_url}/tasks/{task_id}", headers=headers)
            self._raise_for_status(result_response, "查询任务")
            output = result_response.json().get("output") or {}
            status = str(output.get("task_status") or "").upper()
            if status == "SUCCEEDED":
                results = output.get("results") or []
                item = results[0] if results else {}
                if item.get("subtask_status") != "SUCCEEDED" or not item.get("transcription_url"):
                    if is_no_speech(item):
                        raise ASRNoSpeechError("Fun-ASR 未识别到可用文字")
                    raise ASRError(f"Fun-ASR 识别失败：{item.get('message') or '未返回结果地址'}")
                transcript_response = await client.get(item["transcription_url"])
                self._raise_for_status(transcript_response, "下载结果")
                return self._parse_transcript(transcript_response.json(), audio_path, language, timestamps)
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                if is_no_speech(output):
                    raise ASRNoSpeechError("Fun-ASR 未识别到可用文字")
                raise ASRError(f"Fun-ASR 任务失败：{output.get('message') or status}")
            if asyncio.get_running_loop().time() >= deadline:
                raise ASRError("Fun-ASR 任务等待超时")
            await asyncio.sleep(self.poll_interval)

    async def _upload(self, client: httpx.AsyncClient, audio_path: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = await client.get(
            self.upload_policy_url,
            headers=headers,
            params={"action": "getPolicy", "model": "fun-asr"},
        )
        self._raise_for_status(response, "获取上传凭证")
        data = response.json().get("data") or {}
        required = ("upload_host", "upload_dir", "oss_access_key_id", "policy", "signature")
        if any(not data.get(key) for key in required):
            raise ASRError("Fun-ASR 上传凭证不完整")
        filename = Path(audio_path).name
        object_key = f"{data['upload_dir'].rstrip('/')}/{filename}"
        form = {
            "OSSAccessKeyId": data["oss_access_key_id"],
            "policy": data["policy"],
            "Signature": data["signature"],
            "key": object_key,
            "x-oss-object-acl": data.get("x_oss_object_acl", "private"),
            "x-oss-forbid-overwrite": data.get("x_oss_forbid_overwrite", "true"),
            "success_action_status": "200",
        }
        with open(audio_path, "rb") as audio:
            upload = await client.post(data["upload_host"], data=form, files={"file": (filename, audio, "audio/wav")})
        self._raise_for_status(upload, "上传音频")
        return f"oss://{object_key}"

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return
        if response_has_no_speech(response):
            raise ASRNoSpeechError("Fun-ASR 未识别到可用文字")
        if response.status_code == 401:
            raise ASRError("千问 ASR API Key 无效")
        if response.status_code == 403:
            try:
                error = response.json()
            except ValueError:
                error = {}
            if error.get("code") == "AccessDenied.Unpurchased":
                raise ASRError("百炼 ASR 模型尚未开通，请先在当前工作空间获得模型访问资格")
            raise ASRError(f"百炼 ASR 无访问权限：{error.get('message') or response.text[:300]}")
        raise ASRError(f"Fun-ASR {operation}失败（HTTP {response.status_code}）：{response.text[:300]}")

    @staticmethod
    def _parse_transcript(payload: dict, audio_path: str, language: str, timestamps: bool) -> ASRResult:
        transcript_items = payload.get("transcripts") or []
        root = transcript_items[0] if transcript_items else payload
        sentences = root.get("sentences") or payload.get("sentences") or []
        segments: list[ASRSegment] = []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text") or "").strip()
            if not text:
                continue
            begin = sentence.get("begin_time", sentence.get("start_time", 0))
            end = sentence.get("end_time", sentence.get("end", begin))
            try:
                segments.append(ASRSegment(text=text, start=float(begin) / 1000, end=float(end) / 1000))
            except (TypeError, ValueError):
                continue
        text = str(root.get("text") or payload.get("text") or "").strip()
        if not text:
            text = "".join(segment.text for segment in segments)
        duration = wav_duration(audio_path)
        if text and not segments:
            segments = [ASRSegment(text=text, start=0.0, end=duration)]
        return ASRResult(
            text=text,
            duration=duration,
            language_code=language,
            segments=segments if timestamps else [],
        )
