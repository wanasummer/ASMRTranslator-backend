"""HTTP bridge for versioned, YAML-described common TTS plugins.

Existing adapters retain their signatures. Per-segment binding is private to this
bridge and copies context so concurrent calls cannot overwrite each other's input.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from ..error import TTSError, descriptor, is_memory_allocation_error
from ..network import NetworkPolicy, async_client, sync_client
from ..plugin_logs import relay_plugin_logs
from .base import TTSAdapter
from .models import VoiceInfo
from .plugin_contract import AudioAsset, TTSSynthesisRequest, TTSSynthesisResult
from .plugin_manifest import load_manifest, map_audio_parameters, public_parameters, resolve_parameters, secret_values, validate_manifest
from .plugin_writer import write_plugin_result


@dataclass(frozen=True)
class TTSRuntimeContext:
    request_id: str
    task_id: str
    attempt: int
    emit: Callable[[str], None]


class CommonTTSAdapter(TTSAdapter):
    name = "common-tts"
    output_extension = ".wav"

    def __init__(self, voice_map=None, *, manifest=None, manifest_path=None,
                 base_url="", api_key="", parameters=None, task_id="", transport=None,
                 validate_parameters=True):
        super().__init__(voice_map)
        self.manifest = validate_manifest(manifest) if manifest is not None else load_manifest(manifest_path)
        self.parameters = (resolve_parameters(self.manifest, parameters or {})
                           if validate_parameters else copy.deepcopy(parameters or {}))
        try:
            if not isinstance(base_url, str):
                raise ValueError("expected URL string")
            parsed = urlsplit(base_url)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment or parsed.port == 0):
                raise ValueError("invalid endpoint")
        except ValueError as exc:
            raise TTSError("通用 TTS 服务需要不带凭据、查询参数的 HTTP(S) 地址和有效端口") from exc
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self.task_id = task_id or str(uuid4())
        self.transport = transport
        self._network_policy = NetworkPolicy()
        self._request = None
        self._semaphore = None

    def bind_segment(self, segment: dict):
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.manifest["execution"]["max_concurrent"])
        adapter = copy.copy(self)
        adapter._request = TTSSynthesisRequest.from_segment(segment)
        return adapter

    def prepare_text(self, text: str) -> str:
        if self.manifest["capabilities"].get("text_policy") == "preserve":
            return text
        from ..llm_enhancer import strip_emotion_tags
        return strip_emotion_tags(text)

    def cache_inputs(self) -> dict:
        parameters = resolve_parameters(self.manifest, self.parameters,
                                        self._request.overrides if self._request else {})
        assets, parameters = self._assets(parameters, self._request.assets if self._request else ())
        return {"manifest": self.manifest, "endpoint": self.base_url,
                "parameters": public_parameters(self.manifest["parameters"], parameters),
                "assets": [asset.wire_metadata() for asset in assets],
                "segment": {"id": self._request.segment_id, "timing": self._request.timing,
                            "authored_text": self._request.authored_text, "speaker": self._request.speaker,
                            "language": self._request.target_language,
                            "context": self._request.context, "source_text": self._request.source_text}
                if self._request else None}

    @property
    def cache_reusable(self):
        return self.manifest["execution"].get("cache", True)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _client_options(self):
        return dict(network_policy=self._network_policy, timeout=float(self.manifest["execution"]["timeout_seconds"]),
                    transport=self.transport)

    def _check(self, response):
        if response.is_error:
            # Provider bodies may contain request credentials or private text.
            code = "HTTP_ERROR"
            error = {}
            try:
                candidate = response.json().get("error", {})
                if isinstance(candidate, dict):
                    error = candidate
                    code = str(error.get("code", code))[:80]
            except (ValueError, AttributeError):
                pass
            if not code.replace("_", "").isalnum():
                code = "HTTP_ERROR"
            # Inspect provider messages only for classification; never expose them.
            # Older plugins use SYNTHESIS_FAILED even for model allocation errors.
            diagnostic = " ".join(str(error.get(key, "")) for key in ("code", "message", "detail"))
            mapped = None
            if is_memory_allocation_error(diagnostic):
                mapped = "TTS_OUT_OF_MEMORY"
            elif code.upper() == "MODEL_LOAD_FAILED":
                mapped = "TTS_MODEL_LOAD_FAILED"
            detail = f"通用 TTS 请求失败: HTTP {response.status_code} / {code}"
            if mapped:
                raise TTSError(descriptor(mapped), detail=detail,
                               context={"status": response.status_code, "backend": self.manifest["id"]})
            raise TTSError(detail, context={"status": response.status_code, "backend": self.manifest["id"]})

    def get_available_voices(self) -> list[VoiceInfo]:
        if not self.manifest["capabilities"].get("voices", False):
            return []
        try:
            with sync_client(client_factory=httpx.Client, **self._client_options()) as client:
                response = client.get(self.base_url + "/voices", headers=self._headers())
            self._check(response)
            return [VoiceInfo(**{key: item[key] for key in ("id", "name", "gender", "language", "description") if key in item})
                    for item in response.json()["voices"]]
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError("无法读取通用 TTS 音色目录") from exc

    async def health(self):
        async with async_client(client_factory=httpx.AsyncClient, **self._client_options()) as client:
            response = await client.get(self.base_url + "/health", headers=self._headers())
        self._check(response)
        payload = response.json()
        if (payload.get("contract_version") != 1 or payload.get("ready") is not True
                or payload.get("plugin_id") != self.manifest["id"]
                or payload.get("plugin_version") != self.manifest["version"]):
            raise TTSError("通用 TTS 尚未就绪，或插件 ID/版本与 YAML 不一致")

    def validate_reference_assets(self) -> None:
        """Check configured files and limits before upstream segment assets exist."""
        self._assets(self.parameters, (), check_required_roles=False)

    def _assets(self, parameters: dict, source: tuple, *, check_required_roles: bool = True):
        assets = list(source)
        def convert(schema, path, location):
            from hashlib import sha256
            if not isinstance(path, str):
                raise TTSError("参考音频必须为本地文件")
            identity = "parameter_" + sha256(json.dumps(location).encode()).hexdigest()[:24]
            asset = AudioAsset.from_file(identity, path, schema.get("x-role", ".".join(location)))
            assets.append(asset)
            return {"asset_id": asset.id}
        parameters = map_audio_parameters(self.manifest["parameters"], parameters, convert)
        if len(assets) > 32:
            raise TTSError("common-tts/1 每次请求最多支持 32 个音频素材")
        if check_required_roles:
            for role in self.manifest["capabilities"].get("required_assets", []):
                if not any(asset.role == role for asset in assets):
                    raise TTSError(f"插件缺少必需音频: {role}")
        if sum(Path(asset.path).stat().st_size for asset in assets) > self.manifest["execution"]["max_upload_bytes"]:
            raise TTSError("插件音频输入超过 max_upload_bytes")
        return tuple(assets), parameters

    async def synthesize(self, text, voice, output_path, rate="+0%", pitch="+0Hz"):
        if self._request is None:
            raise TTSError("通用 TTS 需要流水线逐段上下文")
        request = replace(self._request, text=text)
        parameters = resolve_parameters(self.manifest, self.parameters, request.overrides)
        runtime = TTSRuntimeContext(str(uuid4()), self.task_id, 1,
                                    lambda message: print(message, flush=True))
        result = await self.synthesize_request(request, {"parameters": parameters, "voice": voice,
                                                        "rate": rate, "pitch": pitch}, runtime)
        return write_plugin_result(result, output_path)

    def _log_emitter(self, runtime, request, parameters):
        secrets = [self.api_key, *secret_values(self.manifest["parameters"], parameters)]
        def emit(message):
            for secret in secrets:
                if secret:
                    message = message.replace(secret, "[REDACTED]")
            runtime.emit(f"task={runtime.task_id} segment={request.segment_id} request={runtime.request_id} {message}")
        return emit

    async def synthesize_request(self, request: TTSSynthesisRequest, config: dict,
                                 runtime: TTSRuntimeContext) -> TTSSynthesisResult:
        capabilities = self.manifest["capabilities"]
        for key, neutral in (("rate", "+0%"), ("pitch", "+0Hz")):
            if config[key] != neutral and not capabilities.get(key, False):
                raise TTSError(f"插件不支持 {key}，请恢复默认值")
        assets, parameters = self._assets(config["parameters"], request.assets)
        payload = {"contract_version": 1, "plugin_id": self.manifest["id"],
                   "plugin_version": self.manifest["version"], "request_id": runtime.request_id,
                   "task_id": runtime.task_id, "attempt": runtime.attempt,
                   "segment": {"id": request.segment_id, "index": request.index,
                               "text": request.text, "authored_text": request.authored_text,
                               "source_text": request.source_text, "speaker": request.speaker,
                               "language": request.target_language, "timing": request.timing,
                               "context": request.context},
                   "parameters": parameters, "controls": {k: config[k] for k in ("voice", "rate", "pitch")},
                   "assets": [asset.wire_metadata() for asset in assets], "output": {"container": "wav"}}
        headers = {**self._headers(), "X-Request-ID": runtime.request_id, "Accept": "audio/wav, application/json"}
        # Shared through copy.copy bindings; create in the active event loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.manifest["execution"]["max_concurrent"])
        async with self._semaphore:
            async with async_client(client_factory=httpx.AsyncClient, **self._client_options()) as client:
                stop = asyncio.Event()
                poll = None
                if self.manifest.get("logs", {}).get("mode") == "poll":
                    poll = asyncio.create_task(relay_plugin_logs(client, self.base_url, runtime.request_id,
                        self._headers(), stop, source=self.manifest["id"],
                        emit=self._log_emitter(runtime, request, config["parameters"])))
                try:
                    with ExitStack() as stack:
                        files = [(asset.id, (asset.id + Path(asset.path).suffix,
                                 stack.enter_context(open(asset.path, "rb")), asset.media_type)) for asset in assets]
                        # A payload file makes even text-only requests multipart.
                        files.insert(0, ("payload", (None, json.dumps(payload, ensure_ascii=False), "application/json")))
                        async with asyncio.timeout(float(self.manifest["execution"]["timeout_seconds"])):
                            async with client.stream("POST", self.base_url + "/job", files=files, headers=headers) as response:
                                if response.is_error:
                                    await response.aread()
                                self._check(response)
                                if (response.headers.get("x-tts-contract-version") != "1"
                                        or response.headers.get("x-request-id") != runtime.request_id):
                                    raise TTSError("通用 TTS 响应协议或请求标识不匹配")
                                content_type = response.headers.get("content-type", "").split(";")[0]
                                if content_type not in ("audio/wav", "audio/x-wav", "application/json"):
                                    raise TTSError("通用 TTS 响应必须为 WAV 或结构化 JSON 结果")
                                audio = bytearray()
                                async for chunk in response.aiter_bytes():
                                    audio.extend(chunk)
                                    if len(audio) > self.manifest["execution"]["max_output_bytes"]:
                                        raise TTSError("通用 TTS 返回音频超过 max_output_bytes")
                    runtime.emit(f"[plugin:{self.manifest['id']}] segment={request.segment_id} request={runtime.request_id} completed")
                    if content_type == "application/json":
                        payload = json.loads(audio)
                        if payload.get("request_id") != runtime.request_id or payload.get("segment_id") != request.segment_id:
                            raise TTSError("通用 TTS 结构化结果的请求或分段标识不匹配")
                        result = TTSSynthesisResult.from_json(payload)
                        for warning in result.warnings:
                            self._log_emitter(runtime, request, config["parameters"])("[warning] " + warning)
                        return result
                    return TTSSynthesisResult(bytes(audio), "audio/wav")
                except asyncio.CancelledError:
                    # v1 cancellation stops local waiting; remote acceptance is unknown.
                    raise
                except TTSError:
                    raise
                except Exception as exc:
                    raise TTSError(f"通用 TTS 传输失败 ({type(exc).__name__})；未自动重提交任务") from exc
                finally:
                    stop.set()
                    if poll is not None:
                        await poll
