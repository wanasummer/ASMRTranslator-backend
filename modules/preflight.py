"""Pipeline service preflight checks executed by the Python sidecar.

The desktop client calls one check at a time so it can render sequential
progress.  Network checks live here instead of in the WebView/Tauri HTTP
plugin, which keeps proxy handling and TLS behavior aligned with the real
pipeline's Python clients.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import edge_tts
import httpx

from .network import async_client, async_operation

from .asr import FunASRAdapter, QwenAudio31ASRFlashAdapter
from .error import ASRError, LLMError
from .llm_enhancer import LLMClient, LLMConfig, is_configured


class PreflightError(RuntimeError):
    """A configuration or connectivity failure safe to show to the user."""


@dataclass(frozen=True)
class PreflightResult:
    success: bool
    message: str

    def to_payload(self) -> dict[str, Any]:
        return {"success": self.success, "message": self.message}


def _text(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key, "")
    return str(value).strip() if value is not None else ""


def _port(config: Mapping[str, Any], key: str) -> int:
    try:
        value = int(config.get(key, 0))
    except (TypeError, ValueError) as exc:
        raise PreflightError("服务端口格式无效") from exc
    if not 1 <= value <= 65535:
        raise PreflightError("服务端口必须在 1 到 65535 之间")
    return value


def _endpoint(base_url: str, port: int) -> str:
    raw = base_url.strip().rstrip("/")
    if not raw:
        raise PreflightError("服务地址未配置")
    normalized = raw if "://" in raw else f"http://{raw}"
    parsed = urlsplit(normalized)
    if not parsed.hostname:
        raise PreflightError("服务地址格式无效")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    netloc = f"{host}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _raise_for_status(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return
    if response.status_code in (401, 403):
        raise PreflightError(f"{label}鉴权失败，请检查 API Key")
    if response.status_code == 429:
        raise PreflightError(f"{label}请求过于频繁，请稍后重试")
    body = response.text.strip()[:300]
    detail = f"：{body}" if body else ""
    raise PreflightError(f"{label}连接失败（HTTP {response.status_code}）{detail}")


async def _check_llm(config: Mapping[str, Any]) -> None:
    llm_config = LLMConfig(
        provider=_text(config, "llmProvider"),
        base_url=_text(config, "llmBaseUrl"),
        api_key=_text(config, "llmApiKey"),
        model=_text(config, "llmModel"),
        tts_backend=_text(config, "ttsBackend"),
        tts_model=_text(config, "fishModel") or _text(config, "minimaxModel"),
    )
    if not is_configured(llm_config):
        raise PreflightError("大模型配置不完整，请填写 Provider、模型和 API Key")
    try:
        await LLMClient(llm_config).check_connection()
    except LLMError as exc:
        status = exc.context.get("status")
        if status == 503:
            raise PreflightError("大模型服务暂时繁忙（HTTP 503），请稍后重试或更换模型") from exc
        if status == 429:
            raise PreflightError("大模型请求达到频率或额度上限（HTTP 429），请稍后重试") from exc
        raise PreflightError(f"大模型：{exc}") from exc
    except Exception as exc:
        raise PreflightError(f"大模型：{exc}") from exc


async def _check_asr(config: Mapping[str, Any]) -> None:
    engine = _text(config, "asrEngine")
    if engine == "fish-asr":
        api_key = _text(config, "asrApiKey")
        if not api_key:
            raise PreflightError("Fish ASR API Key 未配置")
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.get(
                "https://api.fish.audio/model",
                headers=_headers(api_key),
                params={"self": "true", "page_size": 1, "page_number": 1},
            )
        _raise_for_status(response, "Fish ASR")
        return
    if engine in {"qwen-audio-3.1-asr-flash", "fun-asr"}:
        api_key = _text(config, "qwenAsrApiKey")
        if not api_key:
            raise PreflightError("千问 ASR API Key 未配置")
        workspace_id = _text(config, "qwenAsrWorkspaceId")
        if engine == "qwen-audio-3.1-asr-flash" and not workspace_id:
            raise PreflightError("百炼 Workspace ID 未配置")
        adapter = (
            QwenAudio31ASRFlashAdapter(api_key=api_key, workspace_id=workspace_id)
            if engine != "fun-asr"
            else FunASRAdapter(api_key=api_key)
        )
        try:
            await adapter.check_connection()
        except ASRError as exc:
            raise PreflightError(str(exc)) from exc
        return
    if engine == "mimo-asr":
        api_key = _text(config, "mimoAsrApiKey")
        if not api_key:
            raise PreflightError("MiMo ASR API Key 未配置")
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.get(
                "https://api.xiaomimimo.com/v1/models",
                headers={"api-key": api_key},
            )
        _raise_for_status(response, "MiMo ASR")
        return
    if engine == "moss-transcribe-diarize":
        from .analysis_plugins import validate_endpoint
        endpoint = validate_endpoint(_endpoint(_text(config, "asrBaseUrl"), _port(config, "asrPort")))
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.get(endpoint + "/health")
        _raise_for_status(response, "MOSS")
        payload = response.json()
        if (payload.get("service") != engine or payload.get("api_version") != "moss-transcribe-diarize/1"
                or payload.get("status") != "ok" or payload.get("ready") is not True):
            raise PreflightError("MOSS 服务身份不匹配或模型尚未就绪")
        return
    if engine != "common-asr":
        raise PreflightError(f"不支持的 ASR 引擎：{engine or '未配置'}")

    endpoint = _endpoint(_text(config, "asrBaseUrl"), _port(config, "asrPort"))
    async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
        response = await client.get(
            f"{endpoint}/health",
            headers=_headers(_text(config, "asrCustomApiKey")),
        )
    _raise_for_status(response, "自定义 ASR")
    payload = response.json()
    capabilities = payload.get("capabilities") or {}
    if payload.get("status") != "ok" or payload.get("ready") is not True:
        raise PreflightError("自定义 ASR：服务可访问，但 ASR 模型尚未就绪")
    if not capabilities.get("timestamps") or not capabilities.get("external_segments"):
        raise PreflightError("自定义 ASR：服务缺少时间戳或外部分段能力")


async def _check_chickenrice(config: Mapping[str, Any]) -> None:
    endpoint = _endpoint(
        _text(config, "chickenRiceBaseUrl"),
        _port(config, "chickenRicePort"),
    )
    async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
        response = await client.get(
            f"{endpoint}/health",
            headers=_headers(_text(config, "chickenRiceApiKey")),
        )
    _raise_for_status(response, "海南鸡服务")
    payload = response.json()
    if payload.get("status") != "ready":
        raise PreflightError("海南鸡服务：服务可访问，但运行环境尚未就绪")
    capabilities = payload.get("capabilities") or {}
    if not capabilities.get("japanese_asr"):
        raise PreflightError("海南鸡服务：缺少日语 ASR 能力")
    if bool(config.get("chickenRiceTranslationEnabled", False)) and not capabilities.get("chinese_translation"):
        raise PreflightError("海南鸡服务：缺少海南鸡翻译能力")
    if bool(config.get("chickenRiceVadEnabled", False)) and not capabilities.get("chickenrice_vad"):
        raise PreflightError("海南鸡服务：缺少海南鸡专属 VAD 能力")


async def _check_tts(config: Mapping[str, Any]) -> None:
    backend = _text(config, "ttsBackend")
    if backend == "subtitles-only":
        return
    probe_only = bool(config.get("probeOnly", False))
    if backend == "common-tts":
        from .tts.common_tts_adapter import CommonTTSAdapter
        adapter = CommonTTSAdapter(manifest=config.get("commonTtsManifest", {}),
            base_url=_text(config, "commonTtsBaseUrl"), api_key=_text(config, "commonTtsApiKey"),
            parameters=config.get("commonTtsParameters", {}),
            validate_parameters=not probe_only)
        if not probe_only:
            adapter.validate_reference_assets()
        await adapter.health()
        return
    if backend == "edge-tts":
        voices = await async_operation(
            "https://speech.platform.bing.com",
            lambda connection: edge_tts.list_voices(**connection.request_options),
        )
        selected = _text(config, "edgeVoice")
        if selected and not any(voice.get("ShortName") == selected for voice in voices):
            raise PreflightError("Edge TTS 中不存在所选音色")
        return

    if backend == "fish-audio":
        api_key = _text(config, "fishApiKey")
        reference_id = _text(config, "fishReferenceId")
        if not api_key or (not probe_only and not reference_id):
            raise PreflightError("Fish Audio API Key 或音色未配置")
        async with async_client(client_factory=httpx.AsyncClient, timeout=180.0) as client:
            if probe_only:
                response = await client.get(
                    "https://api.fish.audio/model",
                    headers=_headers(api_key),
                    params={"self": "true", "page_size": 1, "page_number": 1},
                )
            else:
                response = await client.get(
                    f"https://api.fish.audio/model/{reference_id}",
                    headers=_headers(api_key),
                )
        _raise_for_status(response, "Fish Audio TTS")
        return

    if backend == "minimax-tts":
        api_key = _text(config, "minimaxApiKey")
        voice = _text(config, "minimaxVoice")
        if not api_key or (not probe_only and not voice):
            raise PreflightError("MiniMax API Key 或音色未配置")
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.post(
                "https://api.minimaxi.com/v1/get_voice",
                headers={**_headers(api_key), "Content-Type": "application/json"},
                json={"voice_type": "all"},
            )
        _raise_for_status(response, "MiniMax TTS")
        payload = response.json()
        base_response = payload.get("base_resp") or {}
        if int(base_response.get("status_code", 0)) != 0:
            raise PreflightError(f"MiniMax TTS：{base_response.get('status_msg') or '服务不可用'}")
        return

    if backend == "mimo-tts":
        api_key = _text(config, "mimoApiKey")
        if not api_key:
            raise PreflightError("MiMo TTS API Key 未配置")
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.get(
                "https://api.xiaomimimo.com/v1/models",
                headers={"api-key": api_key, **_headers(api_key)},
            )
        _raise_for_status(response, "MiMo TTS")
        return

    if backend == "indextts-2.5":
        from .tts.indextts_options import IndexTTSOptions
        if not probe_only:
            try:
                options = IndexTTSOptions.from_dict(config.get("indexTtsOptions", {}))
            except ValueError as exc:
                raise PreflightError(str(exc)) from exc
            if options.voice_source == "reference" and not _text(config, "indexTtsReferenceAudio"):
                raise PreflightError("IndexTTS 固定音色参考音频未配置")
        endpoint = _endpoint(
            _text(config, "indexTtsBaseUrl"),
            _port(config, "indexTtsPort"),
        )
        async with async_client(client_factory=httpx.AsyncClient, timeout=120.0) as client:
            response = await client.get(f"{endpoint}/config")
        _raise_for_status(response, "IndexTTS")
        return

    raise PreflightError(f"不支持的 TTS 引擎：{backend or '未配置'}")


async def _check_analysis(config, *, service, api_version, key, default):
    from .analysis_plugins import validate_endpoint
    endpoint = validate_endpoint(_text(config, key) or default)
    async with async_client(client_factory=httpx.AsyncClient, timeout=15.0) as client:
        response = await client.get(endpoint + "/health")
    _raise_for_status(response, service)
    body = response.json()
    if (body.get("service") != service or body.get("api_version") != api_version
            or body.get("status") != "ok" or body.get("ready") is not True):
        raise PreflightError("分析插件未就绪或服务类型不匹配")


async def _check_vad(config):
    await _check_analysis(config, service="asmr-vad", api_version="asmr-vad/1",
                          key="asmrVadEndpoint", default="http://127.0.0.1:8006")


async def _check_alignment(config):
    await _check_analysis(config, service="qwen-force-align", api_version="qwen-force-align/1",
                          key="qwenAlignEndpoint", default="http://127.0.0.1:8007")


_CHECKS: dict[str, Callable[[Mapping[str, Any]], Awaitable[None]]] = {
    "vad": _check_vad,
    "alignment": _check_alignment,
    "llm": _check_llm,
    "asr": _check_asr,
    "chickenrice": _check_chickenrice,
    "tts": _check_tts,
}


async def run_preflight_check(service: str, config: Mapping[str, Any]) -> PreflightResult:
    check = _CHECKS.get(service)
    if check is None:
        return PreflightResult(False, f"不支持的预检服务：{service or '未指定'}")
    try:
        await check(config)
        return PreflightResult(True, "配置正确，服务连接正常")
    except PreflightError as exc:
        return PreflightResult(False, str(exc))
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        return PreflightResult(False, f"{type(exc).__name__}：{exc}")
    except Exception as exc:  # noqa: BLE001 - sidecar boundary must return JSON
        return PreflightResult(False, f"{type(exc).__name__}：{exc}")


def run_preflight_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    service = str(payload.get("service", "")).strip()
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return PreflightResult(False, "预检配置格式无效").to_payload()
    return asyncio.run(run_preflight_check(service, config)).to_payload()
