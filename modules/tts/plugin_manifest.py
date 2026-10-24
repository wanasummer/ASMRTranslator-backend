"""Safe YAML descriptors and authoritative plugin parameter validation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator
import yaml

from ..error import TTSError


def _defaults(schema: dict, value):
    value = copy.deepcopy(value)
    if not isinstance(schema, dict):
        return value
    if isinstance(value, dict):
        for key, child in schema.get("properties", {}).items():
            if key not in value and isinstance(child, dict) and "default" in child:
                value[key] = copy.deepcopy(child["default"])
            if key in value:
                value[key] = _defaults(child, value[key])
    return value


def validate_manifest(value: dict) -> dict:
    try:
        # Reject recursive YAML aliases, non-JSON values and nonfinite numbers.
        value = json.loads(json.dumps(value, allow_nan=False))
        if type(value.get("schema_version")) is not int or value.get("schema_version") != 1 or value.get("protocol") != "common-tts/1":
            raise ValueError("仅支持 schema_version: 1 和 common-tts/1")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value.get("id", "")):
            raise ValueError("插件 ID 无效")
        if not all(isinstance(value.get(key), str) and value[key].strip() for key in ("version", "name")):
            raise ValueError("插件需要 name 和 version")
        schema = value.get("parameters", {"type": "object", "additionalProperties": False})
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise ValueError("parameters 必须是 additionalProperties: false 的对象 schema")
        def check(node):
            if isinstance(node, dict):
                for ref in ("$ref", "$dynamicRef"):
                    if ref in node and not str(node[ref]).startswith("#/"):
                        raise ValueError("schema 不允许外部引用")
                for child in node.values():
                    check(child)
            elif isinstance(node, list):
                for child in node:
                    check(child)
        check(schema)
        Draft202012Validator.check_schema(schema)
        value["parameters"] = schema
        capabilities = value.setdefault("capabilities", {})
        if value.get("transport", "http") != "http":
            raise ValueError("当前宿主仅支持 HTTP 插件")
        if value.get("logs", {}).get("mode", "none") not in ("none", "poll"):
            raise ValueError("logs.mode 需要为 none 或 poll")
        for key in ("voices", "rate", "pitch", "boundaries"):
            if key in capabilities and not isinstance(capabilities[key], bool):
                raise ValueError(f"capabilities.{key} 必须为布尔值")
        for unsupported in ("streaming", "remote_cancel", "sessions", "idempotency"):
            if capabilities.get(unsupported):
                raise ValueError(f"当前宿主尚未实现能力 {unsupported}")
        if value.get("operations"):
            raise ValueError("当前宿主尚未实现音色管理操作接口")
        if capabilities.get("execution", "segment") != "segment":
            raise ValueError("当前宿主仅支持 segment 执行模式")
        if capabilities.get("text_policy", "strip_tags") not in ("strip_tags", "preserve"):
            raise ValueError("text_policy 无效")
        required = capabilities.get("required_assets", [])
        if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
            raise ValueError("required_assets 必须为字符串列表")
        execution = value.setdefault("execution", {})
        if not isinstance(execution.get("cache", True), bool):
            raise ValueError("execution.cache 必须为布尔值")
        for key, default in (("max_concurrent", 1), ("timeout_seconds", 600), ("max_upload_bytes", 67108864),
                             ("max_output_bytes", 134217728)):
            number = execution.setdefault(key, default)
            if isinstance(number, bool) or not isinstance(number, (float, int)) or number <= 0:
                raise ValueError(f"execution.{key} 必须为正数")
        if not isinstance(execution["max_concurrent"], int):
            raise ValueError("max_concurrent 必须为整数")
        # v1 deliberately has one complete, verifiable downstream format.
        if value.get("output", "wav") != "wav":
            raise ValueError("common-tts/1 当前仅接受完整 PCM WAV 输出")
        return value
    except Exception as exc:
        raise TTSError(f"TTS 插件描述无效: {exc}") from exc


def load_manifest(path: str) -> dict:
    try:
        source = Path(path)
        if source.stat().st_size > 1024 * 1024:
            raise TTSError("TTS YAML 超过 1 MiB")
        text = source.read_text(encoding="utf-8-sig")
        if any(isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for token in yaml.scan(text)):
            raise TTSError("插件 YAML 不支持 anchors 或 aliases，请使用局部 Schema 引用")
        return validate_manifest(yaml.safe_load(text))
    except TTSError:
        raise
    except (OSError, TypeError, UnicodeError, yaml.YAMLError) as exc:
        raise TTSError("无法读取 TTS 插件 YAML；请检查文件和编码") from exc


def resolve_parameters(manifest: dict, parameters: dict, overrides: dict | None = None) -> dict:
    if not isinstance(parameters, dict) or not isinstance(overrides or {}, dict):
        raise TTSError("插件参数必须为对象")
    try:
        json.dumps([parameters, overrides], allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise TTSError("插件参数必须为有限的 JSON 值") from exc
    schema = manifest["parameters"]
    for key in overrides or {}:
        field = schema.get("properties", {}).get(key, {})
        if not isinstance(field, dict) or not field.get("x-segment-override", False):
            raise TTSError(f"插件参数 {key} 不允许逐段覆盖")
    # Object/array overrides replace whole values; null is a value, not deletion.
    resolved = _defaults(schema, {**parameters, **(overrides or {})})
    try:
        errors = sorted(Draft202012Validator(schema).iter_errors(resolved), key=lambda e: str(e.path))
    except Exception as exc:
        raise TTSError("插件参数 Schema 无法执行，请检查局部引用") from exc
    if errors:
        error = errors[0]
        # Validation messages may embed secret values. Emit locations only.
        raise TTSError(f"插件参数校验失败: {'.'.join(map(str, error.path)) or '<root>'} ({error.validator})")
    return resolved


def public_parameters(schema: dict, parameters):
    """Explicit secret metadata, including secrets whose names reveal nothing."""
    schema = schema if isinstance(schema, dict) else {}
    if schema.get("x-secret"):
        return "<redacted>"
    if isinstance(parameters, dict):
        return {k: public_parameters(schema.get("properties", {}).get(k, {}), v) for k, v in parameters.items()}
    if isinstance(parameters, list):
        return [public_parameters(schema.get("items", {}), v) for v in parameters]
    return parameters


def map_audio_parameters(schema: dict, value, convert, location=()):
    """Walk nested objects and arrays without flattening plugin configuration."""
    schema = schema if isinstance(schema, dict) else {}
    if schema.get("x-control") == "audio-file" and value:
        return convert(schema, value, location)
    if isinstance(value, dict):
        return {key: map_audio_parameters(schema.get("properties", {}).get(key, {}), item,
                convert, (*location, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [map_audio_parameters(schema.get("items", {}), item, convert, (*location, str(index)))
                for index, item in enumerate(value)]
    return copy.deepcopy(value)


def secret_values(schema: dict, value) -> list[str]:
    schema = schema if isinstance(schema, dict) else {}
    if schema.get("x-secret"):
        if isinstance(value, (dict, list)):
            return [str(item) for item in _flatten(value) if isinstance(item, str) and item]
        return [value] if isinstance(value, str) and value else []
    if isinstance(value, dict):
        return [secret for key, item in value.items() for secret in
                secret_values(schema.get("properties", {}).get(key, {}), item)]
    if isinstance(value, list):
        return [secret for item in value for secret in secret_values(schema.get("items", {}), item)]
    return []


def _flatten(value):
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [item for child in value for item in _flatten(child)]
    return [value]


def read_runtime_configuration(stream) -> dict:
    """Read a bounded constructor snapshot from an injected binary stream."""
    raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise TTSError("通用 TTS 配置超过 2 MiB")
    try:
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected object")
        json.dumps(value, allow_nan=False)
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise TTSError("通用 TTS stdin 配置必须为有限 UTF-8 JSON 对象") from exc
