"""Shared normalization for provider-specific "no recognizable speech" replies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ..error import ASRError


class ASRNoSpeechError(ASRError):
    """Internal control signal: the request succeeded but contained no words."""


_NO_SPEECH_CODES = {
    "ASR_RESPONSE_HAVE_NO_WORDS",
    "NO_SPEECH",
    "NO_SPEECH_DETECTED",
    "NO_RECOGNIZABLE_SPEECH",
    "NO_WORDS",
}
_NO_SPEECH_PHRASES = (
    "no speech detected",
    "no recognizable speech",
    "no recognisable speech",
    "no words recognized",
    "no words recognised",
    "未检测到语音",
    "未识别到语音",
    "没有识别到文字",
    "无可识别语音",
)
_SEMANTIC_KEYS = {
    "code", "error", "error_code", "error_message", "message", "detail",
    "reason", "status_message", "subtask_status_message",
}
_CONTAINER_KEYS = {"data", "error", "errors", "output", "result", "results"}


def is_no_speech(value: Any) -> bool:
    """Return true only for explicit provider no-speech codes/messages.

    The matcher intentionally does not classify generic empty/malformed payloads
    as silence; those remain response-contract errors.
    """
    if isinstance(value, Mapping):
        return any(
            is_no_speech(item)
            for key, item in value.items()
            if str(key).casefold() in _SEMANTIC_KEYS | _CONTAINER_KEYS
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(is_no_speech(item) for item in value)
    if not isinstance(value, str):
        return False
    text = value.strip()
    normalized = re.sub(r"[\s.-]+", "_", text).upper()
    return (
        any(code == normalized or code in normalized for code in _NO_SPEECH_CODES)
        or any(phrase in text.casefold() for phrase in _NO_SPEECH_PHRASES)
    )


def response_has_no_speech(response: httpx.Response) -> bool:
    """Inspect both structured and plain-text error bodies."""
    try:
        if is_no_speech(response.json()):
            return True
    except (ValueError, TypeError):
        pass
    return is_no_speech(response.text)


def raise_if_no_speech(value: Any) -> None:
    if is_no_speech(value):
        raise ASRNoSpeechError("ASR 未识别到可用文字")
