"""Versioned, JSON-serializable input to independently deployed TTS plugins."""

from __future__ import annotations

import copy
import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
import wave

from ..task_cache import file_digest


@dataclass(frozen=True)
class AudioAsset:
    """A local resource. The HTTP bridge uploads bytes, never local paths."""

    id: str
    path: str
    role: str
    sha256: str
    media_type: str = "audio/wav"
    sample_rate_hz: int | None = None
    channels: int | None = None
    duration_ms: int | None = None
    text: str | None = None

    @classmethod
    def from_file(cls, identity: str, path: str, role: str, text: str | None = None):
        metadata = {}
        if Path(path).suffix.lower() == ".wav":
            with wave.open(path, "rb") as audio:
                metadata = dict(sample_rate_hz=audio.getframerate(), channels=audio.getnchannels(),
                                duration_ms=round(audio.getnframes() * 1000 / audio.getframerate()))
        import mimetypes
        return cls(identity, str(Path(path).resolve()), role, file_digest(path),
                   mimetypes.guess_type(path)[0] or "application/octet-stream", text=text, **metadata)

    def wire_metadata(self) -> dict:
        value = asdict(self)
        value.pop("path")
        return value


@dataclass(frozen=True)
class TTSSynthesisRequest:
    segment_id: str
    index: int
    text: str
    authored_text: str
    speaker: str | None = None
    source_text: str | None = None
    target_language: str = "zh-CN"
    timing: dict[str, Any] = field(default_factory=dict)
    assets: tuple[AudioAsset, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_segment(cls, segment: dict) -> "TTSSynthesisRequest":
        """Copy the stage return value; never retain caller-owned dictionaries."""
        assets = tuple(AudioAsset.from_file(role, segment[key], role)
                       for role, key in (("source_speech", "speech_file"),
                                         ("source_segment", "segment_file")) if segment.get(key))
        return cls(str(segment.get("segment_id", segment["index"])), segment["index"],
                   segment["text"], segment["text"], segment.get("speaker"),
                   segment.get("source_text") or None,
                   timing={key: segment.get(key) for key in (
                       "original_start_ms", "original_end_ms", "max_allowed_tts_ms", "tts_delay_ms")},
                   assets=assets, overrides=copy.deepcopy(segment.get("tts_overrides", {})),
                   context=copy.deepcopy(segment.get("tts_context", {})))


@dataclass(frozen=True)
class TTSSynthesisResult:
    """Decoded transport result; a dedicated writer publishes these artifacts."""

    audio: bytes
    media_type: str
    boundaries: dict | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict) -> "TTSSynthesisResult":
        """Optional structured result for engines returning word timings."""
        if payload.get("contract_version") != 1:
            raise ValueError("Result contract version mismatch")
        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ValueError("Invalid warnings")
        boundaries = payload.get("boundaries")
        if boundaries is not None:
            if not isinstance(boundaries, dict) or boundaries.get("type") != "word" or boundaries.get("unit") != "ms":
                raise ValueError("Invalid word boundary format")
            if not isinstance(boundaries.get("words"), list):
                raise ValueError("Invalid words")
            import math
            for word in boundaries["words"]:
                if not isinstance(word.get("text"), str) or any(
                    isinstance(word.get(key), bool) or not isinstance(word.get(key), (int, float))
                    or not math.isfinite(word[key]) or word[key] < 0 for key in ("offset_ms", "duration_ms")):
                    raise ValueError("Invalid word timing")
        return cls(base64.b64decode(payload["audio_base64"], validate=True), "audio/wav", boundaries, tuple(warnings))


class TTSPlugin(Protocol):
    async def synthesize_request(self, request: TTSSynthesisRequest, config: dict,
                                 runtime: Any) -> TTSSynthesisResult: ...
