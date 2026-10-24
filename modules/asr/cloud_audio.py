"""Shared helpers for cloud ASR adapters."""

from __future__ import annotations

import base64
import mimetypes
import wave
from pathlib import Path


def audio_data_url(audio_path: str) -> str:
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    payload = base64.b64encode(Path(audio_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def wav_duration(audio_path: str) -> float:
    """Return duration for VAD-produced WAV input without an ffprobe process."""
    with wave.open(audio_path, "rb") as audio:
        rate = audio.getframerate()
        return audio.getnframes() / rate if rate else 0.0
