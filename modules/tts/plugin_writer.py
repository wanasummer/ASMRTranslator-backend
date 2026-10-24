"""Validate and persist common-TTS artifacts in the stage's temporary workspace."""

import io
import wave

from ..error import TTSError
from ..io import open_write, write_json
from .plugin_contract import TTSSynthesisResult


def write_plugin_result(result: TTSSynthesisResult, output_path: str) -> str:
    try:
        with wave.open(io.BytesIO(result.audio), "rb") as audio:
            if audio.getcomptype() != "NONE" or audio.getnframes() <= 0:
                raise ValueError("需要非空 PCM WAV")
            expected = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(audio.getnframes())) != expected:
                raise ValueError("WAV 数据不完整")
    except (wave.Error, EOFError, ValueError) as exc:
        raise TTSError("通用 TTS 插件返回了无效或不完整的 PCM WAV") from exc
    with open_write(output_path, "wb") as target:
        target.write(result.audio)
    if result.boundaries is not None:
        write_json(output_path + ".boundaries.json", result.boundaries)
    return output_path
