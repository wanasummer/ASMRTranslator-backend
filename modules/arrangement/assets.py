"""Read-only lookup of arrangement audio assets and the TTS settings that place them."""
from __future__ import annotations

import math
from pathlib import Path

from .contracts import ArrangementError
from ..storage import address, read_json


def resolve_replacement(work: Path, relative: str) -> Path:
    path = (work / relative).resolve()
    if not path.is_relative_to((work / "arrangement/audio").resolve()) or not path.is_file():
        raise ArrangementError("替换音频不存在或超出编排目录")
    return path


def voice_asset_path(work: Path, position: int) -> Path:
    """The processed (artifact-checked, mode-DSP) voice that the timeline renders."""
    return work / f"chunk_0/02_tts/voice_{position:04d}.wav"


def cached_tts(work: Path, position: int) -> Path | None:
    # Matches the studio audio track lookup order; never confuse index with position.
    candidates = (voice_asset_path(work, position),
                  *(work / f"chunk_0/02_tts/tts_{position:04d}.{extension}" for extension in ("mp3", "wav")))
    return next((path for path in candidates if path.is_file()), None)


def pipeline_tts_rates(work: Path) -> dict[int, float]:
    target = address(work / "chunk_0/02_tts/tts_tempo_")
    rates = {}
    for key in target.store.keys(target.scope, target.key):
        value = read_json(work / key)
        index = int(Path(key).stem.removeprefix("tts_tempo_"))
        rate = float(value["rate"])
        if math.isfinite(rate) and rate > 0:
            rates[index] = rate
    return rates


def tts_volume(config: dict) -> float:
    """The user's TTS volume, the same multiplier the pipeline applies to every voice."""
    try:
        value = float(config.get("ttsVolume", 1))
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(2.0, value)) if math.isfinite(value) else 1.0


def tts_max_rate(config: dict) -> float:
    """The user's "最大加速倍率", clamped to the same range as the pipeline."""
    try:
        value = float(config.get("followOriginalRate", 1.15))
    except (TypeError, ValueError):
        return 1.15
    return max(1.0, min(1.25, value)) if math.isfinite(value) else 1.15
