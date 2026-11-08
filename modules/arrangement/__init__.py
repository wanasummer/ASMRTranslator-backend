"""Project-backed arrangement: draft state, subtitle/TTS editing and the audio plan renderer.

Callers outside this package import from here; submodules are internal.
"""
from .assets import cached_tts, voice_asset_path
from .audio import create_arrangement_audio_adapter, write_arrangement_result
from .contracts import (ArrangedClip, ArrangementAudioAdapter, ArrangementAudioResult, ArrangementError,
                        ArrangementPlan, ArrangementState)
from .editor import run_editor_request
from .plan import prepare_arrangement_plan
from .state import initial_state, publish_subtitle_base
from .takes import adopt_rendered_takes

__all__ = [
    "ArrangedClip", "ArrangementAudioAdapter", "ArrangementAudioResult", "ArrangementError",
    "ArrangementPlan", "ArrangementState", "adopt_rendered_takes", "cached_tts",
    "create_arrangement_audio_adapter", "initial_state", "prepare_arrangement_plan",
    "publish_subtitle_base", "run_editor_request", "voice_asset_path", "write_arrangement_result",
]
