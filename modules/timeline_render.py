"""Render the untouched studio arrangement as the pipeline's first audio output.

Overlay listen modes that follow the original produce exactly what the studio
previews and what an arrangement export of the same draft renders: one
whole-timeline overlay of the processed voices, so a voice may extend past its
pink slice up to its landing point.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .arrangement import initial_state, prepare_arrangement_plan, voice_asset_path
from .arrangement_contracts import ArrangementAudioAdapter, ArrangementPlan
from .audio_joiner import SegmentPlacement


@dataclass(frozen=True)
class TimelineRenderConfig:
    max_rate: float
    tts_volume: float

    def arrangement_config(self) -> dict:
        return {"followOriginalRate": self.max_rate, "ttsVolume": self.tts_volume}


@dataclass(frozen=True)
class TimelineRender:
    path: str
    placements: tuple[SegmentPlacement, ...]
    plan: ArrangementPlan


def publish_voice_assets(segments: list[dict], work: Path) -> dict[int, Path]:
    """Write each processed voice to the stable path the studio and arrangement read.

    ``skip_tts`` segments have no voice; any stale asset at their path is removed.
    Segments are not mutated.
    """
    from pydub import AudioSegment
    published = {}
    for seg in segments:
        target = voice_asset_path(work, seg["index"])
        source = seg.get("tts_file")
        if seg.get("skip_tts") or not source or not Path(source).is_file():
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        AudioSegment.from_file(source).export(target, format="wav")
        published[seg["index"]] = target
    return published


def render_initial_timeline(segments: list[dict], output_dir: str, audio_path: str,
                            config: TimelineRenderConfig,
                            adapter: ArrangementAudioAdapter | None = None) -> TimelineRender:
    """Render ``initial_state`` through the arrangement renderer into ``output.wav/mp3``."""
    from .arrangement_audio import create_arrangement_audio_adapter
    work = Path(output_dir)
    published = publish_voice_assets(segments, work)
    plan = prepare_arrangement_plan(work, Path(audio_path), initial_state(work),
                                    config.arrangement_config(), {})
    result = (adapter or create_arrangement_audio_adapter()).render(plan)
    wav, mp3 = work / "output.wav", work / "output.mp3"
    result.audio.export(wav, format="wav")
    result.audio.export(mp3, format="mp3", bitrate="192k")
    shift = result.origin_shift_ms
    placements = tuple(
        SegmentPlacement(str(seg.get("segment_id", seg["index"])),
                         int(seg["start_ms"]) + shift, int(seg["end_ms"]) + shift,
                         int(seg["original_start_ms"]) + shift, int(seg["original_end_ms"]) + shift)
        for seg in segments)
    print(f"       [timeline] 发布 {len(published)} 段处理后人声，按编排时间轴渲染 "
          f"{len(plan.clips)} 段 → {wav}")
    return TimelineRender(str(wav), placements, plan)
