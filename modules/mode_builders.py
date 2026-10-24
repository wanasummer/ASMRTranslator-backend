"""Listen-mode segment builders — Strategy pattern.

The four listen modes (replace / dual / sync / phantom) differ only in how the
loudness-aligned TTS is combined with the original segment. Each is a Strategy
that writes the final per-segment audio (setting seg["vc_file"]), so the
pipeline can swap modes without if/elif chains.

Extending with a new mode:
    1. Create a new builder class inheriting SegmentBuilder
    2. Add a branch to create_segment_builder()
    Done.
"""

from abc import ABC, abstractmethod
from typing import TypedDict

from .audio_joiner import build_dual_segments, build_sync_segments
from .phantom_engine import build_phantom_segments


class BuilderExportSettings(TypedDict, total=False):
    phantom_mode: bool
    retain_original: bool
    sync_mode: str
    follow_original_rate: float
    phantom_delay_ms: int
    phantom_reverb_mix: float
    phantom_lowpass_hz: int
    phantom_orig_fade_ms: int
    phantom_tts_fade_ms: int
    phantom_virtual_fade: bool
    gap_ms: int


class SegmentBuilder(ABC):
    """Strategy: build final per-segment audio for one listen mode."""

    wav_params: list[str] | None = None

    @abstractmethod
    def export_settings(self) -> BuilderExportSettings:
        """Factory configuration sufficient to rebuild from native source slices."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Mode id for logging, e.g. 'phantom', 'sync', 'dual', 'replace'."""
        ...

    @property
    def subdir(self) -> str:
        """Intermediate sub-directory name under the chunk dir."""
        return "04_build"

    @abstractmethod
    def build(self, segments, output_dir, max_workers=4):
        """Set seg['vc_file'] for each segment and return the same list."""
        ...


class ReplaceBuilder(SegmentBuilder):
    """Replace mode: TTS replaces the original — vc_file is already the TTS file."""

    name = "replace"

    def build(self, segments, output_dir, max_workers=4):
        return segments

    def export_settings(self) -> BuilderExportSettings:
        return {}


class DualBuilder(SegmentBuilder):
    """Dual (对照) mode: original + gap + TTS."""

    name = "dual"
    subdir = "04_dual"

    def __init__(self, gap_ms: int = 200):
        self.gap_ms = gap_ms

    def build(self, segments, output_dir, max_workers=4):
        return build_dual_segments(segments, output_dir,
                                   gap_ms=self.gap_ms, max_workers=max_workers,
                                   **({"wav_params": self.wav_params} if self.wav_params else {}))

    def export_settings(self) -> BuilderExportSettings:
        return {"retain_original": True, "gap_ms": self.gap_ms}


class SyncBuilder(SegmentBuilder):
    """Sync (声音同步) mode: TTS overlaid on the original."""

    name = "sync"
    subdir = "04_sync"

    def __init__(self, sync_mode: str = "follow_original", follow_original_rate: float = 1.15):
        self.sync_mode = sync_mode
        self.follow_original_rate = follow_original_rate

    def build(self, segments, output_dir, max_workers=4):
        return build_sync_segments(segments, output_dir,
                                   sync_mode=self.sync_mode,
                                   follow_original_rate=self.follow_original_rate,
                                   max_workers=max_workers,
                                   **({"wav_params": self.wav_params} if self.wav_params else {}))

    def export_settings(self) -> BuilderExportSettings:
        return {"retain_original": True, "sync_mode": self.sync_mode,
                "follow_original_rate": self.follow_original_rate}


class PhantomBuilder(SegmentBuilder):
    """Phantom Whisper mode: delayed reverb-washed TTS overlaid on the original."""

    name = "phantom"
    subdir = "04_phantom"

    def __init__(self, delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                 sync_mode="follow_original", follow_original_rate: float = 1.15,
                 orig_fade_ms=0, tts_fade_ms=0, virtual_fade=False):
        self.delay_ms = delay_ms
        self.reverb_mix = reverb_mix
        self.lowpass_hz = lowpass_hz
        self.sync_mode = sync_mode
        self.follow_original_rate = follow_original_rate
        self.orig_fade_ms = orig_fade_ms
        self.tts_fade_ms = tts_fade_ms
        self.virtual_fade = virtual_fade

    def build(self, segments, output_dir, max_workers=4):
        return build_phantom_segments(
            segments, output_dir,
            delay_ms=self.delay_ms, reverb_mix=self.reverb_mix,
            lowpass_hz=self.lowpass_hz, sync_mode=self.sync_mode,
            follow_original_rate=self.follow_original_rate,
            max_workers=max_workers, orig_fade_ms=self.orig_fade_ms,
            tts_fade_ms=self.tts_fade_ms, virtual_fade=self.virtual_fade,
            **({"wav_params": self.wav_params} if self.wav_params else {}))

    def export_settings(self) -> BuilderExportSettings:
        return {"phantom_mode": True, "phantom_delay_ms": self.delay_ms,
                "phantom_reverb_mix": self.reverb_mix, "phantom_lowpass_hz": self.lowpass_hz,
                "sync_mode": self.sync_mode, "follow_original_rate": self.follow_original_rate,
                "phantom_orig_fade_ms": self.orig_fade_ms, "phantom_tts_fade_ms": self.tts_fade_ms,
                "phantom_virtual_fade": self.virtual_fade}


def create_segment_builder(*, phantom_mode=False, retain_original=False,
                           sync_mode="none", follow_original_rate=1.15,
                           phantom_delay_ms=400,
                           phantom_reverb_mix=0.4, phantom_lowpass_hz=3800,
                           phantom_orig_fade_ms=0, phantom_tts_fade_ms=0,
                           phantom_virtual_fade=False,
                           gap_ms=200) -> SegmentBuilder:
    """Resolve the listen mode into its build strategy.

    Preserves the current precedence: phantom wins, then retain_original →
    sync (when sync_mode is set) or dual, else replace.
    """
    if phantom_mode:
        return PhantomBuilder(phantom_delay_ms, phantom_reverb_mix, phantom_lowpass_hz,
                              sync_mode=sync_mode, follow_original_rate=follow_original_rate,
                              orig_fade_ms=phantom_orig_fade_ms,
                              tts_fade_ms=phantom_tts_fade_ms,
                              virtual_fade=phantom_virtual_fade)
    if retain_original:
        if sync_mode != "none":
            return SyncBuilder(sync_mode, follow_original_rate)
        return DualBuilder(gap_ms)
    return ReplaceBuilder()
