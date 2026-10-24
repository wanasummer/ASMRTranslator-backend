"""Post-TTS handler chain — pluggable audio processing after TTS generation.

The pipeline runs generate_tts, then hands every segment to a chain of
AudioHandler strategies before loudness alignment:

    generate_tts → [TTSArtifactHandler] → [PhantomReverbHandler] → align → VC → builder

Each handler mutates `seg` in place and returns the same list, so handlers can
be added/removed/reordered freely:

  TTSArtifactHandler   : run the artifact detector on the raw TTS; when a segment
                         sounds "male-like/mechanical", regenerate it once and
                         re-check — if the second pass is still flagged the TTS is
                         accepted as-is and the outcome is logged (seg["artifact"]).
  PhantomReverbHandler : apply the phantom whisper DSP (lowpass → convolution
                         reverb → fades) to the raw TTS, in the chain (pre-align)
                         instead of inside the phantom builder. Records the raw
                         TTS path as seg["phantom_voice_src"] so the builder's
                         delay clamp measures the voice (not the reverb tail).

To add a handler: subclass AudioHandler, implement name/process, and register it
in build_tts_handler_chain(). Nothing else in the pipeline needs to change.
"""

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from .phantom_engine import apply_phantom_dsp, get_ir_path
from .tts_artifact import detect_tts_artifact
from .tts_engine import get_tts_adapter, prepare_text_for_synthesis
from .task_cache import JsonResultCache, cache_key, file_digest, public_settings


class AudioHandler(ABC):
    """One post-TTS audio processing step. Mutates seg dicts in place."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler id for logs, e.g. 'tts-artifact', 'phantom-reverb'."""
        ...

    @abstractmethod
    def process(self, segments, output_dir, max_workers=4):
        """Process every segment; return the same (mutated) list."""
        ...


class AudioHandlerChain:
    """Sequential pipeline of AudioHandler steps."""

    def __init__(self, handlers=None):
        self.handlers = list(handlers or [])

    def add(self, handler):
        self.handlers.append(handler)
        return self

    def run(self, segments, output_dir, max_workers=4):
        for handler in self.handlers:
            segments = handler.process(segments, output_dir, max_workers=max_workers)
        return segments


class TTSArtifactHandler(AudioHandler):
    """Detect male-like/mechanical artifacts; regenerate once when flagged."""

    name = "tts-artifact"

    def __init__(self, backend="edge-tts", backend_kwargs=None, rate="+0%",
                 pitch="+0Hz", max_retries=1, sr=24000):
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.rate = rate
        self.pitch = pitch
        self.max_retries = max_retries
        self.sr = sr

    def process(self, segments, output_dir, max_workers=4):
        counts = {}
        cache = JsonResultCache(os.path.join(output_dir, "artifact_results"))

        def _one(seg):
            tts_file = seg.get("tts_file")
            if not tts_file or not os.path.exists(tts_file):
                seg["artifact"] = {"status": "skipped"}
                return "skipped"

            key = cache_key({"index": seg["index"], "audio": file_digest(tts_file),
                             "text": seg.get("text", ""), "speaker": seg.get("speaker"),
                             "backend": self.backend, "settings": public_settings(self.backend_kwargs or {}),
                             "rate": self.rate, "pitch": self.pitch,
                             "sr": self.sr, "max_retries": self.max_retries})
            cached = cache.load(key)
            if cached and os.path.isfile(cached["path"]) and file_digest(cached["path"]) == cached["sha256"]:
                seg["tts_file"] = cached["path"]
                seg["artifact"] = {"status": cached["status"]}
                return cached["status"]

            first = detect_tts_artifact(tts_file, sr=self.sr)
            if first.get("error"):
                raise RuntimeError(f"seg {seg['index']} artifact detection failed: {first['error']}")
            if not first["flagged"]:
                seg["artifact"] = {"status": "clean"}
                return "clean"

            # Sync synthesis runs in a fresh event loop per call. Keep each
            # segment's adapter local to its worker so async clients never
            # cross worker threads or closed event loops.
            from .tts.models import TTSSegmentContext
            adapter = get_tts_adapter(self.backend, self.backend_kwargs)
            if callable(getattr(type(adapter), "for_segment", None)):
                adapter = adapter.for_segment(TTSSegmentContext.from_segment(seg))
            if callable(getattr(type(adapter), "bind_segment", None)):
                adapter = adapter.bind_segment(seg)
            voice = adapter.get_voice_for_speaker(seg.get("speaker"))
            text = prepare_text_for_synthesis(adapter, seg.get("text", ""))
            status = "accepted_flagged"
            regenerated = os.path.join(output_dir, f"artifact_tts_{seg['index']:04d}" + os.path.splitext(tts_file)[1])
            for attempt in range(self.max_retries):
                try:
                    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                        pending = os.path.join(temporary, os.path.basename(tts_file))
                        shutil.copyfile(tts_file, pending)
                        adapter.synthesize_sync(text, voice, pending,
                                                rate=self.rate, pitch=self.pitch)
                        os.replace(pending, regenerated)
                        sidecar = pending + ".boundaries.json"
                        if os.path.isfile(sidecar):
                            os.replace(sidecar, regenerated + ".boundaries.json")
                        elif os.path.isfile(regenerated + ".boundaries.json"):
                            os.remove(regenerated + ".boundaries.json")
                except Exception as error:
                    cache.fail(key, error)
                    raise RuntimeError(f"seg {seg['index']} artifact regeneration failed: {error}") from error
                after = detect_tts_artifact(regenerated, sr=self.sr)
                if not after["flagged"]:
                    status = "regenerated"
                    print(f"       [tts-artifact] seg {seg['index']}: 检出劣化 → 第{attempt + 1}次重生成后干净")
                    break
                status = "accepted_flagged"
            if status == "accepted_flagged":
                print(f"       [tts-artifact] seg {seg['index']}: 重生成后仍检出 → 按正常接受 (记录日志)")
            seg["artifact"] = {"status": status, "detect": first}
            if self.max_retries > 0:
                seg["tts_file"] = regenerated
                cache.save(key, {"path": regenerated, "sha256": file_digest(regenerated), "status": status})
            return status

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for status in pool.map(_one, segments):
                counts[status] = counts.get(status, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"       [tts-artifact] 检查 {len(segments)} 段 → {summary}")
        return segments


class PhantomReverbHandler(AudioHandler):
    """Phantom whisper DSP in the chain: lowpass → convolution reverb → fades.

    Runs on the raw TTS (before loudness alignment). Writes
    phantom_dsp_XXXX.wav next to the TTS, points seg["tts_file"] at it, and
    records the raw path as seg["phantom_voice_src"] so the phantom builder's
    follow_original delay clamp measures the voice, not the reverb tail.
    """

    name = "phantom-reverb"

    def __init__(self, delay_ms=400, reverb_mix=0.4, lowpass_hz=3800,
                 fade_in_ms=120, fade_out_ms=200):
        self.delay_ms = delay_ms
        self.reverb_mix = reverb_mix
        self.lowpass_hz = lowpass_hz
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms

    def process(self, segments, output_dir, max_workers=4):
        ir_path = get_ir_path(output_dir)
        counts = {}

        def _one(seg):
            tts_file = seg.get("tts_file")
            if not tts_file or not os.path.exists(tts_file):
                seg["phantom_dsp_file"] = None
                return "skipped"
            dsp_path = os.path.join(output_dir, f"phantom_dsp_{seg['index']:04d}.wav")
            fresh = False
            try:
                fresh = (os.path.exists(dsp_path)
                         and os.path.getsize(dsp_path) > 100
                         and os.path.getmtime(dsp_path) >= os.path.getmtime(tts_file))
            except OSError:
                fresh = False
            if not fresh:
                try:
                    apply_phantom_dsp(tts_file, dsp_path, self.lowpass_hz, self.reverb_mix,
                                      ir_path, fade_in_ms=self.fade_in_ms,
                                      fade_out_ms=self.fade_out_ms)
                except Exception as error:
                    raise RuntimeError(f"seg {seg['index']} phantom DSP failed: {error}") from error
            seg["phantom_voice_src"] = tts_file  # raw TTS, pre-DSP (delay-clamp basis)
            seg["tts_file"] = dsp_path
            seg["phantom_dsp_file"] = dsp_path
            return "processed"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for status in pool.map(_one, segments):
                counts[status] = counts.get(status, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"       [phantom-reverb] 处理 {len(segments)} 段 → {summary}")
        return segments


def build_tts_handler_chain(*, detect_artifact=True, phantom_mode=False,
                            tts_backend="edge-tts", tts_backend_kwargs=None,
                            tts_rate="+0%", tts_pitch="+0Hz",
                            phantom_delay_ms=400, phantom_reverb_mix=0.4,
                            phantom_lowpass_hz=3800):
    """Build the post-TTS handler chain for a pipeline run.

    detect_artifact=False disables the artifact check entirely (no detector run,
    no regeneration). phantom_mode=True appends the phantom reverb handler so the
    DSP moves from the builder into the chain.
    """
    chain = AudioHandlerChain()
    if detect_artifact:
        chain.add(TTSArtifactHandler(backend=tts_backend, backend_kwargs=tts_backend_kwargs,
                                     rate=tts_rate, pitch=tts_pitch))
    if phantom_mode:
        chain.add(PhantomReverbHandler(delay_ms=phantom_delay_ms,
                                       reverb_mix=phantom_reverb_mix,
                                       lowpass_hz=phantom_lowpass_hz))
    return chain
