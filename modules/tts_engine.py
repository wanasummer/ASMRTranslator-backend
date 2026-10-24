"""Chinese TTS engine — backward-compat wrapper around the TTS adapter system.

Keeps the same public API (generate_tts, VOICE_MAP, get_tts_voice) so callers
in pipeline.py and elsewhere work unchanged. New code should use modules.tts
directly.
"""

import asyncio
import os
import re
import tempfile
from typing import Optional

from .tts import create_tts, EdgeTTSAdapter, TTSAdapter
from .tts.models import TTSSegmentContext
from .llm_enhancer import strip_emotion_tags
from .task_cache import JsonResultCache, cache_key, file_digest, public_settings


# Default voice mappings — per backend
_BACKEND_VOICE_MAPS: dict[str, dict[str, str]] = {
    "common-tts": {"default": ""},
    "edge-tts": {
        "default": "zh-CN-XiaoxiaoNeural",
        "遥": "zh-CN-XiaoyiNeural",
        "瑠衣": "zh-CN-XiaoxiaoNeural",
        "female": "zh-CN-XiaoyiNeural",
    },
    "mimo-tts": {
        "default": "mimo_default",
        "遥": "茉莉",
        "瑠衣": "冰糖",
        "female": "冰糖",
        "male": "Milo",
    },
    "fish-audio": {
        "default": "default",
        "遥": "default",
        "瑠衣": "default",
        "female": "default",
        "male": "default",
    },
    "minimax-tts": {
        "default": "Chinese (Mandarin)_Warm_Girl",
        "female": "Chinese (Mandarin)_Warm_Girl",
    },
    "indextts-2.5": {
        "default": "reference",
        "female": "reference",
        "male": "reference",
    },
}

# Lazy-init adapter shared across calls (edge-tts default). Keyed on the resolved
# voice_map so a later `default_voice` override (or its absence) is never masked
# by a previously cached adapter.
_adapter: Optional[TTSAdapter] = None
_adapter_voice_map: Optional[dict[str, str]] = None


def prepare_text_for_synthesis(adapter, text: str) -> str:
    """Return the text to actually synthesize.

    Emotion tags ([whispering] etc.) stay in the stored content but only the
    backends that support them keep them:
    - fish-audio understands the bracket tags natively;
    - minimax-tts keeps them only on speech-2.8-hd/turbo, where the LLM emits
      MiniMax voice-effect tags ((laughs) (sighs) ...) instead — bracket tags
      are still stripped there (leftovers); on speech-2.6 both forms are
      stripped (2.6 has no tag support and would read them aloud).
    - every other backend strips the tags.
    """
    if callable(getattr(type(adapter), "prepare_text", None)):
        return adapter.prepare_text(text)
    name = getattr(adapter, "name", "")
    if name == "fish-audio":
        return text
    if name == "minimax-tts":
        model = getattr(adapter, "_model", "") or ""
        if model in ("speech-2.8-hd", "speech-2.8-turbo"):
            return strip_emotion_tags(text)  # 保留 (laughs) 等，只删方括号残留
        return strip_emotion_tags(_strip_minimax_voice_tags(text))
    return strip_emotion_tags(text)


# MiniMax 语气词标签（speech-2.6 等不支持，需剥离以免被朗读出来）
_MINIMAX_VOICE_TAG_RE = re.compile(
    r"\((?:laughs|chuckle|coughs|clear-throat|groans|breath|pant|inhale|"
    r"exhale|gasps|sniffs|sighs|snorts|burps|lip-smacking|humming|hissing|"
    r"emm|sneezes)\)"
)


def _strip_minimax_voice_tags(text: str) -> str:
    return _MINIMAX_VOICE_TAG_RE.sub("", text)


def _get_voice_map(backend: str) -> dict[str, str]:
    """Return default voice map for the given backend."""
    return dict(_BACKEND_VOICE_MAPS.get(backend, _BACKEND_VOICE_MAPS["edge-tts"]))


def _get_adapter(backend: str = "edge-tts",
                 backend_kwargs: Optional[dict] = None) -> TTSAdapter:
    global _adapter, _adapter_voice_map
    voice_map = _get_voice_map(backend)

    # Merge optional default_voice override into voice_map
    if backend_kwargs:
        kwargs = dict(backend_kwargs)
        default_voice = kwargs.pop("default_voice", None)
        if default_voice:
            voice_map = dict(voice_map)
            for key in voice_map:
                voice_map[key] = default_voice
    else:
        kwargs = {}

    if backend == "edge-tts" and not kwargs:
        if _adapter is not None and _adapter_voice_map == voice_map:
            return _adapter
        _adapter = create_tts("edge-tts", voice_map=voice_map)
        _adapter_voice_map = voice_map
        return _adapter
    return create_tts(backend, voice_map=voice_map, **kwargs)


def get_tts_adapter(backend: str = "edge-tts",
                    backend_kwargs: Optional[dict] = None) -> TTSAdapter:
    """Public access to the resolved TTS adapter (shares the edge-tts instance).

    Used by the post-TTS handler chain to regenerate a flagged segment with the
    exact same voice/backend configuration as generate_tts.
    """
    return _get_adapter(backend, backend_kwargs)


# Expose VOICE_MAP for introspection / backward compat (default = edge-tts)
VOICE_MAP = _BACKEND_VOICE_MAPS["edge-tts"]


def get_tts_voice(speaker: Optional[str]) -> str:
    """Map speaker name to TTS voice ID."""
    return _get_adapter().get_voice_for_speaker(speaker)


async def _generate_one(
    semaphore: asyncio.Semaphore,
    seg: dict,
    output_dir: str,
    rate: str,
    pitch: str,
    adapter: TTSAdapter,
    cache_settings: Optional[dict] = None,
) -> dict:
    """Generate TTS for one segment with rate limiting."""
    idx = seg["index"]
    if seg.get("skip_tts"):
        # Interjection block (≤3 Chinese chars): no translation — the pipeline
        # keeps the original audio for this span, so no TTS is synthesized.
        return seg
    if callable(getattr(type(adapter), "for_segment", None)):
        try:
            adapter = adapter.for_segment(TTSSegmentContext.from_segment(seg))
        except Exception as error:
            from .error import TtsError, descriptor
            raise TtsError(
                descriptor("TTS_SEGMENT_FAILED", segment=idx + 1),
                detail=str(error), cause=error,
                context={"segment": idx + 1, "backend": adapter.name},
            ) from error
    if callable(getattr(type(adapter), "bind_segment", None)):
        try:
            adapter = adapter.bind_segment(seg)
        except Exception as error:
            from .error import TtsError, descriptor
            raise TtsError(descriptor("TTS_SEGMENT_FAILED", segment=idx + 1),
                detail=str(error), cause=error,
                context={"segment": idx + 1, "backend": adapter.name}) from error
    voice = adapter.get_voice_for_speaker(seg.get("speaker"))
    output_extension = getattr(adapter, "output_extension", ".mp3")
    tts_filename = f"tts_{idx:04d}{output_extension}"
    tts_path = os.path.join(output_dir, tts_filename)

    cache = JsonResultCache(os.path.join(output_dir, "results"))
    text = prepare_text_for_synthesis(adapter, seg["text"])
    segment_inputs = public_settings(
        adapter.cache_inputs() if callable(getattr(type(adapter), "cache_inputs", None)) else {}
    )
    key_data = {"index": idx, "text": text, "voice": voice, "rate": rate, "pitch": pitch,
                "backend": getattr(adapter, "name", type(adapter).__name__),
                "settings": public_settings(cache_settings or {})}
    if segment_inputs:
        key_data["segment_inputs"] = segment_inputs
    if callable(getattr(type(adapter), "bind_segment", None)):
        key_data["plugin_inputs"] = adapter.cache_inputs()
    key = cache_key(key_data)
    cached = cache.load(key) if getattr(adapter, "cache_reusable", True) else None
    if (cached and cached.get("path") == tts_path and os.path.isfile(tts_path)
            and file_digest(tts_path) == cached.get("sha256")):
        print(f"       [cached] {tts_filename}")
        seg["tts_file"] = tts_path
        return seg

    async with semaphore:
        await asyncio.sleep(idx * 0.05)  # Stagger
        try:
            # Failed streams must never become resumable audio. Keep the stem
            # for plugin segment IDs, publishing only after successful synthesis.
            with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                pending = os.path.join(temporary, tts_filename)
                await adapter.synthesize(text, voice, pending, rate=rate, pitch=pitch)
                if not os.path.isfile(pending) or os.path.getsize(pending) <= 0:
                    from .error import TtsError, descriptor
                    raise TtsError(descriptor("TTS_OUTPUT_INVALID"))
                os.replace(pending, tts_path)
                sidecar = pending + ".boundaries.json"
                if os.path.isfile(sidecar):
                    os.replace(sidecar, tts_path + ".boundaries.json")
                elif os.path.isfile(tts_path + ".boundaries.json"):
                    os.remove(tts_path + ".boundaries.json")
            cache.save(key, {"path": tts_path, "sha256": file_digest(tts_path),
                             "segment_inputs": segment_inputs})
        except Exception as error:
            from .error import PipelineError, TtsError, descriptor
            mapped = (error.descriptor if isinstance(error, PipelineError)
                      and error.descriptor.code != "TTS_SEGMENT_FAILED"
                      else descriptor("TTS_SEGMENT_FAILED", segment=idx + 1))
            mapped_error = TtsError(
                mapped,
                detail=str(error), cause=error,
                context={"segment": idx + 1, "backend": adapter.name},
            )
            cache.fail(key, mapped_error)
            raise mapped_error from error
        if not os.path.isfile(tts_path) or os.path.getsize(tts_path) <= 0:
            from .error import TtsError, descriptor
            raise TtsError(
                descriptor("TTS_OUTPUT_INVALID"),
                detail=f"segment {idx + 1} produced an empty output",
                context={"segment": idx + 1, "backend": adapter.name},
            )
        seg["tts_file"] = tts_path

    return seg


def generate_tts(
    segments: list[dict],
    output_dir: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    max_concurrent: int = 8,
    backend: str = "edge-tts",
    backend_kwargs: Optional[dict] = None,
    progress_callback: Optional[callable] = None,
) -> list[dict]:
    """Generate Chinese TTS for each segment with rate limiting.

    Args:
        segments: List of segment dicts with "index", "speaker", "text".
        output_dir: Directory to save TTS audio files.
        rate: Speed adjustment, e.g. "+10%", "-5%", "+0%".
        pitch: Pitch adjustment, e.g. "+10Hz", "-20Hz", "+0Hz".
        max_concurrent: Maximum concurrent TTS requests.
        backend: TTS backend name ("edge-tts", "mimo-tts", etc.).
        backend_kwargs: Backend-specific constructor kwargs.
        progress_callback: Optional fn(done, total) called after each segment.

    Returns:
        Segments mutated with "tts_file" added. All segments are attempted;
        failures are raised after draining the stage and successes persist.
    """
    os.makedirs(output_dir, exist_ok=True)
    adapter = _get_adapter(backend, backend_kwargs)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_all():
        n_total = len(segments)
        n_done = 0
        tasks = [asyncio.create_task(
            _generate_one(semaphore, seg, output_dir, rate, pitch, adapter, backend_kwargs)
        ) for seg in segments]
        failures = []
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as error:
                    failures.append(error)
                    print(f"       [tts-failed] {getattr(getattr(error, 'descriptor', None), 'code', type(error).__name__)}; 继续其他切片")
                n_done += 1
                if progress_callback:
                    progress_callback(n_done, n_total)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            close_adapter = getattr(adapter, "aclose", None)
            if close_adapter is not None:
                await close_adapter()
        # Results are mutated in-place in seg dicts
        if failures:
            error = failures[0]
            from .error import PipelineError, TtsError, descriptor
            raise TtsError(
                error.descriptor if isinstance(error, PipelineError)
                else descriptor("TTS_SEGMENT_FAILED", segment=1),
                detail=str(error), cause=error,
                context={**getattr(error, "context", {}),
                         "failed_count": len(failures), "total": n_total,
                         "failed_segments": [getattr(e, "context", {}).get("segment") for e in failures]},
            ) from error
        return segments

    return asyncio.run(_run_all())
