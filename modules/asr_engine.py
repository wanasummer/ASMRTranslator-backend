"""ASR orchestration — slice transcription → sentences.

``modules.pipeline`` runs the VAD stage itself (FSMN-VAD detect → optimize →
``vad_output/`` artifacts; see ``_run_vad_stage``) and hands the resulting
speech slices to ``transcribe_slices``. That entry point wraps ``process_audio``
and converts the sentence-level segments into ``VTTBlock``. ``process_audio``:

    load config → build+check the Fish ASR adapter → extract the given slices
    into WAV chunks → parallel transcription with a shared httpx client →
    absolute-time + dedupe → Japanese sentence merge → optional debug JSON.

Every final block is one complete VAD speech slice: ASR reads the padded
extraction range, while its times come from the saved unpadded speech timeline
(NOT the Fish ASR timestamps or decode padding) and its text is the
concatenated recognized sentences of that slice. This module owns
the ASR config (``asr_kwargs``); the pipeline's VAD stage is configured by
``run_pipeline``'s ``vad_*`` parameters instead.
"""

import asyncio
import copy
import inspect
import os
import random
import time
from dataclasses import asdict
from typing import Optional

import httpx

from .network import async_client

from .asr import create_asr
from .asr.models import ASRSegment, ASRResult, AudioChunk, SentenceSegment
from .asr.no_speech import ASRNoSpeechError
from .asr.segmentation import (
    dedupe_overlaps,
    extract_ranges,
    merge_into_sentences,
    to_absolute,
)
from .error import ASRError
from .io import intermediate_dir, write_json
from .pipeline_logger import print_progress
from .vtt_parser import VTTBlock
from .task_cache import JsonResultCache, cache_key, file_digest, public_settings

# Default backend — the only ASR engine shipped.
_DEFAULT_BACKEND = "fish-asr"
# Top-level asr_kwargs keys forwarded to the adapter constructor.
_ADAPTER_KEYS = ("api_key", "base_url", "workspace_id", "timeout", "max_retries")
# Epsilon (seconds) for deciding which fragments a slice spans.
_FRAG_EPS = 0.001
# Extra external attempts after the adapter exhausts its internal retries when
# the API is rate-limiting or the network fails — semaphore released while waiting.
_EXTRA_429_TRIES = 2

# Default ASR pipeline configuration. The ``--asr-kwargs`` JSON flag deep-merges
# on top; only recognized section keys are kept so a stray field does not leak
# into adapter constructors (adapter-specific keys like api_key / timeout are
# forwarded separately by the orchestrator).
_DEFAULT_CONFIG: dict = {
    "fish_asr": {
        "language": "ja",
        # false = 让 API 返回逐片段时间戳（ignore_timestamps 默认 true 会返回空 segments）。
        "ignore_timestamps": False,
    },
    "boundary": {
        "merge_gap_ms": 500,
        "candidate_gap_ms": 700,
        "strong_cut_gap_ms": 1200,
    },
    "sentence": {
        "min_sec": 0.4,
        "soft_max_sec": 12,
        "hard_max_sec": 15,
    },
    "debug": False,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_config(asr_kwargs: Optional[dict]) -> dict:
    """Return the effective ASR config, ``asr_kwargs`` overriding the defaults.

    Unknown top-level keys in ``asr_kwargs`` are dropped; recognized section keys
    (e.g. ``vad.target_min_sec``) deep-merge onto the defaults.
    """
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    if not asr_kwargs:
        return cfg
    for key, value in asr_kwargs.items():
        if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
            cfg[key] = _deep_merge(cfg[key], value)
        elif key in cfg:
            cfg[key] = value
    return cfg


def _build_adapter_kwargs(asr_kwargs: dict, cfg: dict, language: str) -> dict:
    """Collect adapter constructor args from top-level kwargs + config sections."""
    kwargs = {k: asr_kwargs[k] for k in _ADAPTER_KEYS if k in asr_kwargs}
    fish_cfg = cfg.get("fish_asr", {})
    kwargs["language"] = (
        asr_kwargs.get("language") or fish_cfg.get("language") or language or "ja"
    )
    kwargs["ignore_timestamps"] = fish_cfg.get("ignore_timestamps", False)
    return kwargs


async def _transcribe_chunks(chunks, adapter, max_workers, progress_cb,
                             result_cache=None, cache_settings=None) -> list:
    """Transcribe all chunks on one event loop, capped at ``max_workers``.

    Adapters that accept a ``client=`` kwarg share a single ``httpx.AsyncClient``
    across every request — one TLS handshake instead of one per chunk. A chunk
    that fails is recorded while all other chunks are attempted. Once drained,
    any failures raise a stage error. Returns ``(chunk, ASRResult)`` pairs in completion
    order; ``progress_cb(done, total)`` fires once per chunk.
    """
    sem = asyncio.Semaphore(max_workers)
    total = len(chunks)
    supports_client = "client" in inspect.signature(adapter.transcribe).parameters

    async def _one(chunk: AudioChunk, client) -> tuple:
        # Fish 的并发限制(429)除适配器内部重试外，这里在信号量外再补几次：
        # 等待期间释放并发槽，其它 chunk 可以继续转写。重试耗尽即停止。
        for attempt in range(_EXTRA_429_TRIES + 1):
            retry_429 = False
            async with sem:
                try:
                    if client is not None:
                        result = await adapter.transcribe(chunk.path, client=client)
                    else:
                        result = await adapter.transcribe(chunk.path)
                except ASRNoSpeechError:
                    result = ASRResult(
                        text="",
                        duration=max(0.0, chunk.end - chunk.start),
                        segments=[],
                    )
                    print(f"       [asr] chunk {chunk.id} 未识别到文字，按空片段继续")
                    return chunk, result
                except ASRError as e:
                    if (("429" in str(e) or e.descriptor.code in
                         ("ASR_UNREACHABLE", "ASR_TIMEOUT", "ASR_RATE_LIMITED"))
                            and attempt < _EXTRA_429_TRIES):
                        retry_429 = True
                    else:
                        from .error import AsrError, descriptor
                        segment = chunks.index(chunk) + 1
                        mapped = (e.descriptor if e.descriptor.code != "ASR_SEGMENT_FAILED"
                                  else descriptor("ASR_SEGMENT_FAILED", segment=segment))
                        raise AsrError(
                            mapped,
                            detail=str(e), cause=e,
                            context={"segment": segment, "backend": adapter.name},
                        ) from e
                except Exception as e:
                    if isinstance(e, httpx.TransportError) and attempt < _EXTRA_429_TRIES:
                        retry_429 = True
                    else:
                        from .error import AsrError, descriptor
                        segment = chunks.index(chunk) + 1
                        raise AsrError(
                            descriptor("ASR_SEGMENT_FAILED", segment=segment),
                            detail=str(e), cause=e,
                            context={"segment": segment, "backend": adapter.name},
                        ) from e
                else:
                    if (result.text or "").strip() and not result.segments:
                        from .error import AsrError, descriptor
                        segment = chunks.index(chunk) + 1
                        raise AsrError(
                            descriptor("ASR_TIMESTAMPS_MISSING"),
                            detail=f"chunk {chunk.id} returned text without timestamps",
                            context={"segment": segment, "backend": adapter.name},
                        )
                    return chunk, result
            if retry_429:
                print(f"       [asr] chunk {chunk.id} 触发限流或网络错误，"
                      f"释放并发槽等待 {2 ** attempt:.0f}s 后重试"
                      f"（{attempt + 1}/{_EXTRA_429_TRIES}）")
                await asyncio.sleep(2 ** attempt + random.uniform(0, 0.5))
        from .error import AsrError, descriptor
        segment = chunks.index(chunk) + 1
        raise AsrError(descriptor("ASR_SEGMENT_FAILED", segment=segment),
                       detail="rate-limit retries exhausted",
                       context={"segment": segment, "backend": adapter.name})

    async def _cached_one(chunk, client):
        key = cache_key({"audio": file_digest(chunk.path), "backend": adapter.name,
                         "settings": public_settings(cache_settings or {})}) if result_cache and os.path.isfile(chunk.path) else None
        cached = result_cache.load(key) if key else None
        if cached is not None:
            try:
                result = ASRResult(**{**cached, "segments": [ASRSegment(**s) for s in cached["segments"]]})
                if result.text.strip() and not result.segments:
                    raise ValueError("cached timestamps missing")
                print(f"       [cached-asr] chunk {chunk.id}")
                return chunk, result
            except (TypeError, KeyError, ValueError, AttributeError):
                pass
        try:
            pair = await _one(chunk, client)
            if key:
                result_cache.save(key, asdict(pair[1]))
            return pair
        except Exception as error:
            if key:
                result_cache.fail(key, error)
            raise

    async def _run(client) -> list:
        coros = [_cached_one(c, client) for c in chunks]
        results: list = []
        done = 0
        tasks = [asyncio.create_task(coro) for coro in coros]
        failures = []
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    chunk, result = await coro
                except Exception as error:
                    failures.append(error)
                    print(f"       [asr-failed] {getattr(getattr(error, 'descriptor', None), 'code', type(error).__name__)}; 继续其他切片")
                else:
                    results.append((chunk, result))
                done += 1
                progress_cb(done, total)
        except BaseException:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if failures:
            error = failures[0]
            from .error import AsrError, PipelineError, descriptor
            raise AsrError(
                error.descriptor if isinstance(error, PipelineError)
                else descriptor("ASR_SEGMENT_FAILED", segment=1),
                detail=str(error), cause=error,
                context={**getattr(error, "context", {}),
                         "failed_count": len(failures), "total": total,
                         "failed_segments": [getattr(e, "context", {}).get("segment") for e in failures]},
            ) from error
        return results

    if supports_client:
        async with async_client(client_factory=httpx.AsyncClient, timeout=getattr(adapter, "timeout", 120.0)) as client:
            return await _run(client)
    return await _run(None)


def _write_debug_json(output_dir: str, chunks, pairs, abs_segs, sentences) -> None:
    debug = {
        "chunks": [
            {"id": c.id, "start": c.start, "end": c.end, "path": c.path}
            for c, _ in pairs
        ],
        "asr_segments": [
            {"text": s.text, "start": s.start, "end": s.end} for s in abs_segs
        ],
        "final_segments": [
            {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
            for s in sentences
        ],
    }
    write_json(
        os.path.join(intermediate_dir(output_dir, "asr_subtitles"),
                     "asr_debug.json"),
        debug,
    )


def process_audio(
    audio_path: str,
    slices,
    language: str = "ja",
    asr_kwargs: Optional[dict] = None,
    output_dir: str = ".",
    max_workers: int = 3,
    backend: str = _DEFAULT_BACKEND,
) -> list[SentenceSegment]:
    """Transcribe the given speech slices and return sentence-level segments.

    Synchronous entry point (drives the async transcription internally). Raises
    ``ASRError`` when the adapter is not configured. ``slices`` are the optimized
    speech regions produced by the pipeline's VAD stage; their spans are extracted
    into WAV chunks under ``output_dir/asr_slices/``. Debug JSON lands under
    ``output_dir/asr_subtitles/asr_debug.json`` when ``debug`` is enabled.
    """
    asr_kwargs = dict(asr_kwargs or {})
    cfg = _load_config(asr_kwargs)
    adapter = create_asr(backend, **_build_adapter_kwargs(asr_kwargs, cfg, language))
    if not adapter.is_configured():
        raise ASRError(f"未配置 {adapter.name} API Key")

    t0 = time.perf_counter()
    slices_dir = intermediate_dir(output_dir, "asr_slices")

    # 1. Extract the given speech slices into WAV chunks.
    if not slices:
        print("       [warn] 无任何 ASR 切片（VAD 未检出人声），字幕将为空。")
        return []
    print_progress("asr", "正在提取音频切片", 0, 1)
    chunks = extract_ranges(
        audio_path,
        [(r.start, r.end) for r in slices],
        slices_dir,
    )
    print_progress("asr", "正在提取音频切片", 1, 1)
    print(f"       [asr] 切片提取完成：{len(chunks)} 个 WAV，"
          f"耗时 {time.perf_counter() - t0:.1f}s")

    # 2. Parallel transcription.
    def _progress(done: int, total: int) -> None:
        print(f"       [asr] 语音识别：{done}/{total}")
        print_progress("asr", "正在识别语音", done, total)

    t1 = time.perf_counter()
    pairs = asyncio.run(_transcribe_chunks(
        chunks, adapter, max_workers, _progress,
        JsonResultCache(os.path.join(output_dir, "asr_results")),
        _build_adapter_kwargs(asr_kwargs, cfg, language),
    ))
    print(f"       [asr] 语音识别完成：{len(pairs)}/{len(chunks)}，"
          f"耗时 {time.perf_counter() - t1:.1f}s")

    # 3. Absolute timestamps + normalize + cross-chunk overlap dedup.
    abs_segs: list[ASRSegment] = []
    for chunk, result in pairs:
        abs_segs.extend(to_absolute(chunk, result))
    abs_segs = dedupe_overlaps(abs_segs)
    if not abs_segs and any(result.text for _, result in pairs):
        print("       [warn] ASR 返回了文本但没有时间戳切片（segments 为空）。"
              "请确认 ignore_timestamps=false 已生效，否则无法切句、字幕为空。")

    # 4. Japanese sentence merge — partition the deduped fragments onto the
    # non-overlapping slice timeline first.  Raw VAD/long-window slices overlap
    # by design; independently filtering abs_segs for every raw slice would let
    # one already-deduped fragment be consumed twice and recreate duplicate
    # sentences downstream.
    t2 = time.perf_counter()
    print_progress("asr", "正在切分句子", 0, 1)
    sentences: list[SentenceSegment] = []
    _, fragment_buckets = _partition_fragments_by_slice(abs_segs, slices)
    for parts in fragment_buckets:
        sentences.extend(merge_into_sentences(
            parts,
            boundary_cfg=cfg["boundary"],
            sentence_cfg=cfg["sentence"],
        ))
    print_progress("asr", "正在切分句子", 1, 1)
    print(f"       [asr] 句段合并完成：{len(sentences)} 句，"
          f"耗时 {time.perf_counter() - t2:.1f}s")

    if cfg.get("debug"):
        _write_debug_json(output_dir, chunks, pairs, abs_segs, sentences)

    print(f"       [asr] ASR 阶段总耗时 {time.perf_counter() - t0:.1f}s")
    return sentences


def _collapse_overlapping_slices(slices):
    """Order slices and trim overlaps so the subtitle timeline never double-covers.

    Decode padding is removed using the saved speech contract first; input
    extraction ranges are never mutated. ``optimize_asmr_segments`` may emit overlapping windows (long spans split by
    :func:`split_long_segment`, or ±padding widening adjacent slices). Each
    slice keeps only the span not already covered by the previous one, so the
    resulting blocks tile the timeline without overlap.
    """
    from dataclasses import replace
    from .vad.timeline import speech_bounds_ms
    out = []
    for r in sorted(slices, key=lambda s: s.start):
        speech_start, speech_end = speech_bounds_ms(r)
        r = replace(r, start=speech_start / 1000, end=speech_end / 1000)
        if not out:
            out.append(r)
            continue
        start = max(r.start, out[-1].end)
        if start < r.end:
            out.append(replace(r, start=start))
    return out


def _partition_fragments_by_slice(fragments, slices):
    """Assign every absolute ASR fragment to at most one slice.

    ASR extraction keeps overlapping source windows so boundary speech is seen
    by both requests.  After :func:`dedupe_overlaps` has reduced those duplicate
    observations to one fragment, sentence merging must consume that fragment
    only once.  We therefore assign by fragment start on the collapsed,
    half-open slice timeline.  A timestamp-drift fallback chooses the slice with
    the largest temporal intersection, but still places the fragment in exactly
    one bucket.
    """
    ordered = _collapse_overlapping_slices(slices)
    buckets = [[] for _ in ordered]

    for fragment in sorted(fragments, key=lambda s: (s.start, s.end)):
        owner = None

        # Prefer the half-open interval containing the fragment start.  The
        # epsilon is only for a fragment just before the first slice; using it
        # on every slice boundary would make ownership ambiguous again.
        for i, region in enumerate(ordered):
            lower = region.start - (_FRAG_EPS if i == 0 else 0.0)
            if lower <= fragment.start < region.end:
                owner = i
                break

        if owner is None:
            best_overlap = 0.0
            for i, region in enumerate(ordered):
                overlap = min(fragment.end, region.end) - max(
                    fragment.start, region.start
                )
                if overlap > best_overlap:
                    owner, best_overlap = i, overlap

        if owner is not None:
            buckets[owner].append(fragment)

    return ordered, buckets


def _group_sentences_into_slices(sentences, slices) -> list[VTTBlock]:
    """Collapse ASR sentences onto the VAD slice timeline, one block per slice.

    Each ``optimize_asmr_segments`` slice becomes exactly one subtitle block:
    ``start_ms``/``end_ms`` come from the slice span (the slice timeline — ASR
    per-fragment timestamps are deliberately NOT used), and ``text`` is the
    concatenation of every sentence whose start falls inside that slice.
    Sentences are assigned by start time so a fragment is never counted twice;
    slices with no recognized text are dropped.
    """
    ordered = _collapse_overlapping_slices(slices)
    text_by_slice: list[list[str]] = [[] for _ in ordered]

    def _assign(sent):
        # Prefer the slice containing the sentence start; fall back to the
        # slice with the largest temporal overlap (ASR time drift edge case).
        for i, r in enumerate(ordered):
            if r.start <= sent.start < r.end:
                return i
        best, best_overlap = None, 0.0
        for i, r in enumerate(ordered):
            overlap = min(sent.end, r.end) - max(sent.start, r.start)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        return best

    for s in sentences:
        idx = _assign(s)
        if idx is not None:
            text_by_slice[idx].append(s.text)

    out: list[VTTBlock] = []
    for i, (r, parts) in enumerate(zip(ordered, text_by_slice)):
        if not parts:
            continue
        out.append(VTTBlock(
            index=len(out),
            start_ms=round(r.start * 1000),
            end_ms=round(r.end * 1000),
            text="".join(parts),
            asr_slice=getattr(r, "asr_slice", None),
            source_text="".join(parts),
        ))
    return out


def transcribe_slices(
    audio_path: str,
    backend: str = _DEFAULT_BACKEND,
    asr_kwargs: Optional[dict] = None,
    output_dir: str = ".",
    slices=None,
    max_workers: int = 3,
) -> list[VTTBlock]:
    """Transcribe the VAD-produced speech slices into ``VTTBlock`` subtitles.

    Each block is one complete slice on the VAD slice timeline: ``start_ms`` /
    ``end_ms`` come from the slice's ``(start, end)`` — NOT from the ASR
    timestamps — and ``text`` is the concatenated recognized text of that slice.
    ``slices`` are the optimized speech regions from the pipeline's VAD stage
    (an empty list yields empty subtitles).

    Raises:
        ASRError: If the adapter is not configured or transcription fails.
    """
    asr_kwargs = dict(asr_kwargs or {})
    sentences = process_audio(
        audio_path,
        slices or [],
        language=asr_kwargs.get("language", "ja"),
        asr_kwargs=asr_kwargs,
        output_dir=output_dir,
        max_workers=max_workers,
        backend=backend,
    )
    return _group_sentences_into_slices(sentences, slices or [])
