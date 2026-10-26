"""Main pipeline orchestrator for ASMR dubbing."""

import asyncio
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .io import intermediate_dir, open_log, write_json, write_text
from .pipeline_logger import PipelineTee, print_progress
from .vtt_parser import parse_vtt
from .audio_splitter import get_audio_duration, split_audio
from .pink_segments import attach_pink_geometry
from .tts_engine import generate_tts
from .audio_joiner import (
    apply_follow_translation_cap, enforce_tts_budget, join_audio,
)
from .audio_loudness import align_tts_to_original
from .mode_builders import create_segment_builder
from .audio_handlers import build_tts_handler_chain
from .pool_manager import PoolConfig
from .error import PipelineError


def run_arrangement_pipeline(payload: dict) -> dict:
    """Rebuild the edited studio timeline from an immutable task snapshot."""
    from contextlib import redirect_stdout
    from uuid import uuid4
    from .arrangement import prepare_arrangement_plan
    from .arrangement_audio import create_arrangement_audio_adapter, write_arrangement_result
    work = Path(payload["workDir"])
    output = work / "arrangement/runs" / uuid4().hex
    log_path = output / "arrangement.log"
    with open_log(str(log_path)) as log:
        with redirect_stdout(PipelineTee(sys.stdout, log)):
            print("[arrangement] 开始编排任务，使用已保存配置快照")
            _required_stage("OUTPUT_WRITE_DENIED", write_json, str(output / "input.json"), payload["state"])
            print_progress("prepare", "读取编排配置与缓存", 0, 1)
            plan = _required_stage("AUDIO_DSP_FAILED", prepare_arrangement_plan,
                                   work, Path(payload["audioPath"]), payload["state"],
                                   payload["config"], payload["ttsKwargs"])
            print_progress("building", "处理编排片段", 0, 1)
            result = _required_stage("AUDIO_DSP_FAILED", create_arrangement_audio_adapter().render, plan)
            print_progress("concat", "重新拼接并导出", 0, 1)
            files = _required_stage("OUTPUT_WRITE_DENIED", write_arrangement_result, result, output, payload["state"])
            print(f"[arrangement] 重新编排完成：{files['output_mp3']}")
            print_progress("concat", "编排完成", 1, 1)
    return {**files, "log_path": str(log_path)}


def _required_stage(code: str, operation, *args, **kwargs):
    """Run one required stage without leaking implementation exceptions."""
    try:
        return operation(*args, **kwargs)
    except PipelineError:
        raise
    except PermissionError as error:
        from .error import ExportError, descriptor
        raise ExportError(descriptor("OUTPUT_WRITE_DENIED"), detail=str(error), cause=error) from error
    except OSError as error:
        from .error import AudioProcessingError, descriptor
        disk_full = getattr(error, "errno", None) == 28 or getattr(error, "winerror", None) == 112
        mapped = "OUTPUT_DISK_FULL" if disk_full else code
        raise AudioProcessingError(descriptor(mapped), detail=str(error), cause=error) from error
    except Exception as error:
        from .error import AudioProcessingError, descriptor
        raise AudioProcessingError(descriptor(code), detail=str(error), cause=error) from error


def _clean_previous_run(output_dir: str) -> int:
    """Remove final outputs while preserving resumable stage artifacts.

    Intermediate ASR, split, TTS, and audio-build directories intentionally
    survive a retry. Their current cache keys are positional; invalidation for
    changed source audio or synthesis settings is deferred until the cache
    contract carries content/configuration fingerprints.

    Returns the number of removed entries (0 when the directory was already clean).
    """
    removed = 0
    stale_files = ["_parsed_vtt.json", "output.wav", "output.mp3", "llm_subtitles/final.vtt",
                   "debug/high_quality_export.json"]
    stale_files.extend(
        entry for entry in os.listdir(output_dir)
        if re.fullmatch(r"pipeline_\d{8}_\d{6}\.log", entry)
    )
    for stale in stale_files:
        try:
            os.remove(os.path.join(output_dir, stale))
            removed += 1
        except OSError:
            pass
    return removed


def _load_cached_asr(output_dir: str, asr_kwargs: dict, expected_key: str | None = None) -> tuple[list, str, str] | None:
    """Load the completed ASR boundary used by the next pipeline stage.

    Japanese ASR resumes from the cleaned artifact because sentence detection
    and Japanese cleanup have already completed. Chinese ASR resumes from the
    raw artifact because it is already the final ASR-stage contract.
    """
    configured = str(
        asr_kwargs.get("output_language", asr_kwargs.get("language", "ja"))
    ).strip().lower()
    language = "zh" if configured.startswith("zh") else "ja"
    filename = "raw.vtt" if language == "zh" else "cleaned.vtt"
    path = os.path.join(output_dir, "asr_subtitles", filename)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return None
    if expected_key:
        from .task_cache import JsonResultCache, file_digest
        record = JsonResultCache(os.path.join(output_dir, "asr_stage_results")).load(expected_key)
        if not record or record.get("sha256") != file_digest(path):
            return None
    try:
        blocks = parse_vtt(path)
    except (OSError, ValueError):
        return None
    if not blocks:
        return None
    return blocks, language, path


def _create_japanese_cleaner_with_fallback(name: str):
    """Create the configured cleaner; configured implementations are required."""
    from .japanese_cleaning import (
        JapaneseCleanerUnavailableError,
        create_japanese_cleaner,
    )

    try:
        return create_japanese_cleaner(name)
    except JapaneseCleanerUnavailableError as error:
        from .error import ConfigurationError, descriptor
        raise ConfigurationError(
            descriptor("VAD_RUNTIME_UNAVAILABLE"),
            detail=f"Japanese cleaner {name} unavailable: {error}", cause=error,
            context={"backend": name},
        ) from error


def _format_vtt_timestamp(ms: int) -> str:
    """Format milliseconds as a WebVTT timestamp (HH:MM:SS.mmm)."""
    ms = max(0, int(ms))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def _write_optimized_subtitles(output_dir: str, blocks) -> str:
    """Write the (LLM-optimized) blocks as a standalone WebVTT file.

    The file lands under ``output_dir/llm_subtitles/optimized.vtt`` so it never
    collides with the pipeline's own artifacts. Speaker prefixes and any English
    emotion tags inside ``block.text`` are preserved as-is (tags are only
    stripped at TTS synthesis time, not here). Blocks flagged ``skip_tts``
    (interjection fragments whose translation TTS is skipped) get the marker
    ``【已跳过翻译】`` appended so they are identifiable at a glance.

    Returns the absolute path of the written file.
    """
    path = os.path.join(
        intermediate_dir(output_dir, "llm_subtitles"), "optimized.vtt"
    )
    lines = ["WEBVTT", ""]
    for b in blocks:
        lines.append(str(b.index + 1))
        lines.append(
            f"{_format_vtt_timestamp(b.start_ms)} --> {_format_vtt_timestamp(b.end_ms)}"
        )
        text = b.text or ""
        if b.speaker:
            text = f"{b.speaker}：{text}"
        if b.skip_tts:
            text = f"{text}【已跳过翻译】"
        lines.append(text)
        lines.append("")
    write_text(path, "\n".join(lines))
    return path


def _write_asr_subtitles(output_dir: str, blocks) -> str:
    """Write the raw ASR-generated blocks as a standalone WebVTT file.

    Lands under ``output_dir/asr_subtitles/raw.vtt``. Native speaker labels
    are retained as prefixes so exports and cache reloads keep diarization.
    """
    path = os.path.join(
        intermediate_dir(output_dir, "asr_subtitles"), "raw.vtt"
    )
    lines = ["WEBVTT", ""]
    for b in blocks:
        lines.append(str(b.index + 1))
        lines.append(
            f"{_format_vtt_timestamp(b.start_ms)} --> {_format_vtt_timestamp(b.end_ms)}"
        )
        lines.append(f"{b.speaker}：{b.text or ''}" if b.speaker else b.text or "")
        lines.append("")
    write_text(path, "\n".join(lines))
    return path


def _serialize(regions):
    return [
        {"start": r.start, "end": r.end, "duration": round(r.end - r.start, 4)}
        for r in regions
    ]


def write_vad_output(output_dir: str, raw_regions, final_slices, cfg: dict) -> str:
    """Write VAD analysis artifacts under ``output_dir/vad_output/``.

    Emits ``raw_regions.json`` / ``final_slices.json`` / ``config.json`` so a
    failed ASR run can be inspected without re-running the detector. Returns
    the ``vad_output`` directory path that was written.
    """
    dir_path = intermediate_dir(output_dir, "vad_output")
    write_json(
        os.path.join(dir_path, "raw_regions.json"),
        {"count": len(raw_regions), "regions": _serialize(raw_regions)},
    )
    write_json(
        os.path.join(dir_path, "final_slices.json"),
        {"count": len(final_slices), "slices": _serialize(final_slices)},
    )
    write_json(
        os.path.join(dir_path, "config.json"),
        {"vad_config": cfg},
    )

    return dir_path


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_cjk_chars(text: str) -> int:
    """Count Chinese characters in ``text`` (CJK Unified Ideographs only).

    Punctuation, whitespace, kana, Latin and digits are all excluded, so the
    result is exactly "the number of Chinese characters after removing
    punctuation" — the rule used to detect interjection fragments.
    """
    return len(_CJK_RE.findall(text or ""))


def mark_interjection_skips(blocks, max_chars: int = 3) -> int:
    """Flag blocks whose translated text is too short to dub.

    A block whose text holds ≤ ``max_chars`` Chinese characters (after removing
    punctuation) is mostly a 语气词/短叹 (e.g. "嗯", "啊，呀！") — set
    ``block.skip_tts = True`` so the pipeline skips its translation TTS and keeps
    the original audio for that span. Returns the number of blocks flagged.
    """
    flagged = 0
    for b in blocks:
        if not b.skip_tts and count_cjk_chars(b.text) <= max_chars:
            b.skip_tts = True
            flagged += 1
    return flagged


# Keys of the run_pipeline vad_* parameters forwarded to optimize_asmr_segments.
_VAD_OPTIMIZE_KEYS = (
    "min_sec", "target_min_sec", "target_max_sec", "max_sec",
    "max_merge_gap_sec", "split_window_sec", "split_overlap_sec",
)


def _vad_optimize_kwargs(vad_cfg: dict) -> dict:
    """Select the ``optimize_asmr_segments`` tuning keys present in the vad config."""
    return {k: vad_cfg[k] for k in _VAD_OPTIMIZE_KEYS if k in vad_cfg}


def _run_vad_stage(audio_path: str, vad_cfg: dict, output_dir: str) -> list:
    """Detect speech regions, optimize them into ASR slices, and persist artifacts.

    Runs FSMN-VAD on the audio, merges short fragments into ASR-sized slices
    (the 1500ms split / 25s cap come from run_pipeline's vad_* parameters), and
    writes the raw regions + final slices under ``output_dir/vad_output/``.
    Returns the final slices. An unavailable/disabled VAD aborts the no-subtitle
    path (报错终止).
    """
    from .error import ASRError
    from .vad import create_vad, optimize_asmr_segments

    vad_engine = create_vad(vad_cfg)
    if not vad_engine.available:
        reason = getattr(vad_engine, "reason", "未知原因")
        raise ASRError(
            "VAD 不可用或未启用{reason}；"
            "无法生成 ASR 切片"
        )
    print_progress("asr", "正在检测人声区间", 0, 1)
    raw_regions = asyncio.run(vad_engine.detect(audio_path))
    print_progress("asr", "正在检测人声区间", 1, 1)
    final_slices = optimize_asmr_segments(
        raw_regions,
        audio_duration_sec=get_audio_duration(audio_path),
        **_vad_optimize_kwargs(vad_cfg),
    )
    write_vad_output(output_dir, raw_regions, final_slices, vad_cfg)
    print(f"       [asr] VAD 检出 {len(raw_regions)} 个人声区间，"
          f"切片优化为 {len(final_slices)} 片（vad_output/ 已写入）")
    return final_slices


def run_pipeline(
    audio_path: str,
    vtt_path: str,
    output_dir: str,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    crossfade_ms: int = 10,
    retain_original: bool = False,
    sync_mode: str = "none",
    tts_volume: float = 1.0,
    speaker_refs: Optional[dict] = None,
    tts_backend: str = "edge-tts",
    tts_backend_kwargs: Optional[dict] = None,
    tts_max_concurrent: int = 0,
    num_workers: int = 0,
    split_workers: int = 0,
    build_workers: int = 0,
    phantom_mode: bool = False,
    phantom_delay_ms: int = 400,
    phantom_reverb_mix: float = 0.4,
    phantom_lowpass_hz: int = 3800,
    phantom_orig_fade_ms: int = 0,
    phantom_tts_fade_ms: int = 0,
    phantom_virtual_fade: bool = False,
    follow_translation_overhang_ms: int = 3000,
    tts_check_artifact: bool = True,
    follow_original_rate: float = 1.15,
    llm_kwargs: Optional[dict] = None,
    asr_kwargs: Optional[dict] = None,
    asr_backend: str = "fish-asr",
    asr_max_concurrent: int = 3,
    japanese_cleaner: str = "mecab-regex",
    vad_enabled: bool = True,
    vad_max_end_silence_time: int = 1500,
    vad_max_single_segment_time: int = 25000,
    vad_min_sec: float = 0.3,
    vad_target_min_sec: float = 5.0,
    vad_target_max_sec: float = 25.0,
    vad_max_sec: float = 25.0,
    vad_max_merge_gap_sec: float = 2.0,
    vad_split_window_sec: float = 20.0,
    vad_split_overlap_sec: float = 4.0,
    vad_sentence_realignment_beta: bool = False,
    sentence_aligner: str = "joint-vad",
    qwen_align_endpoint: str = "http://127.0.0.1:8007",
    vad_backend: str = "fsmn-vad",
    vad_endpoint: str = "http://127.0.0.1:8006",
    vad_threshold: float = 0.5,
    sentence_detector: str = "mecab-japanese",
    detect_enabled: Optional[bool] = None,
) -> dict:
    from .detect import DetectConfig, create_detect_stage, detect_eligible
    from .vad.stage import run_analysis_stage
    if not 0 < vad_threshold < 1:
        raise ValueError("VAD threshold must be between 0 and 1")
    detect_config = DetectConfig(aligner=sentence_aligner, qwen_endpoint=qwen_align_endpoint,
                                 atom_detector="asmr-vad" if vad_backend == "asmr-vad" else "native-vad",
                                 vad_endpoint=vad_endpoint, vad_threshold=vad_threshold,
                                 sentence_detector=sentence_detector)
    detect_enabled = vad_sentence_realignment_beta if detect_enabled is None else detect_enabled
    detect_stage = None
    detect_summary = None
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    resume_cache_present = any(
        os.path.isdir(os.path.join(output_dir, name))
        for name in ("asr_subtitles", "chunk_0")
    )
    cleaned_count = _clean_previous_run(output_dir)
    start_time = time.time()

    # ---- Global resource budget (never multiplied by a chunk count) ----
    pool_cfg = PoolConfig(
        requested_chunks=num_workers,
        split_workers=split_workers,
        build_workers=build_workers,
        tts_concurrency=tts_max_concurrent,
        tts_backend=tts_backend,
    )
    # ---- Set up run log ----
    # Development and packaged runs use the same artifact layout. Keeping the
    # run log beside the audio's other outputs also isolates concurrent tasks.
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = Path(output_dir) / log_filename
    log_fp = open_log(log_path, encoding="utf-8")
    _original_stdout = sys.stdout
    tee = PipelineTee(_original_stdout, log_fp)
    sys.stdout = tee

    def _close_log():
        sys.stdout = tee.close()

    s = 0
    step_times: list[tuple[str, float]] = []

    try:
        print("=" * 60)
        print("  ASMR Dubbing Pipeline")
        print(f"  Log: {log_path}")
        print(f"  Resources: {pool_cfg}")
        print("=" * 60)

        if resume_cache_present:
            print("[resume] 检测到上次任务缓存，将复用已完成的 ASR、TTS 与音频处理片段。")
        if cleaned_count:
            print(f"[cleanup] 已清理 {cleaned_count} 个旧的最终输出或日志文件。")

        total = 1 if tts_backend == "subtitles-only" else 3

        def ps(msg):
            nonlocal s
            s += 1
            if step_times:
                step_times[-1] = (step_times[-1][0], time.time() - step_times[-1][1])
            print(f"\n[{s}/{total}] {msg}")
            step_times.append((msg, time.time()))

        def _fmt_step(label: str, elapsed: float) -> str:
            return f"       [{label}] {elapsed:.1f}s"

        # ---- Step 1 ----
        ps("Preparing subtitles (VTT / ASR)...")
        asr_output_language = None
        asr_preserve_timeline = False

        from .task_cache import JsonResultCache, cache_key, file_digest, public_settings
        asr_cache_key = None
        if not vtt_path and os.path.isfile(audio_path):
            asr_cache_key = cache_key({
                "audio": file_digest(audio_path), "backend": asr_backend,
                "settings": public_settings(asr_kwargs), "cleaner": japanese_cleaner,
                "detect": [detect_enabled, sentence_aligner, sentence_detector, qwen_align_endpoint],
                "vad": [vad_enabled, vad_backend, vad_endpoint, vad_threshold, vad_max_end_silence_time, vad_max_single_segment_time,
                        vad_min_sec, vad_target_min_sec, vad_target_max_sec, vad_max_sec,
                        vad_max_merge_gap_sec, vad_split_window_sec, vad_split_overlap_sec],
            })
        cached_asr = _load_cached_asr(output_dir, asr_kwargs, asr_cache_key) if not vtt_path else None
        if cached_asr is not None:
            print_progress("asr", "正在恢复字幕缓存", 0, 1)
            blocks, asr_output_language, cached_asr_path = cached_asr
            if asr_cache_key:
                cached_metadata = JsonResultCache(os.path.join(output_dir, "asr_stage_results")).load(asr_cache_key) or {}
                asr_preserve_timeline = cached_metadata.get("preserve_source_timeline") is True
            print(f"       [cached-asr] 复用 {len(blocks)} 个字幕块: {cached_asr_path}")
            print_progress("asr", "正在恢复字幕缓存", 1, 1)
        elif vtt_path:
            print_progress("parsing", "正在解析字幕", 0, 1)
            blocks = parse_vtt(vtt_path)
            print(f"       Found {len(blocks)} subtitle blocks")
            print_progress("parsing", "正在解析字幕", 1, 1)
        else:
            # No subtitle file — run VAD (detect speech → optimize into ASR
            # slices → persist vad_output/), then ASR-transcribe the slices into
            # subtitles; the flow then continues identically to the has-subtitle
            # path. An unavailable VAD aborts the run (报错终止).
            if not asr_kwargs:
                raise ValueError(
                    "未提供字幕文件且未配置 ASR 参数（--asr-kwargs），无法继续。"
                )
            # The VAD tuning lives on run_pipeline's vad_* parameters, so the
            # engine config passed here carries only the ASR/transcription knobs.
            vad_cfg = {
                "enabled": vad_enabled,
                "backend": vad_backend,
                "endpoint": vad_endpoint,
                "threshold": vad_threshold,
                "max_end_silence_time": vad_max_end_silence_time,
                "max_single_segment_time": vad_max_single_segment_time,
                "min_sec": vad_min_sec,
                "target_min_sec": vad_target_min_sec,
                "target_max_sec": vad_target_max_sec,
                "max_sec": vad_max_sec,
                "max_merge_gap_sec": vad_max_merge_gap_sec,
                "split_window_sec": vad_split_window_sec,
                "split_overlap_sec": vad_split_overlap_sec,
            }
            print_progress("asr", "正在生成字幕", 0, 1)
            from .error import ASRError
            from .asr_engine import transcribe_slices
            try:
                from .asr import DocumentASRAdapter, create_asr
                adapter = create_asr(asr_backend, **asr_kwargs)
                document_adapter = adapter if isinstance(adapter, DocumentASRAdapter) else None
                needs_project_vad = document_adapter is None or document_adapter.uses_project_vad()
                if detect_eligible(
                    detect_enabled,
                    vtt_path,
                    asr_backend,
                    asr_kwargs.get(
                        "output_language", asr_kwargs.get("language", "ja")
                    ),
                    project_vad=needs_project_vad,
                ):
                    detect_stage = create_detect_stage(detect_config, audio_path)
                    final_slices, _ = run_analysis_stage(
                        audio_path, vad_cfg, output_dir)
                else:
                    final_slices = _run_vad_stage(audio_path, vad_cfg, output_dir) if needs_project_vad else []
                if needs_project_vad and not final_slices:
                    from .error import VadError, descriptor
                    raise VadError(descriptor("VAD_NO_SPEECH"))
                elif document_adapter is not None:
                    from .asr.no_speech import ASRNoSpeechError
                    try:
                        result = asyncio.run(document_adapter.transcribe_document(audio_path, final_slices))
                    except ASRNoSpeechError:
                        blocks = []
                        asr_output_language = asr_kwargs.get(
                            "output_language", asr_kwargs.get("language", "ja")
                        )
                        print("       [asr] 整段音频未识别到文字，按空字幕继续")
                    else:
                        from .asr.document_artifacts import write_document_artifacts
                        write_document_artifacts(result, output_dir)
                        blocks = result.blocks
                        asr_preserve_timeline = getattr(result, "preserve_source_timeline", False) is True
                        asr_output_language = result.language
                        print(f"       [asr] 服务来源 {result.source}，时间轴 {result.timeline_source}")
                else:
                    blocks = transcribe_slices(
                        audio_path, asr_backend, asr_kwargs,
                        output_dir, final_slices, max_workers=asr_max_concurrent,
                    )
                    asr_output_language = "ja"
            except ASRError:
                raise
            print(f"       ASR 生成 {len(blocks)} 个字幕块")
            print_progress("asr", "正在生成字幕", 1, 1)
            try:
                asr_vtt = _write_asr_subtitles(output_dir, blocks)
                print(f"       [asr] 原始字幕已保存: {asr_vtt}")
            except Exception as e:
                print(f"       [warn] ASR 字幕保存失败: {e}")

            if asr_output_language == "ja":
                if detect_stage is not None:
                    from .detect.contracts import DetectStageInput

                    detected = detect_stage.run(
                        DetectStageInput(tuple(blocks)), output_dir
                    )
                    blocks = list(detected.blocks)
                    detect_summary = detected.summary

                # Japanese cleanup consumes only Japanese ASR output. Empty
                # blocks remain on the authored timeline with skip_tts=True.
                from .japanese_cleaning import (
                    write_cleaning_artifacts,
                )
                cleaner = _create_japanese_cleaner_with_fallback(japanese_cleaner)
                cleaning_result = cleaner.clean(blocks)
                blocks = cleaning_result.blocks
                artifacts = write_cleaning_artifacts(cleaning_result, output_dir)
                print(
                    f"       [japanese-clean] {cleaner.name}: 原始 {len(cleaning_result.records)} 段，"
                    f"保留 {cleaning_result.kept_count} 段，"
                    f"清洗后为空 {cleaning_result.skipped_count} 段（保留原声）"
                )
                print(f"       [japanese-clean] 清洗字幕: {artifacts['vtt']}")
                print(f"       [japanese-clean] 清洗报告: {artifacts['report']}")
            elif asr_output_language == "zh":
                print("       [asr] 海南鸡已直出中文，跳过日语清洗、总结和翻译")
            else:
                raise ValueError(f"ASR 返回了不支持的语言：{asr_output_language}")

            if asr_cache_key:
                asr_preserve_timeline = asr_preserve_timeline or detect_stage is not None
                filename = "raw.vtt" if asr_output_language == "zh" else "cleaned.vtt"
                cached_path = os.path.join(output_dir, "asr_subtitles", filename)
                if os.path.isfile(cached_path):
                    JsonResultCache(os.path.join(output_dir, "asr_stage_results")).save(
                        asr_cache_key, {"sha256": file_digest(cached_path),
                                        "preserve_source_timeline": asr_preserve_timeline})

        speakers = set(b.speaker for b in blocks if b.speaker)
        if speakers:
            print(f"       Detected speakers: {', '.join(speakers)}")

        # Enabled LLM stages are part of the result contract and fail atomically.
        # With a real VTT this is the plain phase-1/phase-2 flow; on the ASR path
        # the LLM first translates + splits each slice into Chinese sentences.
        enhance_stats = None
        if llm_kwargs:
                # Inject the TTS backend/model so the phase-2 prompt knows which
                # voice tags it may emit (fish [tags] / minimax-2.8 (tags) / none).
            llm_ctx = dict(llm_kwargs)
            llm_ctx.setdefault("tts_backend", tts_backend)
            llm_ctx.setdefault("tts_model", (tts_backend_kwargs or {}).get("model", ""))
            if vtt_path or asr_output_language == "zh":
                from .llm_enhancer import enhance_blocks
                enhance_stats = enhance_blocks(blocks, llm_ctx, output_dir=output_dir)
            else:
                from .llm_enhancer import enhance_asr_blocks
                blocks, enhance_stats = enhance_asr_blocks(
                    blocks, llm_ctx, preserve_source_timeline=detect_stage is not None or asr_preserve_timeline,
                    output_dir=output_dir)

        if not blocks:
            if tts_backend == "subtitles-only":
                from .error import ExportError, descriptor
                raise ExportError(descriptor("SUBTITLE_FINAL_EMPTY"))
            # Nothing to dub: an empty subtitle file, empty ASR output, or every
            # ASR slice discarded for lacking a Chinese translation. Abort with a
            # clear message rather than crash on zero-sized chunk math below.
            raise ValueError(
                "没有可用于配音的字幕块（字幕为空，或 ASR 片段全部未能翻译为中文而被丢弃）。"
            )

        if tts_backend == "subtitles-only":
            from .subtitles.final_export import build_final_subtitles, write_final_vtt
            from .run_results import record_run
            print_progress("subtitle.export", "正在生成最终字幕", 0, 1)
            cues = _required_stage("SUBTITLE_FINAL_EMPTY", build_final_subtitles,
                                   blocks, include_skipped=True)
            final_vtt = _required_stage("OUTPUT_WRITE_DENIED", write_final_vtt, cues, output_dir)
            run_result = _required_stage(
                "OUTPUT_WRITE_DENIED", record_run, output_dir, "succeeded", audio_path,
                output_mode="subtitles-only")
            print_progress("subtitle.export", "正在生成最终字幕", 1, 1)
            elapsed = time.time() - start_time
            print(f"       [subtitle.export] {len(cues)} cues → {final_vtt}")
            print(f"  Pipeline complete!\n  Subtitle: {final_vtt}\n  Time: {elapsed:.1f} seconds")
            return {
                "final_vtt": final_vtt, "run_result": run_result,
                "num_segments": 0, "speakers": list(speakers),
                "elapsed_seconds": elapsed, "log_path": str(log_path),
                "blocks": blocks, "segments": [], "detect_summary": detect_summary,
                "vad_realignment_summary": detect_summary,
            }

        # Interjection blocks: a translation with ≤3 Chinese characters (after
        # removing punctuation) is mostly a 语气词/短叹 — skip its translation TTS
        # and keep the original audio for that span.
        skipped = mark_interjection_skips(blocks)
        if skipped:
            print(f"       [skip-tts] {skipped} 个语气词/短句片段将跳过翻译 TTS（保留原声）")

        _required_stage("OUTPUT_WRITE_DENIED", write_json,
            os.path.join(intermediate_dir(output_dir, "debug"), "_parsed_vtt.json"), [{
                "index": b.index, "start_ms": b.start_ms, "end_ms": b.end_ms,
                "speaker": b.speaker, "text": b.text,
                "source_text": getattr(b, "source_text", ""),
                "skip_tts": getattr(b, "skip_tts", False),
            } for b in blocks])

        # Export the optimized subtitles as a standalone WebVTT file (only when
        # the LLM actually rewrote something; otherwise there is nothing to show).
        if enhance_stats and enhance_stats.get("enabled") and (
            enhance_stats.get("rewritten", 0) > 0 or enhance_stats.get("translated", 0) > 0
        ):
            try:
                opt_path = _write_optimized_subtitles(output_dir, blocks)
                print(f"       [llm] 优化字幕已导出: {opt_path}")
            except Exception as e:
                print(f"       [warn] 优化字幕导出失败，已跳过: {e}")

        # Pink-segment geometry: divide the whole audio timeline into continuous,
        # seamless slices (one per block) so the slicer keeps the natural pauses
        # and SE between lines in every path — this replaces the old ASR-only
        # align_asr_times re-spanning. Geometry is attached before the global
        # slicer runs so every output span uses the same complete timeline.
        try:
            total_duration_ms = int(get_audio_duration(audio_path) * 1000)
        except Exception as e:
            print(f"       [warn] 获取音频时长失败，按末段结束时间对齐: {e}")
            total_duration_ms = max(b.end_ms for b in blocks)
        attach_pink_geometry(blocks, total_duration_ms)
        print(f"       [pink] 已按pink切片连续覆盖 {total_duration_ms}ms（{len(blocks)} 片）")

        # ---- Step 2: Global staged processing ----
        # One project-wide split/TTS/build flow prevents worker counts from being
        # multiplied by the old chunk pool (8 chunks × 16 ffmpeg processes, etc.).
        builder = create_segment_builder(
            phantom_mode=phantom_mode,
            retain_original=retain_original,
            sync_mode=sync_mode,
            phantom_delay_ms=phantom_delay_ms,
            phantom_reverb_mix=phantom_reverb_mix,
            phantom_lowpass_hz=phantom_lowpass_hz,
            gap_ms=200,
            follow_original_rate=follow_original_rate,
            phantom_orig_fade_ms=phantom_orig_fade_ms,
            phantom_tts_fade_ms=phantom_tts_fade_ms,
            phantom_virtual_fade=phantom_virtual_fade,
        )
        mode_desc = builder.name
        ps(f"Processing {len(blocks)} segments with global pools ({mode_desc} mode)...")

        # Keep the historical chunk_0 directory name so cleanup and packaged I/O
        # filtering remain compatible with existing projects.
        work_dir = intermediate_dir(output_dir, "chunk_0")
        os.makedirs(work_dir, exist_ok=True)

        segments = _required_stage("AUDIO_SPLIT_FAILED", split_audio,
            audio_path,
            blocks,
            os.path.join(work_dir, "01_segments"),
            max_workers=pool_cfg.split_workers,
        )

        def _tts_progress_callback(done, total_count):
            print_progress("tts", "正在生成翻译片段", done, total_count)

        segments = generate_tts(
            segments,
            os.path.join(work_dir, "02_tts"),
            rate=tts_rate,
            pitch=tts_pitch,
            max_concurrent=pool_cfg.tts_concurrency,
            backend=tts_backend,
            backend_kwargs=tts_backend_kwargs,
            progress_callback=_tts_progress_callback,
        )

        missing = [
            seg["index"] for seg in segments
            if not seg.get("tts_file") and not seg.get("skip_tts")
        ]
        if missing:
            from .error import TtsError, descriptor
            raise TtsError(descriptor("TTS_OUTPUT_INVALID"), detail=f"missing segments: {missing}")

        handler_chain = build_tts_handler_chain(
            detect_artifact=tts_check_artifact,
            phantom_mode=phantom_mode,
            tts_backend=tts_backend,
            tts_backend_kwargs=tts_backend_kwargs,
            tts_rate=tts_rate,
            tts_pitch=tts_pitch,
            phantom_delay_ms=phantom_delay_ms,
            phantom_reverb_mix=phantom_reverb_mix,
            phantom_lowpass_hz=phantom_lowpass_hz,
        )
        segments = _required_stage("AUDIO_DSP_FAILED", handler_chain.run,
            segments,
            os.path.join(work_dir, "02_tts"),
            max_workers=pool_cfg.build_workers,
        )
        segments = _required_stage("AUDIO_DSP_FAILED", align_tts_to_original,
            segments, tts_volume=tts_volume, max_workers=pool_cfg.build_workers
        )
        segments = _required_stage("AUDIO_DSP_FAILED", enforce_tts_budget,
            segments,
            max_speed=min(1.25, follow_original_rate),
            mode="follow_translation" if sync_mode == "follow_translation"
            else "follow_original",
            max_workers=pool_cfg.build_workers,
        )

        for seg in segments:
            seg["vc_file"] = seg.get("tts_file")
            if seg.get("skip_tts"):
                seg["vc_file"] = seg.get("segment_file")
                print(f"       [skip-tts] seg {seg['index']}: 语气词片段，跳过翻译（保留原声）")

        segments = _required_stage("AUDIO_DSP_FAILED", builder.build,
            segments,
            os.path.join(work_dir, builder.subdir),
            max_workers=pool_cfg.build_workers,
        )
        if sync_mode == "follow_translation":
            segments = _required_stage("AUDIO_DSP_FAILED", apply_follow_translation_cap,
                segments, max_overhang_ms=follow_translation_overhang_ms
            )
        all_segments = segments
        print(f"       Global stages complete, {len(all_segments)} total segments")

        # ---- Step 3: Join project output once ----
        print_progress("building", "正在合成音频", 1, 1)
        ps("Joining processed segments into final audio...")
        print_progress("concat", "正在合并最终文件", 0, 1)
        final_output = os.path.join(output_dir, "output.wav")
        joined = _required_stage("AUDIO_JOIN_FAILED", join_audio,
                                 all_segments, final_output, crossfade_ms=crossfade_ms,
                                 return_timeline=True)
        print_progress("concat", "正在合并最终文件", 1, 1)

        # Record last step time
        step_times[-1] = (step_times[-1][0], time.time() - step_times[-1][1])

        mp3_output = final_output.replace(".wav", ".mp3")
        if (not os.path.isfile(final_output) or os.path.getsize(final_output) <= 0
                or not os.path.isfile(mp3_output) or os.path.getsize(mp3_output) <= 0):
            from .error import ExportError, descriptor
            raise ExportError(descriptor("AUDIO_JOIN_FAILED"),
                              detail="final WAV or MP3 is missing/empty")
        from .audio_metadata import create_metadata_reader, write_audio_metadata
        source_metadata = _required_stage(
            "AUDIO_METADATA_FAILED", create_metadata_reader().read, audio_path)
        metadata_report = _required_stage(
            "AUDIO_METADATA_FAILED", write_audio_metadata, source_metadata,
            (mp3_output, final_output), intermediate_dir(output_dir, "debug"))
        file_size = os.path.getsize(mp3_output) / (1024 * 1024)
        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"  Pipeline complete!")
        print(f"  Output: {mp3_output}")
        print(f"  WAV:    {final_output}")
        print(f"  Size:   {file_size:.1f} MB")
        print(f"  Time:   {elapsed:.1f} seconds")
        print(f"  Log:    {log_path}")
        print(f"\n  Per-step timing:")
        for label, t in step_times:
            print(f"    {t:6.1f}s  {label}")
        print(f"    {elapsed:6.1f}s  TOTAL")
        print(f"{'=' * 60}")

        result = {
            "output_mp3": mp3_output,
            "output_wav": final_output,
            "metadata_report": metadata_report,
            "num_segments": len(all_segments),
            "speakers": list(speakers),
            "elapsed_seconds": elapsed,
            "log_path": str(log_path),
            "blocks": blocks,
            "segments": all_segments,
            "detect_summary": detect_summary,
            "vad_realignment_summary": detect_summary,
        }
        from .subtitles.final_export import build_aligned_subtitles, write_final_vtt
        try:
            cues = build_aligned_subtitles(blocks, joined.placements)
            result["final_vtt"] = write_final_vtt(cues, output_dir)
            print(f"       [subtitle.export] {len(cues)} cues → {result['final_vtt']}")
        except Exception as error:
            from .error import ExportError, descriptor
            raise ExportError(descriptor("SUBTITLE_FINAL_EMPTY"), detail=str(error), cause=error) from error
        from .high_quality_export import write_export_recipe
        result["high_quality_recipe"] = _required_stage(
            "OUTPUT_WRITE_DENIED", write_export_recipe, audio_path, all_segments,
            builder.export_settings(), output_dir, crossfade_ms, follow_translation_overhang_ms)
        from .run_results import record_run
        result["run_result"] = _required_stage(
            "OUTPUT_WRITE_DENIED", record_run, output_dir, "succeeded", audio_path)
        return result

    except Exception as raw_error:
        import traceback
        from .error import unexpected_error
        error = unexpected_error(raw_error)
        error.log_path = str(log_path)
        print(f"\n!!! Pipeline failed:\n{traceback.format_exc()}")
        raise error
    finally:
        _close_log()
