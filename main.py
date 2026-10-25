#!/usr/bin/env python3
"""
ASMRTranslator - Japanese ASMR + Chinese VTT → Chinese Dubbing

Usage:
    python main.py <audio_file> <vtt_file> [--rate +0%] [--output ./output]

Example:
    python main.py "../asmr-demo/RJ01617941/mp3_NoSE/1.你还会尽情地陪我们玩吧？♡_NoSE.mp3" \
                   "../asmr-demo/RJ01617941/mp3_NoSE/1.你还会尽情地陪我们玩吧？♡_NoSE.mp3.vtt"
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Force UTF-8 stdout/stderr. The PyInstaller console bootloader reconfigures the
# streams to the system ANSI codepage (GBK on Chinese Windows), which garbles
# Chinese text when the Electron host decodes the output as UTF-8.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")


def _load_dotenv(env_path=None):
    """Load config file (.asmrts in the user home dir) into environment variables.

    Uses os.environ.setdefault so existing env vars take precedence.
    """
    if env_path is None:
        env_path = os.path.join(os.path.expanduser("~"), ".asmrts")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if len(value) > 1 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# Load config before any other imports that may read environment variables
_load_dotenv()

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.pipeline import run_pipeline
from modules.error import ConfigurationError, PipelineError, descriptor, unexpected_error
from modules.error_reporter import report_error_async


def _emit_error(error: PipelineError) -> None:
    print("[PIPELINE_ERROR]" + json.dumps(
        error.to_payload(log_path=error.log_path), ensure_ascii=False, separators=(",", ":")
    ), file=sys.stderr, flush=True)


def _emit_result(result: dict) -> None:
    files = {
        "mp3": result.get("output_mp3", ""),
        "wav": result.get("output_wav", ""),
        "finalVtt": result.get("final_vtt", ""),
        "log": result.get("log_path", ""),
    }
    files = {key: value for key, value in files.items() if value}
    payload = {
        "contractVersion": 1, "success": True, "files": files,
        "runResult": result.get("run_result", ""),
        "vadRealignmentSummary": result.get("vad_realignment_summary"),
    }
    print("[PIPELINE_RESULT]" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="ASMRTranslator - Japanese ASMR → Chinese Dubbing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3 subtitles.vtt
  %(prog)s audio.mp3 subtitles.vtt --rate -5%% --output ./my_output
        """,
    )
    parser.add_argument("audio", nargs="?", help="Path to original ASMR audio file")
    parser.add_argument("--arrangement-editor", action="store_true", help="Project arrangement editor stdin protocol")
    parser.add_argument("--arrangement-run", action="store_true", help="Run saved arrangement snapshot from stdin")
    parser.add_argument("vtt", nargs="?", help="Path to Chinese subtitle VTT file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: ./output/<filename>)")
    parser.add_argument("--rate", default="+0%",
                        help="TTS speed adjustment: '+10%%' faster, '-10%%' slower (default: +0%%)")
    parser.add_argument("--crossfade", type=int, default=10,
                        help="Crossfade between segments in ms (default: 10, 0=off)")
    parser.add_argument("--speaker-rate", type=str, default=None,
                        help="Per-speaker rate, format: '遥:+5%%,瑠衣:-3%%'")
    parser.add_argument("--retain-original", action="store_true",
                        help="Retain original Japanese audio, append Chinese TTS after each segment")
    parser.add_argument("--tts-volume", type=float, default=1.0,
                        help="TTS loudness relative to the matched original segment: "
                             "1.0=match original loudness, 0.5=half, 0=silence (default: 1.0)")
    parser.add_argument("--sync-mode", type=str, default="none",
                        choices=["none", "follow_original", "follow_translation"],
                        help="Audio sync mode (requires --retain-original): "
                             "'follow_original' = TTS sped up to fit orig duration, "
                             "'follow_translation' = orig padded to fit TTS (default: none)")
    parser.add_argument("--tts-backend", type=str, default="edge-tts",
                        choices=["subtitles-only", "edge-tts", "mimo-tts", "fish-audio", "minimax-tts", "indextts-2.5", "common-tts"],
                        help="TTS backend to use (default: edge-tts)")
    parser.add_argument("--tts-backend-kwargs", type=str, default=None,
                        help='Backend-specific kwargs as JSON, e.g. '
                             '{"api_key":"...","voice_description":"..."}')
    parser.add_argument("--tts-backend-kwargs-stdin", action="store_true",
                        help="Read common-tts constructor configuration as UTF-8 JSON from stdin")
    parser.add_argument("--tts-pitch", default="+0Hz",
                        help="TTS pitch adjustment: '+10Hz' higher, '-20Hz' lower "
                             "(default: +0Hz). Only Edge TTS supports pitch.")
    parser.add_argument("--list-tts-voices", action="store_true",
                        help="Print the available voices for --tts-backend as JSON and exit")
    parser.add_argument("--check-vad-runtime", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--check-japanese-runtime", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--export-high-quality", metavar="RECIPE", help=argparse.SUPPRESS)
    parser.add_argument("--preflight", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--design-voice", action="store_true",
                        help="MiniMax 音色设计：按文本描述生成自定义音色并试听，打印 "
                             "{\"voice_id\",\"trial_audio\"} 后退出（需 --design-prompt / "
                             "--design-preview-text / --tts-backend-kwargs 的 api_key）")
    parser.add_argument("--design-prompt", type=str, default="",
                        help="音色设计：音色描述文本")
    parser.add_argument("--design-preview-text", type=str, default="",
                        help="音色设计：试听音频文本（≤500 字符）")
    parser.add_argument("--design-voice-id", type=str, default="",
                        help="音色设计：自定义生成音色 ID（可选，留空自动生成）")
    parser.add_argument("--clone-voice", action="store_true",
                        help="MiniMax 音色快速复刻：上传音频并复刻音色，打印 "
                             "{\"voice_id\",\"demo_audio\"} 后退出（需 --clone-file / "
                             "--tts-backend-kwargs 的 api_key；可选 --clone-voice-id）")
    parser.add_argument("--clone-file", type=str, default="",
                        help="音色复刻：待复刻音频路径（mp3/m4a/wav，10s-5min，≤20MB）")
    parser.add_argument("--clone-voice-id", type=str, default="",
                        help="音色复刻：自定义克隆音色 ID（8-256 位、字母开头；"
                             "缺省自动生成）")
    parser.add_argument("--clone-text", type=str, default="",
                        help="音色复刻：复刻试听文本（≤1000 字符，可选；提供时按 T2A 计费）")
    parser.add_argument("--clone-prompt-file", type=str, default="",
                        help="音色复刻：增强相似度的示例音频（mp3/m4a/wav，<8s，可选）")
    parser.add_argument("--clone-prompt-text", type=str, default="",
                        help="音色复刻：增强示例音频的对应文本（需与示例音频同时提供）")
    parser.add_argument("--clone-text-validation", type=str, default="",
                        help="音色复刻：主参考音频的预期文本（≤200 字符，可选）")
    parser.add_argument("--clone-accuracy", type=float, default=None,
                        help="音色复刻：ASR 文本相似度阈值 [0,1]（默认 0.7）")
    parser.add_argument("--clone-noise-reduction", action="store_true",
                        help="音色复刻：开启参考音频降噪")
    parser.add_argument("--clone-volume-normalization", action="store_true",
                        help="音色复刻：开启参考音频音量归一化")
    parser.add_argument("--list-voices", action="store_true",
                        help="MiniMax 音色管理：列出用户自己生成的全部音色，打印 "
                             "{\"voices\":[...]} 后退出（仅用户音色，不含系统音色）")
    parser.add_argument("--delete-voice", type=str, default="",
                        help="MiniMax 音色管理：删除指定 voice_id（配合 "
                             "--delete-voice-type voice_cloning|voice_generation）")
    parser.add_argument("--delete-voice-type", type=str, default="voice_cloning",
                        choices=["voice_cloning", "voice_generation"],
                        help="音色删除类型（默认 voice_cloning）")
    parser.add_argument("--list-fish-voices", action="store_true",
                        help="Fish Audio 音色管理：列出当前账号拥有的全部音色")
    parser.add_argument("--delete-fish-voice", type=str, default="",
                        help="Fish Audio 音色管理：删除指定 reference_id")
    parser.add_argument("--design-fish-voice", action="store_true",
                        help="Fish Audio 音色设计：生成候选并保存为私人音色")
    parser.add_argument("--clone-fish-voice", action="store_true",
                        help="Fish Audio 音色克隆：从参考音频创建私人音色")
    parser.add_argument("--fish-voice-title", type=str, default="",
                        help="Fish Audio 新音色名称")
    parser.add_argument("--fish-voice-description", type=str, default="",
                        help="Fish Audio 音色描述")
    parser.add_argument("--fish-design-instruction", type=str, default="",
                        help="Fish Audio 音色设计提示词")
    parser.add_argument("--fish-preview-text", type=str, default="",
                        help="Fish Audio 音色设计试听文本或克隆音频对应文本")
    parser.add_argument("--fish-language", type=str, default="zh",
                        help="Fish Audio 音色设计语言提示（zh/en/ja）")
    parser.add_argument("--fish-clone-file", type=str, default="",
                        help="Fish Audio 音色克隆参考音频")
    parser.add_argument("--tts-max-concurrent", type=int, default=0,
                        help="Max global concurrent TTS requests (0 = auto)")
    parser.add_argument("--split-workers", type=int, default=0,
                        help="Reserved compatibility option (0 = single-pass auto)")
    parser.add_argument("--build-workers", type=int, default=0,
                        help="Global audio post-processing workers (0 = auto)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Legacy compatibility option; pipeline stages are global")
    parser.add_argument("--phantom-mode", action="store_true",
                        help="Phantom Whisper mode: quiet, delayed, reverb-washed Chinese "
                             "overlaid on the untouched stereo original (implies --retain-original)")
    parser.add_argument("--phantom-delay", type=int, default=400,
                        help="Delay in ms before the phantom voice surfaces (default: 400)")
    parser.add_argument("--phantom-reverb", type=float, default=0.4,
                        help="Reverb wet intensity relative to dry (0-1, default: 0.4)")
    parser.add_argument("--phantom-lowpass", type=int, default=3800,
                        help="Lowpass cutoff in Hz for sibilance softening (default: 3800)")
    parser.add_argument("--phantom-orig-fade", type=int, default=0,
                        help="跟随翻译：原声淡化时长 ms，原声播完前慢慢变弱（0=关，0-3000）")
    parser.add_argument("--phantom-tts-fade", type=int, default=0,
                        help="跟随翻译：尾声淡化时长 ms，中文播完后慢慢变弱（0=关，0-3000）")
    parser.add_argument("--phantom-virtual-fade", action="store_true",
                        help="跟随翻译：虚拟淡化——尾声淡化不计入空白音频，"
                             "淡化期间直接开始播放下一段（需 --phantom-tts-fade > 0）")
    parser.add_argument("--follow-translation-overhang", type=int, default=3000,
                        help="跟随翻译：允许翻译超出原声窗口的保底空白时长 ms（0-3000，"
                             "默认 3000；超出将截断）")
    parser.add_argument("--no-tts-check", action="store_true",
                        help="Disable the post-TTS artifact check (male-voice/"
                             "mechanical detection + regenerate-once)")
    parser.add_argument("--follow-original-rate", type=float, default=1.15,
                        help="Max speed-up applied to fit TTS into the original in "
                             "'follow_original' mode (1.0-1.25, default: 1.15). When the "
                             "TTS needs more than this it is truncated to the original.")
    parser.add_argument("--llm-kwargs", type=str, default=None,
                        help='LLM subtitle-enhancement settings as JSON, e.g. '
                             '{"provider":"openai","base_url":"https://api.openai.com/v1",'
                             '"api_key":"...","model":"gpt-4o-mini"}. When provided, the '
                             'parsed subtitles are optimized before TTS.')
    parser.add_argument("--asr-backend", type=str, default="fish-asr",
                        choices=["fish-asr", "qwen-audio-3.1-asr-flash", "fun-asr", "mimo-asr", "chickenrice", "common-asr", "moss-transcribe-diarize"],
                        help="ASR engine used to generate subtitles when no VTT "
                             "file is provided (default: fish-asr)")
    parser.add_argument("--asr-kwargs", type=str, default=None,
                        help='ASR speech-recognition settings as JSON, e.g. '
                             '{"api_key":"...","language":"ja","ignore_timestamps":false}. '
                             'Only used when no subtitle file is provided.')
    parser.add_argument("--asr-max-concurrent", type=int, default=3,
                        help="Max concurrent ASR chunk/transcribe workers. Keep "
                             "low — Fish Audio ASR rate-limits concurrent requests "
                             "with HTTP 429 (default: 3)")
    parser.add_argument("--japanese-cleaner", type=str, default="mecab-regex",
                        help="Japanese ASR cleanup implementation registered in "
                             "modules.japanese_cleaning (default: mecab-regex)")
    parser.add_argument("--detect-stage", "--vad-sentence-realignment-beta",
                        dest="detect_stage", action="store_true",
                        help="Enable Japanese sentence detection after ASR and before cleanup")
    parser.add_argument("--sentence-aligner", type=str, default="joint-vad",
                        help="Detect-stage sentence aligner implementation (default: joint-vad)")
    parser.add_argument("--qwen-align-endpoint", default="http://127.0.0.1:8007")
    parser.add_argument("--vad-backend", choices=["fsmn-vad", "asmr-vad"], default="fsmn-vad")
    parser.add_argument("--vad-endpoint", default="http://127.0.0.1:8006")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--sentence-detector", type=str, default="mecab-japanese",
                        help="Detect-stage Japanese sentence detector (default: mecab-japanese)")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable VAD slicing (vad.enabled=false): the "
                             "no-subtitle path aborts instead of generating slices")
    parser.add_argument("--vad-max-end-silence-time", type=int, default=1500,
                        help="FSMN-VAD pause (ms) that splits a sentence; "
                             "passed per generate() call (default: 1500)")
    parser.add_argument("--vad-max-single-segment-time", type=int, default=25000,
                        help="FSMN-VAD max voiced segment length (ms); read at "
                             "AutoModel build time (default: 25000)")
    parser.add_argument("--vad-min-sec", type=float, default=0.3,
                        help="Fragments shorter than this (s) may be merged (default: 0.3)")
    parser.add_argument("--vad-target-min-sec", type=float, default=5.0,
                        help="Merge fragments until the slice reaches this length (s) (default: 5.0)")
    parser.add_argument("--vad-target-max-sec", type=float, default=25.0,
                        help="Do not merge past this length (s) (default: 25.0)")
    parser.add_argument("--vad-max-sec", type=float, default=25.0,
                        help="Slices longer than this (s) get re-split (default: 25.0)")
    parser.add_argument("--vad-max-merge-gap-sec", type=float, default=2.0,
                        help="Fragments closer than this (s) may be merged (default: 2.0)")
    parser.add_argument("--vad-split-window-sec", type=float, default=20.0,
                        help="Re-split window length (s) for over-long slices (default: 20.0)")
    parser.add_argument("--vad-split-overlap-sec", type=float, default=4.0,
                        help="Overlap (s) between adjacent re-split windows (default: 4.0)")

    args = parser.parse_args()

    if args.arrangement_editor or args.arrangement_run:
        from contextlib import redirect_stdout
        from modules.arrangement import run_editor_request
        from modules.pipeline import run_arrangement_pipeline
        try:
            payload = json.load(sys.stdin)
            if args.arrangement_run:
                _emit_result(run_arrangement_pipeline(payload))
            else:
                with redirect_stdout(sys.stderr):
                    result = run_editor_request(payload)
                print(json.dumps({"success": True, **result}, ensure_ascii=False), flush=True)
        except Exception as error:
            if args.arrangement_run:
                _emit_error(unexpected_error(error))
                sys.exit(1)
            print(json.dumps({"success": False, "message": str(error),
                              "error": unexpected_error(error).to_payload()}, ensure_ascii=False), flush=True)
        return

    if args.export_high_quality:
        from modules.high_quality_export import render_high_quality, HighQualityExportError
        if not args.output:
            parser.error("高音质导出需要指定 --output 文件路径")
        try:
            render_high_quality(args.export_high_quality, args.output)
        except (HighQualityExportError, OSError, ValueError, KeyError) as error:
            print(f"EXPORT_HIGH_QUALITY_FAILED: {error}", file=sys.stderr, flush=True)
            raise SystemExit(1) from error
        return

    if args.tts_backend_kwargs_stdin:
        if args.tts_backend != "common-tts" or args.tts_backend_kwargs:
            parser.error("--tts-backend-kwargs-stdin 仅用于 common-tts，且不能同时使用 --tts-backend-kwargs")
        from modules.tts.plugin_manifest import read_runtime_configuration
        from modules.error import TTSError
        try:
            args.tts_backend_kwargs = json.dumps(read_runtime_configuration(sys.stdin.buffer), ensure_ascii=False)
        except TTSError as error:
            parser.error(str(error))

    # Desktop preflight protocol: configuration is received over stdin so API
    # keys never appear in the process command line.  Always return one JSON
    # object; service-level failures are data, not process crashes.
    if args.preflight:
        from contextlib import redirect_stdout
        from modules.preflight import run_preflight_payload

        try:
            payload = json.load(sys.stdin)
            if not isinstance(payload, dict):
                raise ValueError("预检请求必须是 JSON 对象")
            with redirect_stdout(sys.stderr):
                result = run_preflight_payload(payload)
        except Exception as error:
            result = {"success": False, "message": f"预检请求无效：{error}"}
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
        sys.exit(0)

    # Build/runtime health check: load the packaged ctypes DLL rather than only
    # checking that a file with the expected name exists.
    if args.check_vad_runtime:
        from modules.vad.native import NativeVAD, _GGUF_PATH, _find_dll

        dll_path = _find_dll()
        diagnostic = {
            "available": False,
            "dll": dll_path,
            "gguf": _GGUF_PATH,
        }
        try:
            NativeVAD(dll_path)
            if not os.path.isfile(_GGUF_PATH):
                raise FileNotFoundError(f"fsmn-vad.gguf not found: {_GGUF_PATH}")
            diagnostic["available"] = True
        except Exception as e:
            diagnostic["error"] = str(e)
        print(json.dumps(diagnostic, ensure_ascii=False))
        sys.exit(0 if diagnostic["available"] else 1)

    # Build/runtime health check: initialize the packaged MeCab extension,
    # open the bundled UniDic files, and perform one real tokenization.
    if args.check_japanese_runtime:
        from modules.japanese_cleaning import MecabTokenizer

        diagnostic = {"available": False, "tokenizer": "mecab-unidic-lite"}
        try:
            tokens = MecabTokenizer().tokenize("今日は")
            if not tokens:
                raise RuntimeError("MeCab 未返回任何分词结果")
            diagnostic["available"] = True
            diagnostic["tokens"] = [token.surface for token in tokens]
        except Exception as error:
            diagnostic["error"] = str(error)
        print(json.dumps(diagnostic, ensure_ascii=False))
        sys.exit(0 if diagnostic["available"] else 1)

    # Voice catalog mode: print JSON and exit (used by the Electron voice picker).
    # Handled before input validation so it works without positional args.
    if args.list_tts_voices:
        from modules.tts import create_tts
        adapter = create_tts(args.tts_backend, **(json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}))
        voices = [vars(v) for v in adapter.get_available_voices()]
        print(json.dumps(voices, ensure_ascii=False))
        sys.exit(0)

    # MiniMax voice-design mode: generate a custom voice from a text description,
    # save the trial audio, print {voice_id, trial_audio} and exit.
    if args.design_voice:
        import asyncio
        from modules.tts import create_tts
        if not args.design_prompt or not args.design_preview_text:
            print("Error: --design-voice 需要 --design-prompt 和 --design-preview-text",
                  file=sys.stderr)
            sys.exit(1)
        if args.output and os.path.isdir(args.output):
            print(f"Error: --output 指向了目录，应为文件路径: {args.output}",
                  file=sys.stderr)
            sys.exit(1)
        try:
            kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
            adapter = create_tts("minimax-tts", **kwargs)
        except Exception as e:
            print(f"Error: 无效的 --tts-backend-kwargs: {e}", file=sys.stderr)
            sys.exit(1)
        if args.output:
            trial = os.path.abspath(args.output)
        else:
            trial = os.path.join(os.getcwd(), "minimax_trial.mp3")
        try:
            voice_id = asyncio.run(adapter.design_voice(
                args.design_prompt, args.design_preview_text, trial,
                voice_id=args.design_voice_id or None))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"voice_id": voice_id, "trial_audio": trial},
                         ensure_ascii=False))
        sys.exit(0)

    # MiniMax voice-clone mode: upload the sample audio, clone the voice,
    # download the demo audio, print {voice_id, demo_audio} and exit.
    if args.clone_voice:
        import asyncio
        from modules.tts import create_tts
        if not args.clone_file or not os.path.exists(args.clone_file):
            print("Error: --clone-voice 需要有效的 --clone-file 音频路径")
            sys.exit(1)
        kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
        adapter = create_tts("minimax-tts", **kwargs)
        if args.output:
            demo = os.path.abspath(args.output)
        else:
            demo = os.path.join(os.getcwd(), "minimax_clone_demo.mp3")
        try:
            async def _run():
                if bool(args.clone_prompt_file) != bool(args.clone_prompt_text):
                    raise ValueError("增强示例音频和对应文本必须同时提供")
                file_id = await adapter.upload_voice_file(
                    os.path.abspath(args.clone_file), purpose="voice_clone")
                prompt_audio_id = None
                if args.clone_prompt_file:
                    if not os.path.exists(args.clone_prompt_file):
                        raise ValueError("增强示例音频不存在")
                    prompt_audio_id = await adapter.upload_voice_file(
                        os.path.abspath(args.clone_prompt_file),
                        purpose="prompt_audio")
                voice_id = await adapter.clone_voice(
                    file_id, voice_id=args.clone_voice_id or None,
                    text=args.clone_text or None,
                    model=kwargs.get("model"),
                    prompt_audio_id=prompt_audio_id,
                    prompt_text=args.clone_prompt_text or None,
                    text_validation=args.clone_text_validation or None,
                    accuracy=args.clone_accuracy,
                    need_noise_reduction=args.clone_noise_reduction,
                    need_volume_normalization=args.clone_volume_normalization,
                    output_path=demo if args.clone_text else None)
                return voice_id
            voice_id = asyncio.run(_run())
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(json.dumps({"voice_id": voice_id,
                          "demo_audio": demo if args.clone_text else ""},
                         ensure_ascii=False))
        sys.exit(0)

    # MiniMax voice-management modes: list the user's own voices / delete one.
    if args.list_voices:
        import asyncio
        from modules.tts import create_tts
        kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
        adapter = create_tts("minimax-tts", **kwargs)
        try:
            voices = asyncio.run(adapter.get_voices())
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(json.dumps({"voices": voices}, ensure_ascii=False))
        sys.exit(0)

    if args.delete_voice:
        import asyncio
        from modules.tts import create_tts
        vtype = getattr(args, "delete_voice_type", "") or "voice_cloning"
        kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
        adapter = create_tts("minimax-tts", **kwargs)
        try:
            asyncio.run(adapter.delete_voice(args.delete_voice, vtype))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(json.dumps({"ok": True}, ensure_ascii=False))
        sys.exit(0)

    if args.list_fish_voices or args.delete_fish_voice:
        import asyncio
        from modules.tts import create_tts
        try:
            kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
            adapter = create_tts("fish-audio", **kwargs)
            if args.delete_fish_voice:
                asyncio.run(adapter.delete_voice(args.delete_fish_voice))
                result = {"ok": True}
            else:
                result = {"voices": asyncio.run(adapter.get_voices())}
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)

    if args.design_fish_voice or args.clone_fish_voice:
        import asyncio
        from modules.tts import create_tts
        try:
            kwargs = json.loads(args.tts_backend_kwargs) if args.tts_backend_kwargs else {}
            adapter = create_tts("fish-audio", **kwargs)
            if not args.fish_voice_title.strip():
                raise ValueError("请输入音色名称")
            if args.design_fish_voice:
                if not args.fish_design_instruction.strip():
                    raise ValueError("请输入音色描述")
                preview = os.path.abspath(args.output or "fish_voice_design.wav")

                async def _design_and_create():
                    await adapter.design_voice(
                        args.fish_design_instruction,
                        args.fish_preview_text,
                        preview,
                        language=args.fish_language,
                    )
                    model = await adapter.create_voice(
                        args.fish_voice_title,
                        [preview],
                        description=args.fish_voice_description or args.fish_design_instruction,
                        texts=[args.fish_preview_text] if args.fish_preview_text else None,
                    )
                    return model

                model = asyncio.run(_design_and_create())
                result = {**model, "trial_audio": preview}
            else:
                if not args.fish_clone_file or not os.path.isfile(args.fish_clone_file):
                    raise ValueError("请选择有效的参考音频")
                model = asyncio.run(adapter.create_voice(
                    args.fish_voice_title,
                    [os.path.abspath(args.fish_clone_file)],
                    description=args.fish_voice_description,
                    texts=[args.fish_preview_text] if args.fish_preview_text else None,
                ))
                result = model
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)

    # Validate inputs. The subtitle (vtt) is optional — without it the ASR engine
    # generates subtitles from the audio (requires --asr-kwargs).
    if not args.audio or not os.path.exists(args.audio):
        raise ConfigurationError(descriptor("CFG_AUDIO_MISSING"), detail=args.audio or "")
    if args.vtt and not os.path.exists(args.vtt):
        raise ConfigurationError(descriptor("CFG_SUBTITLE_MISSING"), detail=args.vtt)
    if not args.vtt and not args.asr_kwargs:
        raise ConfigurationError(descriptor("CFG_ASR_KEY_MISSING"))

    # Resolve absolute paths
    args.audio = os.path.abspath(args.audio)
    if args.vtt:
        args.vtt = os.path.abspath(args.vtt)

    # Default output directory
    if args.output is None:
        stem = Path(args.audio).stem
        args.output = os.path.join(os.path.dirname(args.audio) or ".", f"output_{stem}")

    # Handle per-speaker rate (simplified: just use the base rate for now)
    tts_rate = args.rate
    if args.speaker_rate:
        print(f"Note: Per-speaker rates ({args.speaker_rate}) parsed but using base rate: {tts_rate}")
        print("      Full per-speaker rate support coming in Phase 2.")

    print("ASMRTranslator")
    print(f"  Audio: {args.audio}")
    print(f"  VTT:   {args.vtt if args.vtt else '（未提供，将用 ASR 生成）'}")
    print(f"  Output:{args.output}")
    print(f"  TTS rate: {tts_rate}")
    print()

    # Parse backend kwargs from JSON string
    tts_backend_kwargs = None
    if args.tts_backend_kwargs and args.tts_backend != "subtitles-only":
        try:
            tts_backend_kwargs = json.loads(args.tts_backend_kwargs)
        except json.JSONDecodeError as e:
            raise ConfigurationError(descriptor("CFG_TTS_INCOMPLETE"), detail=str(e)) from e

    # Parse LLM subtitle-enhancement kwargs from JSON string
    llm_kwargs = None
    if args.llm_kwargs:
        try:
            llm_kwargs = json.loads(args.llm_kwargs)
        except json.JSONDecodeError as e:
            raise ConfigurationError(descriptor("CFG_LLM_INCOMPLETE"), detail=str(e)) from e

    # Parse ASR kwargs from JSON string
    asr_kwargs = None
    if args.asr_kwargs:
        try:
            asr_kwargs = json.loads(args.asr_kwargs)
        except json.JSONDecodeError as e:
            raise ConfigurationError(descriptor("CFG_ASR_KEY_MISSING"), detail=str(e)) from e

    result = run_pipeline(
        audio_path=args.audio,
        vtt_path=args.vtt,
        output_dir=args.output,
        tts_rate=tts_rate,
        tts_pitch=args.tts_pitch,
        crossfade_ms=args.crossfade,
        retain_original=args.retain_original or args.phantom_mode,
        sync_mode=args.sync_mode,
        tts_volume=args.tts_volume,
        tts_backend=args.tts_backend,
        tts_backend_kwargs=tts_backend_kwargs,
        tts_max_concurrent=args.tts_max_concurrent,
        num_workers=args.num_workers,
        split_workers=args.split_workers,
        build_workers=args.build_workers,
        phantom_mode=args.phantom_mode,
        phantom_delay_ms=args.phantom_delay,
        phantom_reverb_mix=args.phantom_reverb,
        phantom_lowpass_hz=args.phantom_lowpass,
        phantom_orig_fade_ms=max(0, min(3000, args.phantom_orig_fade)),
        phantom_tts_fade_ms=max(0, min(3000, args.phantom_tts_fade)),
        phantom_virtual_fade=args.phantom_virtual_fade,
        follow_translation_overhang_ms=max(0, min(3000, args.follow_translation_overhang)),
        tts_check_artifact=not args.no_tts_check,
        follow_original_rate=max(1.0, min(1.25, args.follow_original_rate)),
        llm_kwargs=llm_kwargs,
        asr_kwargs=asr_kwargs,
        asr_backend=args.asr_backend,
        asr_max_concurrent=args.asr_max_concurrent,
        japanese_cleaner=args.japanese_cleaner,
        vad_enabled=not args.no_vad,
        detect_enabled=args.detect_stage,
        sentence_aligner=args.sentence_aligner,
        qwen_align_endpoint=args.qwen_align_endpoint,
        vad_backend=args.vad_backend,
        vad_endpoint=args.vad_endpoint,
        vad_threshold=args.vad_threshold,
        sentence_detector=args.sentence_detector,
        vad_max_end_silence_time=args.vad_max_end_silence_time,
        vad_max_single_segment_time=args.vad_max_single_segment_time,
        vad_min_sec=args.vad_min_sec,
        vad_target_min_sec=args.vad_target_min_sec,
        vad_target_max_sec=args.vad_target_max_sec,
        vad_max_sec=args.vad_max_sec,
        vad_max_merge_gap_sec=args.vad_max_merge_gap_sec,
        vad_split_window_sec=args.vad_split_window_sec,
        vad_split_overlap_sec=args.vad_split_overlap_sec,
    )

    print(f"\nDone! Output saved to: {result.get('output_mp3') or result['final_vtt']}")
    _emit_result(result)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as error:
        import traceback
        _emit_error(error)
        report_error_async(error, traceback.format_exc())
        sys.exit(1)
    except Exception as error:
        import traceback
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, end="")
        pipeline_error = unexpected_error(error)
        _emit_error(pipeline_error)
        report_error_async(pipeline_error, traceback_text)
        sys.exit(1)
