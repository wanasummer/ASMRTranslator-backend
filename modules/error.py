"""Stable, user-safe error contract shared by all pipeline stages."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Mapping

__all__ = ["ErrorDescriptor", "PipelineError", "ConfigurationError", "ProjectError",
    "VadError", "AsrError", "LlmError", "TtsError", "AudioProcessingError",
    "ExportError", "ASRError", "TTSError", "_RetryableTTSError", "LLMError",
    "VADUnavailableError", "descriptor", "safe_detail", "unexpected_error",
    "is_memory_allocation_error"]

@dataclass(frozen=True)
class ErrorDescriptor:
    code: str
    stage: str
    user_message: str
    action: str
    retryable: bool

_ERRORS = {
    "CFG_AUDIO_MISSING": ("prepare", "源音频不存在，请重新选择文件。", "locate_audio", False),
    "CFG_SUBTITLE_MISSING": ("prepare", "字幕文件不存在，请重新选择或改用自动生成。", "locate_subtitle", False),
    "CFG_SUBTITLE_UNSUPPORTED": ("prepare", "暂不支持此字幕格式，请选择 VTT、SRT 或 LRC。", "locate_subtitle", False),
    "CFG_SUBTITLE_ENCODING": ("prepare", "无法识别或解码字幕编码，请指定编码或将字幕转存为 UTF-8。", "locate_subtitle", False),
    "CFG_OUTPUT_INVALID": ("prepare", "无法使用当前项目输出目录，请检查目录权限。", "open_project", False),
    "CFG_ASR_KEY_MISSING": ("prepare", "未配置 ASR API Key，无法自动生成字幕。", "open_settings", False),
    "CFG_ASR_ENDPOINT_INVALID": ("prepare", "ASR 服务地址或端口无效，请检查设置。", "open_settings", False),
    "CFG_LLM_INCOMPLETE": ("prepare", "大模型翻译配置不完整，请填写 Provider、模型和 API Key。", "open_settings", False),
    "CFG_TTS_INCOMPLETE": ("prepare", "当前 TTS 引擎配置不完整，请检查音色、服务或 API Key。", "open_studio", False),
    "PROJECT_MANIFEST_INVALID": ("project", "项目记录已损坏，无法安全读取。", "open_project", False),
    "PROJECT_SAVE_FAILED": ("project", "无法保存项目记录，请检查目录权限或磁盘空间。", "open_project", True),
    "SCAN_ROOT_UNREADABLE": ("scan", "无法读取所选项目文件夹。", "choose_project", False),
    "VAD_RUNTIME_UNAVAILABLE": ("vad", "人声检测组件不可用，请重新安装或联系管理员。", "contact_admin", False),
    "VAD_DECODE_FAILED": ("vad", "音频无法解码，无法执行人声检测。", "locate_audio", False),
    "VAD_INFERENCE_FAILED": ("vad", "人声检测运行失败，请重试；持续出现请联系管理员。", "retry", True),
    "VAD_NO_SPEECH": ("vad", "未在音频中检测到可识别的人声。", "locate_audio", False),
    "ASR_AUTH_INVALID": ("asr", "ASR API Key 无效，请检查设置。", "open_settings", False),
    "ASR_CREDIT_EXHAUSTED": ("asr", "ASR 额度不足，请充值或更换服务。", "open_settings", False),
    "ASR_RATE_LIMITED": ("asr", "ASR 请求过于频繁，重试仍未成功，请稍后再试。", "retry_later", True),
    "ASR_UNREACHABLE": ("asr", "无法连接 ASR 服务，请检查网络或服务状态。", "open_settings", True),
    "ASR_TIMEOUT": ("asr", "ASR 识别超时，请检查服务状态后重试。", "retry", True),
    "ASR_RESPONSE_INVALID": ("asr", "ASR 返回了无法识别的数据，请联系服务提供方。", "view_log", False),
    "ASR_TIMESTAMPS_MISSING": ("asr", "ASR 未返回必要时间戳，无法生成字幕。", "open_settings", False),
    "ASR_SEGMENT_FAILED": ("asr", "第 {segment} 个识别片段失败，任务已停止。", "retry", True),
    "ASR_CONTENT_REJECTED": ("asr", "语音识别服务因内容策略拒绝该片段，请查看日志。", "view_log", False),
    "TTS_CONTENT_REJECTED": ("tts", "配音服务因内容策略拒绝该片段，请查看日志。", "view_log", False),
    "LLM_AUTH_INVALID": ("llm", "文字大模型 API Key 无效，请检查设置。", "open_settings", False),
    "LLM_CONTENT_REJECTED": ("llm", "大模型服务因内容策略拒绝请求，具体原因请查看日志。", "view_log", False),
    "LLM_ACCESS_DENIED": ("llm", "大模型服务拒绝访问（HTTP 403），请检查服务权限；具体原因请查看日志。", "view_log", False),
    "LLM_MODEL_UNAVAILABLE": ("llm", "当前大模型不可用或模型名错误，请检查设置。", "open_settings", False),
    "LLM_CREDIT_EXHAUSTED": ("llm", "大模型额度不足，请充值或更换模型。", "open_settings", False),
    "LLM_RATE_LIMITED": ("llm", "大模型请求过于频繁，重试仍未成功，请稍后再试。", "retry_later", True),
    "LLM_GEMINI_RATE_LIMITED": ("llm", "Gemini 请求达到限额，自动重试仍未成功，请稍后手动重试。", "retry_later", True),
    "LLM_SERVICE_BUSY": ("llm", "Gemini 服务暂时繁忙，自动重试仍未成功，请稍后手动重试。", "retry_later", True),
    "LLM_UNREACHABLE": ("llm", "无法连接文字大模型服务，请检查网络和服务地址。", "open_settings", True),
    "LLM_TIMEOUT": ("llm", "大模型处理超时，请稍后重试。", "retry", True),
    "LLM_RESPONSE_INVALID": ("llm", "大模型返回格式不正确，无法安全继续。", "view_log", True),
    "LLM_SUMMARY_FAILED": ("llm.summary", "剧情分析失败，任务已停止。", "retry", True),
    "LLM_TRANSLATION_INCOMPLETE": ("llm.translate", "部分字幕未能完成翻译，任务已停止以避免缺句。", "retry", True),
    "LLM_OPTIMIZE_INCOMPLETE": ("llm.optimize", "部分字幕未能完成优化，任务已停止。", "retry", True),
    "TTS_AUTH_INVALID": ("tts", "TTS API Key 无效，请检查当前引擎设置。", "open_studio", False),
    "TTS_ACCESS_DENIED": ("tts", "Edge TTS 服务拒绝访问，请稍后重试；持续出现请查看日志或更换引擎。", "view_log", True),
    "TTS_VOICE_INVALID": ("tts", "当前音色不可用，请重新选择音色。", "open_studio", False),
    "TTS_CREDIT_EXHAUSTED": ("tts", "TTS 额度不足，请充值或更换引擎。", "open_studio", False),
    "TTS_RATE_LIMITED": ("tts", "TTS 请求过于频繁，重试仍未成功，请稍后再试。", "retry_later", True),
    "TTS_UNREACHABLE": ("tts", "无法连接 TTS 服务，请检查网络或本地插件。", "open_studio", True),
    "TTS_OUT_OF_MEMORY": ("tts", "TTS 服务内存分配失败，无法完成模型加载或配音生成。请关闭未使用的模型服务和高内存程序，释放内存后重试。", "view_log", True),
    "TTS_MODEL_LOAD_FAILED": ("tts", "TTS 音色模型加载失败，请查看插件日志，检查模型文件和可用内存后重试。", "view_log", True),
    "TTS_SERVICE_ERROR": ("tts", "TTS 服务处理请求失败，请查看插件日志，检查模型加载和推理状态后重试。", "view_log", True),
    "TTS_SEGMENT_FAILED": ("tts", "第 {segment} 段配音生成失败，任务已停止。", "retry", True),
    "TTS_OUTPUT_INVALID": ("tts", "TTS 返回了空音频或损坏音频，任务已停止。", "retry", True),
    "AUDIO_FFMPEG_MISSING": ("audio", "未找到 FFmpeg，无法处理音频，请重新安装。", "contact_admin", False),
    "AUDIO_DECODE_FAILED": ("audio", "无法读取该音频文件，请确认文件未损坏。", "locate_audio", False),
    "AUDIO_SPLIT_FAILED": ("audio.split", "音频切分失败，任务已停止。", "view_log", True),
    "AUDIO_DSP_FAILED": ("audio.build", "声音处理失败，任务已停止。", "view_log", True),
    "AUDIO_JOIN_FAILED": ("audio.concat", "最终音频合并失败，任务已停止。", "view_log", True),
    "AUDIO_METADATA_FAILED": ("audio.metadata", "原音频标签或封面复制失败，请查看日志后重试。", "view_log", True),
    "SUBTITLE_FINAL_EMPTY": ("subtitle.export", "最终没有可导出的中文字幕。", "view_log", False),
    "OUTPUT_WRITE_DENIED": ("output", "无法写入项目目录，请检查权限。", "open_project", False),
    "OUTPUT_DISK_FULL": ("output", "磁盘空间不足，无法完成生成。", "open_project", True),
    "QUEUE_TASK_DUPLICATE": ("queue", "该音频已在等待或转换中。", "open_tasks", False),
    "QUEUE_PROCESS_START_FAILED": ("queue", "无法启动转换引擎，请重新安装或联系管理员。", "contact_admin", True),
    "EXPORT_SOURCE_STALE": ("export", "源音频已变化，请重新转换后再导出。", "open_project", False),
    "EXPORT_ARTIFACT_MISSING": ("export", "转换产物不完整，请重新转换该音频。", "open_project", False),
    "EXPORT_WRITE_FAILED": ("export", "无法写入 artifact 文件夹，请检查权限和磁盘空间。", "open_project", True),
    "PROGRAM_RESULT_MISSING": ("program", "转换引擎未返回完整结果，请联系管理员。", "contact_admin", False),
    "PROGRAM_PROTOCOL_INVALID": ("program", "转换引擎通信异常，请联系管理员。", "contact_admin", False),
    "PROGRAM_UNEXPECTED": ("program", "程序发生异常，请联系管理员。", "contact_admin", False),
    "USER_CANCELLED": ("queue", "任务已取消。", "none", False),
}

def descriptor(code: str, **values: Any) -> ErrorDescriptor:
    stage, message, action, retryable = _ERRORS[code]
    try:
        message = message.format(**values)
    except KeyError:
        pass
    return ErrorDescriptor(code, stage, message, action, retryable)

_SECRET_RE = re.compile(r"(?i)(authorization|api[-_ ]?key|token|secret|password)([\"']?\s*[:=]\s*[\"']?)([^\s,;\"'}]+)")
_BEARER_RE = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")
_QUERY_RE = re.compile(r"(?i)([?&](?:key|api_key|token|secret|ocp-apim-subscription-key|trustedclienttoken|sec-ms-gec)=)[^&#\s]+")

def safe_detail(value: Any, limit: int = 500) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _SECRET_RE.sub(r"\1\2[REDACTED]", text)
    return _QUERY_RE.sub(r"\1[REDACTED]", text)[:limit]


def is_memory_allocation_error(value: str) -> bool:
    lower = value.lower()
    return any(term in lower for term in (
        "bad_alloc", "bad allocation", "out of memory", "out_of_memory",
        "memoryerror", "内存不足", "内存分配失败"))

class PipelineError(Exception):
    def __init__(self, error: ErrorDescriptor, *, detail: str = "", cause: Exception | None = None,
                 context: Mapping[str, Any] | None = None):
        super().__init__(error.user_message)
        self.descriptor, self.detail, self.cause = error, safe_detail(detail), cause
        self.log_path = ""
        allowed = {"segment", "status", "backend", "path", "operation"}
        self.context = {k: v for k, v in dict(context or {}).items() if k in allowed}

    def to_payload(self, *, log_path: str = "") -> dict[str, Any]:
        value = {"contractVersion": 1, "code": self.descriptor.code,
            "stage": self.descriptor.stage, "message": self.descriptor.user_message,
            "action": self.descriptor.action, "retryable": self.descriptor.retryable}
        if self.detail: value["detail"] = self.detail
        if self.context: value["context"] = self.context
        if log_path: value["logPath"] = log_path
        return value

class ConfigurationError(PipelineError): pass
class ProjectError(PipelineError): pass
class VadError(PipelineError): pass
class AsrError(PipelineError): pass
class LlmError(PipelineError): pass
class TtsError(PipelineError): pass
class AudioProcessingError(PipelineError): pass
class ExportError(PipelineError): pass

class _LegacyStageError(PipelineError):
    default_code = "PROGRAM_UNEXPECTED"
    def __init__(self, value: ErrorDescriptor | str, **kwargs: Any):
        code = self.default_code
        if isinstance(value, str):
            lower = value.lower()
            family = code.split("_", 1)[0]
            if family == "TTS" and is_memory_allocation_error(value):
                code = "TTS_OUT_OF_MEMORY"
            elif family in ("ASR", "TTS", "LLM") and any(term in lower for term in ("content_policy_violation", "content_filter", "内容策略")):
                code = f"{family}_CONTENT_REJECTED"
            elif "401" in lower or "key invalid" in lower or "invalid api key" in lower or "api key 无效" in lower:
                code = f"{family}_AUTH_INVALID"
            elif "403" in lower:
                code = "LLM_ACCESS_DENIED" if family == "LLM" else f"{family}_AUTH_INVALID"
            elif "402" in lower or "credit" in lower or "额度" in lower or "余额" in lower:
                code = f"{family}_CREDIT_EXHAUSTED"
            elif "429" in lower or "rate limit" in lower:
                code = f"{family}_RATE_LIMITED"
            elif re.search(r"\b5\d\d\b", lower):
                candidate = "TTS_SERVICE_ERROR" if family == "TTS" else f"{family}_UNREACHABLE"
                code = candidate if candidate in _ERRORS else code
            elif "timeout" in lower or "超时" in lower:
                code = f"{family}_TIMEOUT" if f"{family}_TIMEOUT" in _ERRORS else code
            elif "404" in lower and family == "LLM":
                code = "LLM_MODEL_UNAVAILABLE"
            elif any(term in lower for term in ("connect", "dns", "tls", "connection reset", "unreachable", "无法连接")):
                code = f"{family}_UNREACHABLE"
        super().__init__(value if isinstance(value, ErrorDescriptor) else descriptor(code),
                         detail=kwargs.pop("detail", str(value) if isinstance(value, str) else ""), **kwargs)

    def __str__(self) -> str:
        return self.detail or super().__str__()

class ASRError(_LegacyStageError): default_code = "ASR_SEGMENT_FAILED"
class TTSError(_LegacyStageError): default_code = "TTS_SEGMENT_FAILED"
class _RetryableTTSError(TTSError): pass
class LLMError(_LegacyStageError): default_code = "LLM_RESPONSE_INVALID"
class VADUnavailableError(_LegacyStageError, RuntimeError): default_code = "VAD_RUNTIME_UNAVAILABLE"

def unexpected_error(error: Exception) -> PipelineError:
    return error if isinstance(error, PipelineError) else PipelineError(
        descriptor("PROGRAM_UNEXPECTED"), detail=str(error), cause=error)
