"""MiniMax TTS adapter — T2A v2 synchronous synthesis (Hex/JSON).

API: POST https://api.minimaxi.com/v1/t2a_v2  (fallback api-bj.minimaxi.com)
Auth: Bearer token in Authorization header (MINIMAX_API_KEY env or api_key=).

Models: speech-02-hd (default) and speech-02-turbo are exposed by the Tauri UI.
        The adapter retains the 2.8/2.6 identifiers for legacy backend callers.
        Expressiveness on 2.8 is driven by
        voice-effect tags ((laughs) (sighs) ...). The emotion constructor
        argument is retained for API compatibility but intentionally omitted
        from synthesis requests because it produced poor results in this app.

Non-streaming responses carry the audio as a HEX string under
``data.audio`` (status 2 = finished). Errors surface in
``base_resp.status_code`` (0 = ok, 1004 = auth, 1039 = TPM rate limit,
2013 = bad params). ``data`` itself may be null and must be checked.

Voices come from MiniMax's system voice catalog (see the docs
`/faq/system-voice-id`). Speaker→voice mapping follows the project's
voice_map convention; a ``default_voice`` override (via tts_engine) wins.
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..audio_splitter import get_audio_duration
from ..error import TTSError, _RetryableTTSError
from ..io import open_write
from .base import TTSAdapter
from .models import VoiceInfo

# ---- Global request throttle ----------------------------------------------
# MiniMax enforces a per-minute request cap (error 1002 "rate limit exceeded
# (RPM)"). The pipeline synthesizes many segments concurrently (8 parallel
# chunks × up to 8 workers), so a naive burst blows past the cap and every
# segment fails. This module-level gate enforces a minimum interval between
# requests ACROSS all threads/event loops (threading lock, not asyncio), so
# the aggregate rate stays under the cap regardless of concurrency.
_throttle_lock = threading.Lock()
_last_request_ts = 0.0


def _global_throttle(min_interval_sec: float) -> None:
    """Block (async-sleep) until ``min_interval_sec`` elapsed since the last
    request anywhere in the process. Call before each API request."""
    global _last_request_ts
    while True:
        with _throttle_lock:
            now = time.monotonic()
            wait = min_interval_sec - (now - _last_request_ts)
            if wait <= 0:
                _last_request_ts = now
                return
        time.sleep(wait)  # sync sleep: safe across threads/loops

# MiniMax system voice catalog — 只保留适配 ASMR 的中文（普通话）女声：
# 温柔/甜美/御姐/少女系，排除成熟女性、大婶/奶奶、以及带口音的音色（如港普空姐）。
# 来源：官方系统音色列表（platform.minimaxi.com/docs/faq/system-voice-id）。
_KNOWN_VOICES: dict[str, str] = {
    # 基础系（jingpin 为 beta 精修版）
    "female-shaonv": "少女音色",
    "female-yujie": "御姐音色",
    "female-tianmei": "甜美女性音色",
    "female-shaonv-jingpin": "少女音色-beta",
    "female-yujie-jingpin": "御姐音色-beta",
    "female-tianmei-jingpin": "甜美女性音色-beta",
    # 角色系（ASMR 向）
    "lovely_girl": "萌萌女童",
    "tianxin_xiaoling": "甜心小玲",
    "qiaopi_mengmei": "俏皮萌妹",
    "wumei_yujie": "妩媚御姐",
    "diadia_xuemei": "嗲嗲学妹",
    "danya_xuejie": "淡雅学姐",
    "Arrogant_Miss": "嚣张小姐",
    "Chinese (Mandarin)_Mature_Woman": "傲娇御姐",
    "Chinese (Mandarin)_Warm_Bestie": "温暖闺蜜",
    "Chinese (Mandarin)_Sweet_Lady": "甜美女声",
    "Chinese (Mandarin)_Warm_Girl": "温暖少女",
    "Chinese (Mandarin)_Crisp_Girl": "清脆少女",
    "Chinese (Mandarin)_Soft_Girl": "柔和少女",
    "Chinese (Mandarin)_Gentle_Senior": "温柔学姐",
}

# T2A models exposed by this app for the /v1/t2a_v2 endpoint.
_MODELS = (
    "speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
    "speech-02-hd", "speech-02-turbo",
)

# Status codes that warrant a retry (transient / rate limiting).
_RETRYABLE_CODES = {1001, 1002, 1039}


class MiniMaxTTSAdapter(TTSAdapter):
    """MiniMax T2A v2 synchronous synthesis backend.

    Examples:
        tts = MiniMaxTTSAdapter(
            voice_map={"default": "Chinese (Mandarin)_Warm_Girl",
                       "瑠衣": "Chinese (Mandarin)_Soft_Girl"},
            api_key="...", model="speech-02-hd",
        )
        path = await tts.synthesize("你好世界", "Chinese (Mandarin)_Warm_Girl",
                                    "out.mp3", rate="+0%")
    """

    _API_URL = "https://api.minimaxi.com/v1/t2a_v2"
    _API_URL_BJ = "https://api-bj.minimaxi.com/v1/t2a_v2"
    _DESIGN_URL = "https://api.minimaxi.com/v1/voice_design"
    _DESIGN_URL_BJ = "https://api-bj.minimaxi.com/v1/voice_design"
    _UPLOAD_URL = "https://api.minimaxi.com/v1/files/upload"
    _UPLOAD_URL_BJ = "https://api-bj.minimaxi.com/v1/files/upload"
    _CLONE_URL = "https://api.minimaxi.com/v1/voice_clone"
    _CLONE_URL_BJ = "https://api-bj.minimaxi.com/v1/voice_clone"
    _VOICES_URL = "https://api.minimaxi.com/v1/get_voice"
    _VOICES_URL_BJ = "https://api-bj.minimaxi.com/v1/get_voice"
    _DELETE_VOICE_URL = "https://api.minimaxi.com/v1/delete_voice"
    _DELETE_VOICE_URL_BJ = "https://api-bj.minimaxi.com/v1/delete_voice"

    def __init__(
        self,
        voice_map: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        model: str = "speech-02-hd",
        base_url: Optional[str] = None,
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: int = 0,
        emotion: Optional[str] = None,
        language_boost: Optional[str] = None,
        sample_rate: int = 32000,
        bitrate: int = 128000,
        audio_format: str = "mp3",
        channel: int = 1,
        max_retries: int = 3,
        min_interval_sec: float = 3.0,
    ):
        """Initialize the MiniMax TTS adapter.

        Args:
            voice_map: Speaker→voice_id mapping (falls back to "default").
            api_key: MiniMax API key (defaults to MINIMAX_API_KEY env var).
            model: T2A model id, e.g. "speech-02-hd" (default).
            base_url: Override API base URL (e.g. the Beijing endpoint).
            speed: Voice speed [0.5, 2]; used only when rate="+0%".
            vol: Voice volume (0, 10].
            pitch: Voice pitch [-12, 12] semitones (0 = original).
            emotion: Retained for compatibility but intentionally ignored;
                this app does not send the field to MiniMax.
            language_boost: Optional language hint for pronunciation
                normalization. Defaults to None, so the field is omitted and
                MiniMax decides automatically. Set e.g. "Chinese",
                "Japanese" or "Chinese,Yue" only when explicitly required.
            sample_rate: Output sample rate (8000/16000/22050/24000/32000/44100).
            bitrate: Output bitrate for mp3 (32000/64000/128000/256000).
            audio_format: "mp3", "wav", "flac", "pcm", ...
            channel: 1 or 2.
            max_retries: Max attempts on transient/rate-limit errors.
            min_interval_sec: Minimum interval between API requests (global,
                across all threads/instances). Default 3.0 → ≤20 RPM, matching
                MiniMax's per-minute cap on common accounts (e.g. 20 RPM).
                Raise it if 1002 rate-limit errors still appear.
        """
        super().__init__(voice_map)
        self._network_policy = NetworkPolicy()
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if model not in _MODELS:
            raise ValueError(
                f"Unknown MiniMax T2A model '{model}'. "
                f"Available: {', '.join(_MODELS)}"
            )
        self._model = model
        self._base_url = (base_url or self._API_URL).rstrip("/")
        # 音色设计/文件上传/音色复刻端点与 T2A 同一节点（北京节点若配置则跟随）。
        is_bj = "api-bj" in self._base_url
        self._design_url = self._DESIGN_URL_BJ if is_bj else self._DESIGN_URL
        self._upload_url = self._UPLOAD_URL_BJ if is_bj else self._UPLOAD_URL
        self._clone_url = self._CLONE_URL_BJ if is_bj else self._CLONE_URL
        self._voices_url = self._VOICES_URL_BJ if is_bj else self._VOICES_URL
        self._delete_voice_url = (
            self._DELETE_VOICE_URL_BJ if is_bj else self._DELETE_VOICE_URL)
        self._speed = speed
        self._vol = vol
        self._pitch = pitch
        self._emotion = emotion
        self._language_boost = language_boost
        self._sample_rate = sample_rate
        self._bitrate = bitrate
        self._audio_format = audio_format
        self._channel = channel
        self._max_retries = max_retries
        self._min_interval_sec = max(0.0, float(min_interval_sec))

    @property
    def name(self) -> str:
        return "minimax-tts"

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synthesize speech via the MiniMax T2A v2 API.

        Args:
            text: Text to speak (Chinese/Japanese/English; supports MiniMax
                inline tags like ``<#0.5#>`` pauses and ``(laughs)`` on 2.8).
            voice: MiniMax voice_id.
            output_path: Where to save the audio file.
            rate: Speed adjustment "+10%" / "-5%"; mapped onto MiniMax's
                ``voice_setting.speed`` [0.5, 2] (clamped).
            pitch: Not supported by this backend (MiniMax pitch is fixed via
                the constructor) — accepted for API parity, ignored.

        Returns:
            Path to the generated audio file.
        """
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )

        last_error = None
        for attempt in range(self._max_retries):
            try:
                audio_bytes = await self._call_api(text, voice, rate)
                return self._save_audio(audio_bytes, output_path)
            except _RetryableTTSError as e:
                # Transient (rate limit / timeout): retry with backoff.
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    detail = str(e)[:200]
                    print(
                        f"       [retry {attempt + 1}] MiniMax '{text[:20]}...' "
                        f"-> wait {wait}s ({type(e).__name__}: {detail})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                # Network-level failure (connection reset, timeout, ...).
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    detail = str(e)[:200]
                    print(
                        f"       [retry {attempt + 1}] MiniMax '{text[:20]}...' "
                        f"-> wait {wait}s ({type(e).__name__}: {detail})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            # Anything else (auth, bad params, malformed response) is permanent:
            # surface it immediately instead of wasting retries.

        raise TTSError(
            f"MiniMax TTS failed after retries: {last_error}"
        ) from last_error

    async def design_voice(self, prompt: str, preview_text: str,
                           output_path: str, voice_id: Optional[str] = None,
                           aigc_watermark: bool = False) -> str:
        """Design a custom voice from a text description (POST /v1/voice_design).

        Args:
            prompt: 音色描述（文本描述想要的声音）。
            preview_text: 试听音频文本（≤500 字符；试听合成按 2 元/万字符计费）。
            output_path: Where to save the trial audio (hex-decoded, MP3).
            voice_id: Optional custom voice_id; when omitted MiniMax generates
                one and returns it. The returned id works directly in T2A
                synthesis as ``voice_setting.voice_id``.
            aigc_watermark: Whether to append an audio rhythm watermark to the
                trial audio.

        Returns:
            The generated ``voice_id``.
        """
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )
        payload: dict = {
            "prompt": prompt,
            "preview_text": preview_text,
            "aigc_watermark": aigc_watermark,
        }
        if voice_id:
            payload["voice_id"] = voice_id

        self._log_api("voice_design 请求", voice_id=voice_id,
                      prompt=self._brief_text(prompt),
                      preview_text=self._brief_text(preview_text))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
                    response = await client.post(
                        self._design_url, headers=headers, json=payload
                    )
                    if response.status_code == 429:
                        raise _RetryableTTSError("MiniMax API rate limited (429)")
                    if response.status_code != 200:
                        body = response.text[:500]
                        raise TTSError(
                            f"MiniMax API returned {response.status_code}: {body}"
                        )
                    result = response.json()
                base_resp = result.get("base_resp") or {}
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    msg = base_resp.get("status_msg", "")
                    trace_id = result.get("trace_id", "")
                    if status_code in _RETRYABLE_CODES:
                        raise _RetryableTTSError(
                            f"MiniMax transient error {status_code}: {msg} "
                            f"(trace_id={trace_id})"
                        )
                    raise TTSError(
                        f"MiniMax API error {status_code}: {msg} "
                        f"(trace_id={trace_id})"
                    )
                new_voice_id = result.get("voice_id")
                if not new_voice_id:
                    raise TTSError(
                        f"MiniMax voice_design returned no voice_id "
                        f"(trace_id={result.get('trace_id', '')})"
                    )
                trial = result.get("trial_audio")
                if not trial:
                    raise TTSError(
                        f"MiniMax voice_design returned no trial_audio "
                        f"(trace_id={result.get('trace_id', '')})"
                    )
                try:
                    audio_bytes = bytes.fromhex(trial)
                except ValueError as e:
                    raise TTSError(
                        f"MiniMax returned malformed hex trial audio: {e} "
                        f"(trace_id={result.get('trace_id', '')})"
                    ) from e
                self._save_audio(audio_bytes, output_path)
                self._log_api("voice_design 返回", status_code=0,
                              voice_id=new_voice_id,
                              trial_audio_bytes=len(audio_bytes),
                              trace_id=result.get("trace_id"))
                return new_voice_id
            except _RetryableTTSError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax voice_design "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax voice_design "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)

        raise TTSError(
            f"MiniMax voice_design failed after retries: {last_error}"
        ) from last_error

    # ---- 音色快速复刻（voice_clone）----

    _CLONE_AUDIO_EXTS = {".mp3", ".m4a", ".wav"}
    _CLONE_MAX_BYTES = 20 * 1024 * 1024          # ≤20MB
    _CLONE_MIN_SEC = 10.0                        # voice_clone: ≥10s
    _CLONE_MAX_SEC = 300.0                       # voice_clone: ≤5min
    _PROMPT_MAX_SEC = 8.0                        # prompt_audio: <8s
    _DEMO_MAX_BYTES = 50 * 1024 * 1024           # demo_audio 下载大小上限

    async def upload_voice_file(self, file_path: str,
                                purpose: str = "voice_clone") -> int:
        """Upload an audio file for voice cloning (POST /v1/files/upload).

        ``purpose`` is either ``"voice_clone"`` (复刻原始文件, 10s–5min) or
        ``"prompt_audio"`` (示例音频 <8s). Validates existence, extension,
        size (≤20MB) and duration locally before uploading.

        Returns:
            The ``file_id`` (int) for use in :meth:`clone_voice`.
        """
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )
        path = Path(file_path)
        if not path.exists():
            raise TTSError(f"复刻音频不存在: {file_path}")
        ext = path.suffix.lower()
        if ext not in self._CLONE_AUDIO_EXTS:
            raise TTSError(
                f"复刻音频格式不支持（{ext}），仅支持 mp3/m4a/wav"
            )
        size = path.stat().st_size
        if size > self._CLONE_MAX_BYTES:
            raise TTSError(
                f"复刻音频过大（{size // (1024 * 1024)}MB），最大 20MB"
            )
        try:
            duration = get_audio_duration(str(path))
        except Exception as e:
            raise TTSError(f"无法读取复刻音频时长: {e}") from e
        if purpose == "prompt_audio":
            if duration >= self._PROMPT_MAX_SEC:
                raise TTSError(
                    f"示例音频时长 {duration:.1f}s 需小于 8s"
                )
        else:
            if duration < self._CLONE_MIN_SEC or duration > self._CLONE_MAX_SEC:
                raise TTSError(
                    f"复刻音频时长 {duration:.1f}s 需在 10s–5min 之间"
                )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        self._log_api("files/upload 请求", purpose=purpose,
                      filename=path.name, size_bytes=size,
                      duration_s=round(duration, 2))
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                with open(path, "rb") as f:
                    files = {"file": (path.name, f, "application/octet-stream")}
                    async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=300.0) as client:
                        response = await client.post(
                            self._upload_url, headers=headers,
                            data={"purpose": purpose}, files=files,
                        )
                        if response.status_code == 429:
                            raise _RetryableTTSError("MiniMax API rate limited (429)")
                        if response.status_code != 200:
                            body = response.text[:500]
                            raise TTSError(
                                f"MiniMax API returned {response.status_code}: {body}"
                            )
                        result = response.json()
                base_resp = result.get("base_resp") or {}
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    msg = base_resp.get("status_msg", "")
                    if status_code in _RETRYABLE_CODES:
                        raise _RetryableTTSError(
                            f"MiniMax transient error {status_code}: {msg}"
                        )
                    raise TTSError(
                        f"MiniMax upload error {status_code}: {msg}"
                    )
                file_id = (result.get("file") or {}).get("file_id")
                if not file_id:
                    raise TTSError("MiniMax upload returned no file_id")
                self._log_api("files/upload 返回", status_code=0,
                              file_id=file_id)
                return int(file_id)
            except _RetryableTTSError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax upload "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax upload "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)

        raise TTSError(
            f"MiniMax upload failed after retries: {last_error}"
        ) from last_error

    async def clone_voice(self, file_id: int, voice_id: Optional[str] = None,
                          text: Optional[str] = None,
                          model: Optional[str] = None,
                          prompt_audio_id: Optional[int] = None,
                          prompt_text: Optional[str] = None,
                          language_boost: Optional[str] = None,
                          text_validation: Optional[str] = None,
                          accuracy: Optional[float] = None,
                          need_noise_reduction: bool = False,
                          need_volume_normalization: bool = False,
                          output_path: Optional[str] = None) -> str:
        """Clone a voice from an uploaded file (POST /v1/voice_clone).

        Args:
            file_id: 复刻音频的 file_id（来自 :meth:`upload_voice_file`）。
            voice_id: 自定义克隆音色 ID（8-256 位、字母开头、字母数字-_、
                末位不可为 -/_）；缺省自动生成 "CloneVoice"+时间戳。
            text: 复刻试听文本（≤1000 字符；试听按 T2A 计费）。提供时
                ``model`` 必传。
            model: 试听合成模型（如 "speech-02-hd"）。
            prompt_audio_id / prompt_text: 可选示例音频（增强相似度）。
            language_boost: 可选语言增强；默认不传，由 MiniMax 自动判断。
            text_validation: 复刻样本的预期文本（≤200 字符），用于 ASR
                相似度校验。
            accuracy: ASR 相似度阈值 [0, 1]；仅配合 text_validation 使用。
            need_noise_reduction: 是否对复刻样本开启降噪。
            need_volume_normalization: 是否对复刻样本开启音量归一化。
            output_path: 可选——把返回的试听音频（demo_audio 链接）下载到
                该路径。

        Returns:
            The cloned ``voice_id`` (usable in T2A synthesis).
        """
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )
        if not voice_id:
            voice_id = f"CloneVoice{int(time.time() * 1000)}"
        if not (8 <= len(voice_id) <= 256):
            raise TTSError("克隆音色 ID 长度必须在 8–256 位之间")
        if (not voice_id[0].isalpha()
                or not all(ch.isalnum() or ch in "-_" for ch in voice_id)
                or voice_id[-1] in "-_"):
            raise TTSError("克隆音色 ID 必须以字母开头，仅含字母、数字、-、_，且不能以 - 或 _ 结尾")
        if bool(prompt_audio_id) != bool(prompt_text and prompt_text.strip()):
            raise TTSError("增强示例音频和对应文本必须同时提供")
        if text_validation and len(text_validation) > 200:
            raise TTSError("音频预期文本不能超过 200 字符")
        if accuracy is not None and not 0 <= float(accuracy) <= 1:
            raise TTSError("ASR 相似度阈值必须在 0–1 之间")
        payload: dict = {"file_id": int(file_id), "voice_id": voice_id}
        if prompt_audio_id and prompt_text:
            payload["clone_prompt"] = {
                "prompt_audio": int(prompt_audio_id),
                "prompt_text": prompt_text,
            }
        if text:
            if not model:
                raise TTSError("复刻试听文本（text）提供时必须同时指定 model")
            payload["text"] = text
            payload["model"] = model
        if language_boost:
            payload["language_boost"] = language_boost
        if text_validation:
            payload["text_validation"] = text_validation
            if accuracy is not None:
                payload["accuracy"] = float(accuracy)
        if need_noise_reduction:
            payload["need_noise_reduction"] = True
        if need_volume_normalization:
            payload["need_volume_normalization"] = True
        self._log_api("voice_clone 请求", file_id=file_id,
                      voice_id=voice_id,
                      has_prompt=bool(prompt_audio_id and prompt_text),
                      text=self._brief_text(text), model=model,
                      language_boost=language_boost,
                      has_text_validation=bool(text_validation),
                      accuracy=accuracy,
                      need_noise_reduction=need_noise_reduction,
                      need_volume_normalization=need_volume_normalization)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=300.0) as client:
                    response = await client.post(
                        self._clone_url, headers=headers, json=payload
                    )
                    if response.status_code == 429:
                        raise _RetryableTTSError("MiniMax API rate limited (429)")
                    if response.status_code != 200:
                        body = response.text[:500]
                        raise TTSError(
                            f"MiniMax API returned {response.status_code}: {body}"
                        )
                    result = response.json()
                base_resp = result.get("base_resp") or {}
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    msg = base_resp.get("status_msg", "")
                    if status_code == 2038:
                        raise TTSError(
                            "无复刻权限（2038）：请在 MiniMax 平台完成个人/企业认证后重试"
                        )
                    if status_code in _RETRYABLE_CODES:
                        raise _RetryableTTSError(
                            f"MiniMax transient error {status_code}: {msg}"
                        )
                    raise TTSError(
                        f"MiniMax clone error {status_code}: {msg}"
                    )
                if output_path:
                    demo = result.get("demo_audio")
                    if not demo:
                        raise TTSError(
                            f"MiniMax clone returned no demo_audio "
                            f"(trace_id={result.get('trace_id', '')})"
                        )
                    await self._download_demo_audio(demo, output_path)
                self._log_api("voice_clone 返回", status_code=0,
                              voice_id=voice_id,
                              demo_downloaded=bool(output_path),
                              trace_id=result.get("trace_id"))
                return voice_id
            except _RetryableTTSError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax clone "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax clone "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)

        raise TTSError(
            f"MiniMax clone failed after retries: {last_error}"
        ) from last_error

    async def _download_demo_audio(self, url: str, output_path: str) -> None:
        """Download the demo_audio URL from MiniMax (size-capped)."""
        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise TTSError(
                        f"试听音频下载失败（HTTP {resp.status_code}）"
                    )
                size = 0
                with open_write(output_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > self._DEMO_MAX_BYTES:
                            raise TTSError(
                                "试听音频超过 50MB 上限，已中止下载"
                            )
                        f.write(chunk)
                if size < 200:
                    raise TTSError(
                        f"试听音频过小（{size} bytes）"
                    )

    # ---- 音色管理（get_voice / delete_voice）----

    async def get_voices(self) -> list[dict]:
        """List the USER's own generated voices (POST /v1/get_voice).

        Only the user's cloned (voice_cloning) and designed (voice_generation)
        voices are returned — the ``system_voice`` array is dropped, and any
        known system voice IDs are filtered out as a second safeguard.

        Returns:
            List of ``{"voice_id", "voice_type", "created_time", ...}`` dicts.
        """
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
                    response = await client.post(
                        self._voices_url, headers=headers,
                        json={"voice_type": "all"}
                    )
                    if response.status_code == 429:
                        raise _RetryableTTSError("MiniMax API rate limited (429)")
                    if response.status_code != 200:
                        body = response.text[:500]
                        raise TTSError(
                            f"MiniMax API returned {response.status_code}: {body}"
                        )
                    result = response.json()
                base_resp = result.get("base_resp") or {}
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    msg = base_resp.get("status_msg", "")
                    if status_code in _RETRYABLE_CODES:
                        raise _RetryableTTSError(
                            f"MiniMax transient error {status_code}: {msg}"
                        )
                    raise TTSError(
                        f"MiniMax get_voice error {status_code}: {msg}"
                    )
                voices: list[dict] = []
                for vtype in ("voice_cloning", "voice_generation"):
                    for v in result.get(vtype) or []:
                        if not isinstance(v, dict) or not v.get("voice_id"):
                            continue
                        if v["voice_id"] in _KNOWN_VOICES:
                            continue  # 系统音色 ID，双保险排除
                        item = dict(v)
                        item.setdefault("voice_type", vtype)
                        voices.append(item)
                self._log_api("get_voice 返回", status_code=0,
                              user_voices=len(voices),
                              system_dropped=len(result.get("system_voice") or []))
                return voices
            except _RetryableTTSError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax get_voice "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax get_voice "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)

        raise TTSError(
            f"MiniMax get_voice failed after retries: {last_error}"
        ) from last_error

    async def delete_voice(self, voice_id: str, voice_type: str) -> None:
        """Delete a user-generated voice (POST /v1/delete_voice).

        Args:
            voice_id: The voice to delete (cloned or designed).
            voice_type: ``"voice_cloning"`` or ``"voice_generation"``.
        """
        if voice_type not in ("voice_cloning", "voice_generation"):
            raise TTSError(f"不支持的 voice_type: {voice_type}")
        if not self._api_key:
            raise TTSError(
                "MINIMAX_API_KEY not set. Pass api_key= or set MINIMAX_API_KEY env var."
            )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
                    response = await client.post(
                        self._delete_voice_url, headers=headers,
                        json={"voice_type": voice_type, "voice_id": voice_id}
                    )
                    if response.status_code == 429:
                        raise _RetryableTTSError("MiniMax API rate limited (429)")
                    if response.status_code != 200:
                        body = response.text[:500]
                        raise TTSError(
                            f"MiniMax API returned {response.status_code}: {body}"
                        )
                    result = response.json()
                base_resp = result.get("base_resp") or {}
                status_code = base_resp.get("status_code", 0)
                if status_code != 0:
                    msg = base_resp.get("status_msg", "")
                    if status_code in _RETRYABLE_CODES:
                        raise _RetryableTTSError(
                            f"MiniMax transient error {status_code}: {msg}"
                        )
                    raise TTSError(
                        f"MiniMax delete_voice error {status_code}: {msg}"
                    )
                self._log_api("delete_voice 返回", status_code=0,
                              voice_id=voice_id)
                return None
            except _RetryableTTSError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax delete_voice "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt + 1}] MiniMax delete_voice "
                        f"-> wait {wait}s ({type(e).__name__}: {str(e)[:200]})"
                    , file=sys.stderr)
                    await asyncio.sleep(wait)

        raise TTSError(
            f"MiniMax delete_voice failed after retries: {last_error}"
        ) from last_error

    # ---- Internals ----

    @staticmethod
    def _log_api(label: str, **fields):
        """Print one MiniMax API call summary (脱敏：不打印 api_key / 音频数据).

        String values are truncated to 120 chars so payloads stay readable.
        Goes to stderr — the IPC helper modes (--list-voices / --design-voice /
        --clone-voice / --delete-voice) must keep stdout a pure JSON payload.
        """
        parts = [f"{k}={v}" for k, v in fields.items() if v is not None]
        print(f"       [minimax] {label}: {' '.join(parts)}", file=sys.stderr)

    @staticmethod
    def _brief_text(text: str, limit: int = 60) -> str:
        if not text:
            return ""
        t = text if len(text) <= limit else text[:limit] + "…"
        return repr(t)

    @staticmethod
    def _rate_to_speed(rate: str, default_speed: float) -> float:
        """Map "+10%" / "-5%" onto MiniMax speed [0.5, 2] (clamped)."""
        try:
            factor = 1.0 + float(rate.replace("%", "")) / 100.0
        except ValueError:
            factor = default_speed
        return max(0.5, min(2.0, factor))

    async def _call_api(self, text: str, voice: str, rate: str) -> bytes:
        """POST /v1/t2a_v2 (non-streaming) and return decoded audio bytes."""
        voice_setting: dict = {"voice_id": voice if voice else ""}
        voice_setting["speed"] = self._rate_to_speed(rate, self._speed)
        voice_setting["vol"] = self._vol
        voice_setting["pitch"] = self._pitch
        # emotion intentionally omitted: testing showed worse output quality.

        payload: dict = {
            "model": self._model,
            "text": text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": self._sample_rate,
                "bitrate": self._bitrate,
                "format": self._audio_format,
                "channel": self._channel,
            },
            "subtitle_enable": False,
            "output_format": "hex",
        }
        if self._language_boost:
            payload["language_boost"] = self._language_boost

        # API 调用日志（脱敏）：明确显示实际使用的 voice_id，便于核对
        # 生成音频用的是自定义音色还是系统音色。
        self._log_api(
            "T2A 请求", model=self._model,
            voice_id=voice_setting.get("voice_id"),
            speed=voice_setting.get("speed"), vol=voice_setting.get("vol"),
            pitch=voice_setting.get("pitch"),
            language_boost=self._language_boost,
            text=self._brief_text(text),
        )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Global rate gate: keeps the aggregate request rate under MiniMax's
        # RPM cap regardless of pipeline concurrency.
        if self._min_interval_sec > 0:
            await asyncio.to_thread(_global_throttle, self._min_interval_sec)

        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
            response = await client.post(
                self._base_url, headers=headers, json=payload
            )
            if response.status_code == 429:
                raise _RetryableTTSError("MiniMax API rate limited (429)")
            if response.status_code != 200:
                body = response.text[:500]
                raise TTSError(
                    f"MiniMax API returned {response.status_code}: {body}"
                )
            result = response.json()

        base_resp = result.get("base_resp") or {}
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            trace_id = result.get("trace_id", "")
            msg = base_resp.get("status_msg", "")
            self._log_api("T2A 返回", http=response.status_code,
                          status_code=status_code, status_msg=msg,
                          trace_id=trace_id)
            if status_code in _RETRYABLE_CODES:
                raise _RetryableTTSError(
                    f"MiniMax transient error {status_code}: {msg} "
                    f"(trace_id={trace_id})"
                )
            raise TTSError(
                f"MiniMax API error {status_code}: {msg} (trace_id={trace_id})"
            )
        data = result.get("data")
        if not data or not data.get("audio"):
            raise TTSError(
                f"MiniMax returned no audio data "
                f"(trace_id={result.get('trace_id', '')})"
            )
        try:
            audio_bytes = bytes.fromhex(data["audio"])
        except ValueError as e:
            raise TTSError(
                f"MiniMax returned malformed hex audio: {e} "
                f"(trace_id={result.get('trace_id', '')})"
            ) from e
        extra = result.get("extra_info") or {}
        self._log_api(
            "T2A 返回", status_code=0, audio_bytes=len(audio_bytes),
            audio_length_ms=extra.get("audio_length"),
            audio_format=extra.get("audio_format"),
            usage_chars=extra.get("usage_characters"),
            trace_id=result.get("trace_id"),
        )
        return audio_bytes

    def _save_audio(self, audio_bytes: bytes, output_path: str) -> str:
        """Persist audio bytes (MP3 default) to ``output_path``."""
        if len(audio_bytes) < 200:
            raise TTSError(
                f"MiniMax returned only {len(audio_bytes)} bytes of audio"
            )
        with open_write(output_path, "wb") as f:
            f.write(audio_bytes)
        return output_path

    def get_available_voices(self) -> list[VoiceInfo]:
        """Return the MiniMax system voice catalog (from the docs)."""
        return [
            VoiceInfo(
                id=vid,
                name=desc.split(" · ")[-1],
                description=desc,
                gender="female" if "女" in desc or "少女" in desc else "male",
                language="zh-CN",
            )
            for vid, desc in _KNOWN_VOICES.items()
        ]
