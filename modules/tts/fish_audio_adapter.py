"""Fish Audio TTS adapter — cloud TTS with emotion tags.

API docs: https://api.fish.audio/v1/tts
Models: s2.1-pro-free (free), s2.1-pro (paid)
Auth:   Bearer token in Authorization header
"""

import asyncio
import base64
import binascii
import os
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import TTSError
from ..io import open_write
from .base import TTSAdapter


class FishAudioTTSAdapter(TTSAdapter):
    """Fish Audio TTS backend.

    Supports inline emotion/effect tags in text:
        [angry], [sad], [whispering], [laughing], [pause], etc.

    Examples:
        tts = FishAudioTTSAdapter(api_key="...")
        path = await tts.synthesize(
            "[whispering] 你好世界", "reference_id", "out.mp3", rate="+0%"
        )
    """

    _API_URL = "https://api.fish.audio/v1/tts"
    _MODELS_URL = "https://api.fish.audio/model"
    _VOICE_DESIGN_URL = "https://api.fish.audio/v1/voice-design"
    _KNOWN_VOICES: dict[str, str] = {}  # dynamic — user provides reference_id

    # Model name header values
    _MODEL_FREE = "s2.1-pro-free"
    _MODEL_PRO = "s2.1-pro"

    def __init__(
        self,
        voice_map: Optional[dict[str, str]] = None,
        api_key: Optional[str] = None,
        model: str = "s2.1-pro-free",
        reference_id: Optional[str] = None,
        latency: str = "normal",
        output_format: str = "mp3",
        max_retries: int = 3,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize Fish Audio TTS adapter.

        Args:
            voice_map: Speaker→reference_id mapping.
            api_key: Fish Audio API key (defaults to FISH_API_KEY env var).
            model: Model name ("s2.1-pro-free" or "s2.1-pro").
            reference_id: Default voice model reference ID (used when
                voice_map doesn't match and no voice override given).
            latency: "normal", "balanced", or "interactive".
            output_format: "mp3", "wav", or "pcm".
            max_retries: Max retry attempts on failure.
        """
        super().__init__(voice_map)
        self._network_policy = NetworkPolicy()
        self._api_key = api_key or os.environ.get("FISH_API_KEY", "")
        self._model = model
        self._default_reference_id = reference_id
        self._latency = latency
        self._output_format = output_format
        self._max_retries = max_retries
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        return "fish-audio"

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synthesize speech via Fish Audio API.

        Args:
            text: Text to speak (supports inline [emotion] tags).
            voice: Voice reference_id (or "default" to use self._default_reference_id).
            output_path: Where to save the audio file.
            rate: Speed adjustment (applied via ffmpeg atempo post-processing).
            pitch: Unsupported by this backend — accepted for API parity, ignored.

        Returns:
            Path to the generated audio file.
        """
        if not self._api_key:
            raise TTSError(
                "FISH_API_KEY not set. Pass api_key= or set FISH_API_KEY env var."
            )

        # Resolve reference_id
        ref_id = None
        if voice and voice != "default":
            ref_id = voice
        elif self._default_reference_id:
            ref_id = self._default_reference_id

        last_error = None

        for attempt in range(self._max_retries):
            try:
                audio_bytes = await self._call_api(text, ref_id)
                return self._save_audio(audio_bytes, output_path, rate)
            except Exception as e:
                last_error = e
                if isinstance(e, TTSError) and not e.descriptor.retryable:
                    raise
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt+1}] FishAudio '{text[:20]}...' "
                        f"-> wait {wait}s ({type(e).__name__})"
                    )
                    await asyncio.sleep(wait)

        raise TTSError(
            f"Fish Audio TTS failed after retries: {last_error}"
        ) from last_error

    async def _call_api(self, text: str, reference_id: Optional[str]) -> bytes:
        """Call Fish Audio TTS API and return raw audio bytes."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "model": self._model,
        }

        payload: dict = {"text": text}
        if reference_id:
            payload["reference_id"] = reference_id
        if self._output_format != "mp3":
            payload["format"] = self._output_format
        if self._latency != "normal":
            payload["latency"] = self._latency

        client = self._get_synthesis_client()
        response = await client.post(
            self._API_URL,
            headers=headers,
            json=payload,
        )
        if response.status_code == 429:
            raise TTSError("Fish Audio API rate limited (429)")
        if response.status_code == 402:
            raise TTSError("Fish Audio API: insufficient credits (402)")
        if response.status_code != 200:
            body = response.text[:500]
            raise TTSError(
                f"Fish Audio API returned {response.status_code}: {body}"
            )

        return response.content

    def _get_synthesis_client(self) -> httpx.AsyncClient:
        """Return the connection-pooled client shared by this TTS batch."""
        if self._client is None:
            self._client = async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0)
        return self._client

    async def aclose(self) -> None:
        """Close the owned synthesis pool after the batch completes."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get_voices(self, page_size: int = 50) -> list[dict]:
        """Return all voice models owned by the current Fish Audio account."""
        if not self._api_key:
            raise TTSError("Fish Audio API Key 未配置")

        headers = {"Authorization": f"Bearer {self._api_key}"}
        voices: list[dict] = []
        page_number = 1

        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=60.0) as client:
            while True:
                response = await client.get(
                    self._MODELS_URL,
                    headers=headers,
                    params={
                        "self": "true",
                        "page_size": page_size,
                        "page_number": page_number,
                    },
                )
                if response.status_code != 200:
                    raise TTSError(
                        f"Fish Audio 音色列表请求失败 ({response.status_code}): "
                        f"{response.text[:500]}"
                    )

                payload = response.json()
                for item in payload.get("items", []):
                    reference_id = (
                        item.get("reference_id")
                        or item.get("_id")
                        or item.get("id")
                    )
                    if not reference_id:
                        continue
                    voices.append({
                        "reference_id": str(reference_id),
                        "name": str(item.get("title") or item.get("name") or "未命名音色"),
                        "description": str(item.get("description") or ""),
                        "visibility": str(item.get("visibility") or "private"),
                        "state": str(item.get("state") or ""),
                        "languages": item.get("languages") or [],
                        "tags": item.get("tags") or [],
                    })

                if not payload.get("has_more"):
                    break
                page_number += 1

        return voices

    async def delete_voice(self, reference_id: str) -> None:
        """Delete one voice model owned by the current Fish Audio account."""
        if not self._api_key:
            raise TTSError("Fish Audio API Key 未配置")
        if not reference_id.strip():
            raise TTSError("缺少要删除的 Fish Audio 音色")

        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=60.0) as client:
            response = await client.delete(
                f"{self._MODELS_URL}/{reference_id}",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.status_code not in (200, 204):
                raise TTSError(
                    f"Fish Audio 删除音色失败 ({response.status_code}): "
                    f"{response.text[:500]}"
                )

    async def design_voice(
        self,
        instruction: str,
        reference_text: str,
        output_path: str,
        language: str = "zh",
    ) -> dict:
        """Generate one Fish Voice Design candidate and save its WAV preview."""
        if not self._api_key:
            raise TTSError("Fish Audio API Key 未配置")
        if not instruction.strip():
            raise TTSError("请输入音色描述")
        if len(instruction) > 2000:
            raise TTSError("音色描述不能超过 2000 个字符")
        if len(reference_text) > 300:
            raise TTSError("试听文本不能超过 300 个字符")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "model": "voice-design-1",
        }
        payload = {
            "instruction": instruction.strip(),
            "language": language or None,
            "n": 1,
        }
        if reference_text.strip():
            payload["reference_text"] = reference_text.strip()

        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=180.0) as client:
            response = await client.post(
                self._VOICE_DESIGN_URL,
                headers=headers,
                json=payload,
            )
            if response.status_code != 200:
                raise TTSError(
                    f"Fish Audio 音色设计失败 ({response.status_code}): "
                    f"{response.text[:500]}"
                )
            candidates = response.json().get("candidates") or []
            if not candidates or not candidates[0].get("audio_base64"):
                raise TTSError("Fish Audio 音色设计没有返回候选音频")
            candidate = candidates[0]
            try:
                audio = base64.b64decode(candidate["audio_base64"], validate=True)
            except (ValueError, binascii.Error) as error:
                raise TTSError("Fish Audio 返回的候选音频格式无效") from error
            if not audio:
                raise TTSError("Fish Audio 返回了空的候选音频")
            with open_write(output_path, "wb") as file:
                file.write(audio)
            return {
                "candidate_id": str(candidate.get("id") or ""),
                "sample_rate": candidate.get("sample_rate"),
                "duration_ms": candidate.get("duration_ms"),
                "audio_path": output_path,
            }

    async def create_voice(
        self,
        title: str,
        audio_paths: list[str],
        description: str = "",
        texts: Optional[list[str]] = None,
        enhance_audio_quality: bool = True,
    ) -> dict:
        """Create a persistent private Fish Audio voice from reference clips."""
        if not self._api_key:
            raise TTSError("Fish Audio API Key 未配置")
        if not title.strip():
            raise TTSError("请输入音色名称")
        if not audio_paths:
            raise TTSError("请选择参考音频")
        for path in audio_paths:
            if not Path(path).is_file():
                raise TTSError(f"参考音频不存在: {path}")
            if Path(path).suffix.lower() not in {".wav", ".mp3", ".m4a", ".opus"}:
                raise TTSError("Fish Audio 参考音频仅支持 wav、mp3、m4a 或 opus")

        data = {
            "type": "tts",
            "title": title.strip(),
            "description": description.strip(),
            "visibility": "private",
            "train_mode": "fast",
            "enhance_audio_quality": str(enhance_audio_quality).lower(),
        }
        if texts:
            data["texts"] = [text for text in texts if text.strip()]

        handles = []
        try:
            files = []
            for path in audio_paths:
                handle = open(path, "rb")
                handles.append(handle)
                suffix = Path(path).suffix.lower()
                mime = {
                    ".wav": "audio/wav",
                    ".mp3": "audio/mpeg",
                    ".m4a": "audio/mp4",
                    ".opus": "audio/opus",
                }.get(suffix, "application/octet-stream")
                files.append(("voices", (Path(path).name, handle, mime)))

            async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=240.0) as client:
                response = await client.post(
                    self._MODELS_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    data=data,
                    files=files,
                )
                if response.status_code not in (200, 201):
                    raise TTSError(
                        f"Fish Audio 音色克隆失败 ({response.status_code}): "
                        f"{response.text[:500]}"
                    )
                model = response.json()
        finally:
            for handle in handles:
                handle.close()

        reference_id = model.get("_id") or model.get("id") or model.get("reference_id")
        if not reference_id:
            raise TTSError("Fish Audio 创建音色成功，但响应中缺少音色 ID")
        return {
            "reference_id": str(reference_id),
            "name": str(model.get("title") or title.strip()),
            "state": str(model.get("state") or "created"),
        }

    def _save_audio(self, audio_bytes: bytes, output_path: str, rate: str) -> str:
        """Save audio bytes to file, applying rate adjustment if needed."""
        if len(audio_bytes) < 200:
            raise TTSError(
                f"Fish Audio returned only {len(audio_bytes)} bytes of audio"
            )

        if rate == "+0%":
            with open_write(output_path, "wb") as f:
                f.write(audio_bytes)
            return output_path

        # Save temp file, apply atempo, then replace
        ext = Path(output_path).suffix or f".{self._output_format}"
        tmp_path = output_path + ".fish_tmp" + ext
        with open_write(tmp_path, "wb") as f:
            f.write(audio_bytes)

        try:
            self._apply_atempo(tmp_path, output_path, rate)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 200:
            raise TTSError(f"Rate adjustment failed for: {output_path}")

        return output_path

    def _apply_atempo(self, src: str, dst: str, rate: str):
        """Apply time-stretching via ffmpeg atempo filter."""
        try:
            factor = 1.0 + float(rate.replace("%", "")) / 100.0
        except ValueError:
            factor = 1.0

        ffmpeg = str(
            Path(__file__).parent.parent.parent / "ffmpeg_bin" / "ffmpeg.exe"
        )
        if not Path(ffmpeg).exists():
            ffmpeg = "ffmpeg"

        result = subprocess.run(
            [ffmpeg, "-y", "-i", src, "-filter:a", f"atempo={factor}", dst],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise TTSError(
                f"ffmpeg atempo failed (rate={rate}, factor={factor}): "
                f"{result.stderr.strip()}"
            )

    def get_available_voices(self) -> list:
        """Fish Audio voices are account-specific and loaded through get_voices()."""
        return []
