"""MiMo TTS adapter — OpenAI API-compatible TTS from Xiaomi.

Uses the MiMo TTS API (https://api.xiaomimimo.com/v1).
Requires MIMO_API_KEY environment variable or api_key parameter.

Models:
    mimo-v2.5-tts (default):    Preset voices (冰糖, 茉莉, 苏打, etc.)
    mimo-v2.5-tts-voicedesign:  Voice design via text description
    mimo-v2.5-tts-voiceclone:   Voice cloning from audio sample
"""

import asyncio
import base64
import os
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from ..network import NetworkPolicy, async_client

from ..error import TTSError
from ..io import open_write
from .base import TTSAdapter
from .models import VoiceInfo


class MimoTTSAdapter(TTSAdapter):
    """MiMo TTS backend.

    Uses Xiaomi's OpenAI-compatible TTS API. Rate adjustment is applied
    via ffmpeg post-processing since the API doesn't natively support it.

    Three modes (set via ``model``):
        - ``mimo-v2.5-tts`` (standard):    Preset voices, optional description
        - ``mimo-v2.5-tts-voicedesign``:   Voice defined by text description
        - ``mimo-v2.5-tts-voiceclone``:    Voice cloned from audio sample

    Examples:
        tts = MimoTTSAdapter(
            voice_map={"default": "冰糖", "遥": "茉莉"},
            voice_description="活泼轻快的少女声",
        )
        path = await tts.synthesize("你好世界", "冰糖", "out.wav", rate="+0%")
    """

    _API_BASE = "https://api.xiaomimimo.com/v1"

    _KNOWN_VOICES = {
        "mimo_default": "默认 (Default) — 预置通用音色",
        "冰糖": "冰糖 (Rock Sugar) — 清亮甜美女声",
        "茉莉": "茉莉 (Jasmine) — 温柔自然女声",
        "苏打": "苏打 (Soda) — 清爽活力女声",
        "白桦": "白桦 (Birch) — 沉稳知性女声",
        "Mia": "Mia — 活泼女声",
        "Chloe": "Chloe — 温柔女声",
        "Milo": "Milo — 阳光男声",
        "Dean": "Dean — 成熟男声",
    }

    def __init__(
        self,
        voice_map: Optional[dict[str, str]] = None,
        model: str = "mimo-v2.5-tts",
        api_key: Optional[str] = None,
        voice_description: Optional[str] = None,
        optimize_text: bool = False,
        max_retries: int = 3,
        voice_clone_sample_path: Optional[str] = None,
    ):
        """Initialize MiMo TTS adapter.

        Args:
            voice_map: Speaker→voice preset mapping.
            model: Model name ("mimo-v2.5-tts", "mimo-v2.5-tts-voicedesign",
                   "mimo-v2.5-tts-voiceclone").
            api_key: API key (defaults to MIMO_API_KEY env var).
            voice_description: Optional voice style description sent as
                user message to guide tone/personality.
            optimize_text: Enable AI text optimization for better TTS.
            max_retries: Max retry attempts on failure.
            voice_clone_sample_path: Path to audio sample for voice clone.
        """
        super().__init__(voice_map)
        self._network_policy = NetworkPolicy()
        self._model = model
        self._api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        self._voice_description = voice_description
        self._optimize_text = optimize_text
        self._max_retries = max_retries
        self._voice_clone_sample_path = voice_clone_sample_path

    @property
    def name(self) -> str:
        return "mimo-tts"

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> str:
        """Synthesize speech via MiMo API.

        Args:
            text: Text to speak (sent as assistant message).
            voice: Preset voice ID (e.g. "mimo_default", "冰糖").
            output_path: Where to save the audio file.
            rate: Speed adjustment (applied via ffmpeg atempo since
                  the API doesn't support it natively).
            pitch: Unsupported by this backend — accepted for API parity, ignored.

        Returns:
            Path to the generated audio file.
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                audio_bytes = await self._call_api(text, voice)
                return self._save_audio(audio_bytes, output_path, rate)
            except Exception as e:
                last_error = e
                if isinstance(e, TTSError) and not e.descriptor.retryable:
                    raise
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"       [retry {attempt+1}] MiMo '{text[:20]}...' "
                        f"-> wait {wait}s ({type(e).__name__})"
                    )
                    await asyncio.sleep(wait)

        # Fallback to default voice
        default_voice = self._voice_map.get("default", "mimo_default")
        if voice != default_voice:
            print(f"       [fallback] default voice for: '{text[:30]}...'")
            try:
                audio_bytes = await self._call_api(text, default_voice)
                return self._save_audio(audio_bytes, output_path, rate)
            except Exception as e:
                last_error = e

        raise TTSError(f"MiMo TTS failed after retries: {last_error}") from last_error

    async def _call_api(self, text: str, voice: str) -> bytes:
        """Call MiMo chat completions API and return raw audio bytes."""
        if not self._api_key:
            raise TTSError(
                "MIMO_API_KEY not set. Pass api_key= or set MIMO_API_KEY env var."
            )
        headers = {
            "api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if self._voice_description:
            messages.append({"role": "user", "content": self._voice_description})
        messages.append({"role": "assistant", "content": text})

        # Build audio options per mode
        audio_opts: dict = {"format": "wav"}

        if self._model == "mimo-v2.5-tts-voicedesign":
            # Voice design — voice defined by description only, no voice field
            if self._optimize_text:
                audio_opts["optimize_text_preview"] = True
        elif self._model == "mimo-v2.5-tts-voiceclone":
            # Voice clone — embed audio sample as data URI in voice field
            if self._voice_clone_sample_path and os.path.exists(self._voice_clone_sample_path):
                mime = self._detect_mime(self._voice_clone_sample_path)
                with open(self._voice_clone_sample_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                audio_opts["voice"] = f"data:{mime};base64,{b64}"
        else:
            # Standard mode — always include preset voice
            audio_opts["voice"] = voice if voice else "mimo_default"

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "audio": audio_opts,
        }

        async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=120.0) as client:
            response = await client.post(
                f"{self._API_BASE}/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code == 429:
                raise TTSError("MiMo API rate limited (429)")
            if response.status_code != 200:
                body = response.text[:500]
                raise TTSError(
                    f"MiMo API returned {response.status_code}: {body}"
                )
            result = response.json()

        try:
            audio_b64 = result["choices"][0]["message"]["audio"]["data"]
        except (KeyError, IndexError, TypeError) as e:
            raise TTSError(f"Unexpected MiMo API response: {e}") from e

        return base64.b64decode(audio_b64)

    @staticmethod
    def _detect_mime(path: str) -> str:
        """Detect MIME type from audio file extension."""
        ext = Path(path).suffix.lower()
        return {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".wma": "audio/x-ms-wma",
        }.get(ext, "audio/wav")

    def _save_audio(self, audio_bytes: bytes, output_path: str, rate: str) -> str:
        """Save audio bytes to file, applying rate adjustment if needed."""
        if len(audio_bytes) < 100:
            raise TTSError(f"MiMo returned only {len(audio_bytes)} bytes of audio")

        if rate == "+0%":
            with open_write(output_path, "wb") as f:
                f.write(audio_bytes)
            return output_path

        # Save temp file, apply atempo, then replace
        ext = Path(output_path).suffix or ".wav"
        tmp_path = output_path + ".mimo_tmp" + ext
        with open_write(tmp_path, "wb") as f:
            f.write(audio_bytes)

        try:
            self._apply_atempo(tmp_path, output_path, rate)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
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

    def get_available_voices(self) -> list[VoiceInfo]:
        """Return known MiMo preset voices."""
        return [
            VoiceInfo(
                id=vid,
                name=desc.split("—")[0].strip(),
                description=desc,
                gender="female" if "女" in desc else "male",
                language="zh-CN" if vid not in ("Mia", "Chloe", "Milo", "Dean") else "en-US",
            )
            for vid, desc in self._KNOWN_VOICES.items()
        ]
