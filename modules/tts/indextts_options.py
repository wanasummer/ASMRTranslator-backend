"""Validated generation controls supported by the official IndexTTS 2.5 UI."""

import math
from dataclasses import dataclass, fields
from typing import Literal


EmotionMode = Literal["same", "segment", "reference", "vector", "text"]
EMOTION_METHODS = {
    "same": "与音色参考音频相同",
    "segment": "使用情感参考音频",
    "reference": "使用情感参考音频",
    "vector": "使用情感向量控制",
    "text": "使用情感描述文本控制",
}


@dataclass(frozen=True)
class IndexTTSOptions:
    voice_source: Literal["segment", "reference"] = "segment"
    emotion_mode: EmotionMode = "segment"
    emotion_reference: str = ""
    emotion_vector: tuple[float, ...] = (0.0,) * 8
    emotion_text: str = ""
    emotion_random: bool = False
    language: str = "ZH"
    duration_factor: float = 1.0
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500
    max_text_tokens_per_segment: int = 120

    @classmethod
    def from_dict(cls, value: dict) -> "IndexTTSOptions":
        if not isinstance(value, dict):
            raise ValueError("IndexTTS options 必须是对象")
        unknown = value.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"IndexTTS 不支持的选项: {', '.join(sorted(unknown))}")
        values = dict(value)
        if "emotion_vector" in values:
            try:
                values["emotion_vector"] = tuple(values["emotion_vector"])
            except TypeError as exc:
                raise ValueError("IndexTTS 情感向量必须含 8 个数值") from exc
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.voice_source not in ("segment", "reference"):
            raise ValueError("IndexTTS 音色来源无效")
        if not isinstance(self.emotion_mode, str) or self.emotion_mode not in EMOTION_METHODS:
            raise ValueError("IndexTTS 情感模式无效")
        if self.language not in ("ZH", "EN", "JA", "AR", "ES"):
            raise ValueError("IndexTTS 语言无效")
        for name in ("emotion_random", "do_sample"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"IndexTTS {name} 必须是布尔值")
        for name in ("emotion_reference", "emotion_text"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"IndexTTS {name} 必须是文本")
        limits = {
            "duration_factor": (0.5, 2.0), "top_p": (0.0, 1.0),
            "top_k": (0, 100), "temperature": (0.1, 2.0),
            "length_penalty": (-2.0, 2.0), "num_beams": (1, 10),
            "repetition_penalty": (0.1, 20.0), "max_mel_tokens": (50, None),
            "max_text_tokens_per_segment": (20, None),
        }
        integers = {"top_k", "num_beams", "max_mel_tokens", "max_text_tokens_per_segment"}
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < minimum
                    or (maximum is not None and value > maximum)
                    or (name in integers and not isinstance(value, int))):
                raise ValueError(f"IndexTTS {name} 超出有效范围")
        if len(self.emotion_vector) != 8 or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or not 0 <= v <= 1 for v in self.emotion_vector
        ):
            raise ValueError("IndexTTS 情感向量必须含 8 个 0～1 的有限数值")
        if self.emotion_mode == "reference" and not self.emotion_reference.strip():
            raise ValueError("IndexTTS 需要选择独立情感参考音频")
