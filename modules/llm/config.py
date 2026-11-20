"""Validated connection settings and application-level capability options."""
from dataclasses import dataclass, fields
import math
from typing import ClassVar

@dataclass
class LLMConnectionConfig:
    provider: str = ''
    base_url: str = ''
    api_key: str = ''
    model: str = ''
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 120.0
    max_retries: int = 2
    _DEFAULT_BASE: ClassVar[dict[str, str]] = {
        'anthropic': 'https://api.anthropic.com', 'openai': 'https://api.openai.com/v1',
    }

    def __post_init__(self):
        if (type(self.max_tokens) is not int or self.max_tokens <= 0
                or type(self.max_retries) is not int or self.max_retries < 0
                or not math.isfinite(self.timeout) or self.timeout <= 0
                or not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError('Invalid LLM connection configuration')

    def resolved_base(self) -> str:
        return (self.base_url or self._DEFAULT_BASE.get(self.provider, '')).rstrip('/')

@dataclass
class LLMConfig(LLMConnectionConfig):
    batch_size: int = 80
    tts_backend: str = ''
    tts_model: str = ''
    optimize_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError('LLM batch size must be a positive integer')

    def connection(self) -> LLMConnectionConfig:
        return LLMConnectionConfig(**{field.name: getattr(self, field.name)
                                     for field in fields(LLMConnectionConfig)})

def is_configured(cfg: LLMConnectionConfig) -> bool:
    from .providers import PROVIDERS
    return cfg.provider in PROVIDERS and bool(cfg.api_key) and bool(cfg.model)
