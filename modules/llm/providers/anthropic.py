"""Anthropic Messages protocol."""
from ..config import LLMConnectionConfig
from .parsing import parse_response

class AnthropicProvider:
    def __init__(self, config: LLMConnectionConfig):
        self.cfg = config

    def endpoint(self) -> str:
        base = self.cfg.resolved_base()
        if base.endswith('/messages'): return base
        return base + '/messages' if base.endswith('/v1') else base + '/v1/messages'

    def headers(self) -> dict:
        return {'Content-Type': 'application/json', 'x-api-key': self.cfg.api_key,
                'anthropic-version': '2023-06-01'}

    def payload(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        return {'model': self.cfg.model, 'max_tokens': max_tokens or self.cfg.max_tokens,
                'system': system, 'messages': [{'role': 'user', 'content': user}],
                'thinking': {'type': 'disabled'}}

    def parse(self, value) -> str:
        return parse_response('anthropic', value)

    def is_direct_gemini(self) -> bool:
        return False
