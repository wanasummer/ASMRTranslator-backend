"""OpenAI-compatible protocol, including scoped provider extensions."""
from ..config import LLMConnectionConfig
from .parsing import parse_response

class OpenAIProvider:
    def __init__(self, config: LLMConnectionConfig):
        self.cfg = config

    def _openai_token_limit_key(self) -> str:
        """Return the Chat Completions token-limit field for this model.

        GPT-5 and o-series models reject the legacy ``max_tokens`` field. Keep
        using it for older and unknown OpenAI-compatible models so custom
        endpoints retain their existing request contract.
        """
        model = self.cfg.model.lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return "max_completion_tokens"
        return "max_tokens"


    def _is_direct_deepseek(self) -> bool:
        from urllib.parse import urlsplit

        return (
            self.cfg.provider == "openai"
            and urlsplit(self.cfg.resolved_base()).hostname == "api.deepseek.com"
        )


    def is_direct_gemini(self) -> bool:
        from urllib.parse import urlsplit

        return (
            self.cfg.provider == "openai"
            and self.cfg.model.lower().startswith("gemini-")
            and urlsplit(self.cfg.resolved_base()).hostname == "generativelanguage.googleapis.com"
        )

    def endpoint(self) -> str:
        base = self.cfg.resolved_base()
        return base if base.endswith('/chat/completions') else base + '/chat/completions'

    def headers(self) -> dict:
        return {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.cfg.api_key}'}

    def payload(self, system: str, user: str, max_tokens: int | None = None) -> dict:
        result = {'model': self.cfg.model, self._openai_token_limit_key(): max_tokens or self.cfg.max_tokens,
                  'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                  'temperature': self.cfg.temperature}
        if self._is_direct_deepseek(): result['thinking'] = {'type': 'disabled'}
        return result

    def parse(self, value) -> str:
        return parse_response('openai', value)
