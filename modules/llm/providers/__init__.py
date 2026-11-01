"""Provider registry: wire protocols are replaceable independently of capabilities."""
from ..config import LLMConnectionConfig
from ..contracts import Provider
from ...error import LLMError
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider

PROVIDERS = {'openai': OpenAIProvider, 'anthropic': AnthropicProvider}

def create_provider(config: LLMConnectionConfig) -> Provider:
    try:
        factory = PROVIDERS[config.provider or 'openai']
    except KeyError as error:
        raise LLMError(f'Unknown LLM provider: {config.provider}') from error
    return factory(config)
