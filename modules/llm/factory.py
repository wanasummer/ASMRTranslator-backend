"""Compose registered LLM capabilities around one task-owned runtime."""

from dataclasses import dataclass
from .config import LLMConfig
from .contracts import Summary, Translation, Polish, Restructure, TextGenerator
from .runtime import LLMRuntime
from .capabilities.capSummary import SummaryCapability
from .capabilities.capTranslation import TranslationCapability
from .capabilities.capPolish import PolishCapability
from .capabilities.capRestructure import RestructureCapability


CAPABILITIES = {
    "summary": SummaryCapability,
    "translation": TranslationCapability,
    "polish": PolishCapability,
    "restructure": RestructureCapability,
}


@dataclass(frozen=True)
class LLMCapabilities:
    runtime: TextGenerator
    summary: Summary
    translation: Translation
    polish: Polish
    restructure: Restructure


def create_capability(name: str, runtime: TextGenerator, config: LLMConfig) -> Summary | Translation | Polish | Restructure:
    try:
        factory = CAPABILITIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown LLM capability: {name}") from error
    return factory(runtime, config)


def create_capabilities(config: LLMConfig, cache_dir: str | None = None, *,
                        runtime: TextGenerator | None = None) -> LLMCapabilities:
    runtime = runtime if runtime is not None else LLMRuntime(config, cache_dir)
    return LLMCapabilities(
        runtime,
        create_capability("summary", runtime, config),
        create_capability("translation", runtime, config),
        create_capability("polish", runtime, config),
        create_capability("restructure", runtime, config),
    )
