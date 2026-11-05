"""Factory for the subtitle business stage, independent of LLM providers."""

from pathlib import Path
from ..llm import LLMConfig, is_configured
from ..llm.factory import create_capabilities
from .stage import LLMSubtitleStage
from .contracts import SubtitleStage


def _create_llm_stage(config: dict, output_dir: str | None = None) -> SubtitleStage:
    settings = LLMConfig(**config)
    capabilities = create_capabilities(
        settings, cache_dir=output_dir,
    ) if is_configured(settings) else None
    return LLMSubtitleStage(settings, capabilities)


STAGES = {"llm": _create_llm_stage}


def create_subtitle_stage(config: dict, output_dir: str | None = None, *,
                          implementation: str = "llm") -> SubtitleStage:
    try:
        factory = STAGES[implementation]
    except KeyError as error:
        raise ValueError(f"Unknown subtitle stage: {implementation}") from error
    return factory(config, output_dir)
