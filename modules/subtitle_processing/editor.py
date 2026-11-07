"""Fragment editing adapter composed from reusable LLM capabilities."""

from typing import Sequence

from ..llm import LLMConfig
from ..llm.contracts import RestructurePlan, TextRow, TimedSourceRow
from ..llm.factory import LLMCapabilities, create_capabilities


class SubtitleEditor:
    def __init__(self, capabilities: LLMCapabilities):
        self.capabilities = capabilities

    async def optimize_from_japanese(self, block: TextRow, japanese: str) -> str:
        return await self.capabilities.translation.optimize_from_japanese(block, japanese)

    async def optimize_chinese_text(self, block: TextRow, chinese: str) -> str:
        return await self.capabilities.polish.optimize_chinese_text(block, chinese)

    async def optimize_japanese_text(self, block: TextRow, japanese: str) -> str:
        return await self.capabilities.polish.optimize_japanese_text(block, japanese)

    async def summarize(self, rows: Sequence[TextRow]) -> str:
        return await self.capabilities.summary.summarize(rows)

    async def restructure(self, rows: Sequence[TimedSourceRow], summary: str, instruction: str) -> RestructurePlan:
        """Plan one batch of active fragments in draft order; merges never leave the batch."""
        return await self.capabilities.restructure.plan(rows, summary, instruction)


def create_subtitle_editor(config: LLMConfig) -> SubtitleEditor:
    return SubtitleEditor(create_capabilities(config))
