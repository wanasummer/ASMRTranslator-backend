"""Single-fragment editing adapter composed from reusable LLM capabilities."""

from ..llm import LLMConfig
from ..llm.contracts import TextRow
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


def create_subtitle_editor(config: LLMConfig) -> SubtitleEditor:
    return SubtitleEditor(create_capabilities(config))
