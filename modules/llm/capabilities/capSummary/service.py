"""Whole-document summary capability."""
from ..base import BatchedCapability
from ...formatting import _format_rows
from .prompts import _PHASE1_SYSTEM
from typing import Sequence
from ...contracts import TextRow

class SummaryCapability(BatchedCapability):
    async def summarize(self, blocks: Sequence[TextRow]) -> str:
        return await self.runtime.complete(_PHASE1_SYSTEM, _format_rows(blocks))
