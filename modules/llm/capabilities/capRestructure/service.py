"""Subtitle-block restructuring: modify, delete and merge adjacent rows by user instruction."""
from typing import Sequence
from ....error import LlmError, descriptor
from ...contracts import RestructurePlan, TimedSourceRow
from ..base import BatchedCapability
from .parsing import parse_restructure_plan
from .prompts import build_restructure_system


def _cell(value: str | None) -> str:
    return " ".join((value or "").split()) or "-"


def format_restructure_rows(rows: Sequence[TimedSourceRow]) -> str:
    return "\n".join(f"{row.index}\t{row.start_ms}-{row.end_ms}\t{_cell(row.speaker)}\t"
                     f"{_cell(row.source_text)}\t{_cell(row.text)}" for row in rows)


class RestructureCapability(BatchedCapability):
    async def plan(self, rows: Sequence[TimedSourceRow], summary: str, instruction: str) -> RestructurePlan:
        """Plan one batch; ``rows`` must be the batch's active fragments in draft order."""
        if not instruction.strip():
            raise ValueError("优化要求为空")
        if not rows:
            return RestructurePlan(())
        system = build_restructure_system(summary, instruction)
        user = format_restructure_rows(rows)
        budget = max(self.cfg.max_tokens, 1024 + len(rows) * 160)
        order = [row.index for row in rows]
        problem = ""
        for _ in range(2):
            raw = await self.runtime.complete(system, user, max_tokens=budget)
            try:
                return parse_restructure_plan(raw, order)
            except ValueError as error:
                problem = f"{error}；响应片段：{raw[:800]!r}"
                if self.cache:
                    self.cache.fail(self.runtime.request_key(system, user, budget),
                                    LlmError(descriptor("LLM_RESPONSE_INVALID"), detail=problem))
        raise LlmError(descriptor("LLM_RESPONSE_INVALID"), detail=f"全局字幕优化格式无效：{problem}",
                       context={"operation": "restructure"})
