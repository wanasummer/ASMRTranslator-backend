"""Shared capability mechanics: batching, partial recovery errors and single-text validation."""
from dataclasses import asdict
from typing import NoReturn
from ...error import LLMError, PipelineError
from ...task_cache import cache_key, public_settings
from ..config import LLMConfig
from ..contracts import TextGenerator
from ..formatting import _format_rows
from ..parsing import _parse_single_optimized_text

class BatchedCapability:
    def __init__(self, runtime: TextGenerator, config: LLMConfig):
        self.runtime = runtime
        self.cfg = config
        self.cache = runtime.cache
        self._batch_errors: list[PipelineError] = []
        self._failed_batches = 0

    async def _run_batched(self, system: str, blocks, budget_per_row: int = 96,
                           formatter=None, parser=None,
                           incomplete_code: str = "LLM_OPTIMIZE_INCOMPLETE",
                           allow_invalid_batches: bool = False):
        """POST ``system`` for each batch of blocks, collecting raw responses.

        Returns ``(raw_texts, failed)`` — one raw response string per successful
        batch (in order) and the count of invalid batches. With
        ``allow_invalid_batches``, the caller can recover their rows itself.
        The caller parses each raw text with the phase-appropriate parser.
        """
        batches = [
            blocks[i:i + self.cfg.batch_size]
            for i in range(0, len(blocks), self.cfg.batch_size)
        ]
        raw_texts: list[str] = []
        failed = 0
        self._batch_errors = []
        for batch in batches:
            # Reasoning models spend many tokens on internal "thinking" before
            # producing the final text. A fixed budget (default 1024) gets
            # exhausted by thinking alone for a whole batch of rows, so the text
            # block never materializes. Size the budget from the row count.
            budget = max(self.cfg.max_tokens, 512 + len(batch) * budget_per_row)
            try:
                user_text = (formatter or _format_rows)(batch)
                raw = await self.runtime.complete(system, user_text, max_tokens=budget)
                raw_texts.append(raw)
                wanted = {block.index for block in batch}
                response_parser = parser or self.parse_rows
                if self.cache and wanted - set(response_parser(raw, wanted)):
                    # A transport success with omitted/invalid rows is not a
                    # reusable completed request. Parsed valid rows survive in
                    # their own records; missing rows must get a new request.
                    from ...error import descriptor
                    self.cache.fail(
                        self.runtime.request_key(system, user_text, budget),
                        LLMError(descriptor("LLM_RESPONSE_INVALID"),
                                 context={"missing_segments": sorted(wanted - set(response_parser(raw, wanted)))}),
                    )
            except LLMError as e:
                self._batch_errors.append(e)
                failed += 1
                if allow_invalid_batches:
                    print(f"       [warn] LLM 批次失败，继续后续批次并尝试定向补全: {e}")
                else:
                    print(f"       [warn] LLM 批次失败，继续后续批次: {e}")
        self._failed_batches = failed
        if failed and not allow_invalid_batches:
            from ...error import LlmError, descriptor
            raise LlmError(
                descriptor(incomplete_code),
                detail=f"{failed}/{len(batches)} LLM batches failed",
                context={"operation": "batch"},
            )
        return raw_texts, failed


    def _raise_incomplete(self, keys: dict[int, str], missing: set[int],
                          code: str, errors: list[PipelineError]) -> NoReturn:
        from ...error import LlmError, descriptor, safe_detail
        error = errors[0] if errors else LlmError(
            descriptor(code), detail=f"missing segments after retries: {sorted(missing)}")
        label = "missing optimized rows after retries" if code == "LLM_OPTIMIZE_INCOMPLETE" else "missing slices after retries"
        error.detail = safe_detail(f"{label}: {sorted(missing)}; {error.detail}")
        error.context.update({"missing_segments": sorted(missing)})
        if self.cache:
            for idx in missing:
                self.cache.fail(keys[idx], error)
        raise error


    async def _optimize_single_text(self, system: str, text: str, operation: str) -> str:
        raw = await self.runtime.complete(system, text, max_tokens=max(self.cfg.max_tokens, 512 + len(text) * 3))
        result = _parse_single_optimized_text(raw)
        if not result:
            from ...error import LlmError, descriptor
            raise LlmError(descriptor("LLM_OPTIMIZE_INCOMPLETE"),
                           detail=f"{operation}未返回有效单段文本；响应片段：{raw[:800]!r}",
                           context={"operation": operation})
        return result
