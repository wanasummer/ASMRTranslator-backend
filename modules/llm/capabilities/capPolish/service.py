"""Chinese subtitle polishing and same-language single-fragment editing."""
from dataclasses import asdict
from typing import Sequence
from ...contracts import TextRow
from ...formatting import _format_rows
from ....error import LLMError, PipelineError
from ....task_cache import cache_key, public_settings
from ..base import BatchedCapability
from .parsing import _parse_lines
from ...tts_style import build_polish_system

class PolishCapability(BatchedCapability):
    parse_rows = staticmethod(_parse_lines)

    async def optimize_chinese_text(self, block: TextRow, chinese: str) -> str:
        if not chinese.strip():
            raise ValueError("中文字幕为空，无法优化")
        system = build_polish_system("", self.cfg.tts_backend, self.cfg.tts_model)
        system += "\n本次只处理一条中文字幕。覆盖前面的批量输出格式：只输出优化后的中文纯文本，不要序号、制表符、代码块或解释，不要翻译成其他语言。"
        return await self._optimize_single_text(system, chinese, "优化中文文本")


    async def optimize_japanese_text(self, block: TextRow, japanese: str) -> str:
        """Polish Japanese in Japanese, without translating or adding TTS tags."""
        if not japanese.strip():
            raise ValueError("日语草稿为空，无法优化")
        system = ("你是日语字幕编辑。修复当前日语字幕的明显识别错误、断句和机械重复，"
                  "清除装饰符号、格式标记和不可见字符，使用自然日语标点。保留有意重复、"
                  "语气、专名、数字和原意，不扩写，不翻译成中文，不添加情绪标签。"
                  "只输出优化后的日语纯文本，不要序号、制表符、代码块或解释。")
        return await self._optimize_single_text(system, japanese, "优化日文文本")


    async def _recover_missing_optimized_rows(self, system: str, blocks,
                                              optimized: dict[int, str]) -> None:
        self._recovery_errors = {}
        missing_blocks = [block for block in blocks if block.index not in optimized]
        print(f"       [llm] 优化结果缺少 {len(missing_blocks)} 行，开始定向补全")

        # Small numbered batches recover long omitted runs without one request
        # per subtitle. Parse only the indices owned by that retry batch.
        used_batch_retry = len(missing_blocks) > 1
        if used_batch_retry:
            for start in range(0, len(missing_blocks), 8):
                batch = missing_blocks[start:start + 8]
                try:
                    raw = await self.runtime.complete(
                        system, _format_rows(batch),
                        max_tokens=max(self.cfg.max_tokens, 512 + len(batch) * 96),
                    )
                except PipelineError as error:
                    for block in batch:
                        self._recovery_errors[block.index] = error
                    print(f"       [warn] LLM 优化补全批次失败（{len(batch)} 行）: {error}")
                    continue
                optimized.update(_parse_lines(raw, {block.index for block in batch}))

        # A single unnumbered request cannot shift a recovered row onto another
        # subtitle's index. Retry only rows still missing.
        single_attempts = max(1, self.cfg.max_retries - int(used_batch_retry))
        single_system = (system + "\n本次只优化一条字幕。只输出优化后的纯文本，不要序号、制表符、代码块或解释。")
        rejected = set()
        for _ in range(single_attempts):
            remaining = [block for block in missing_blocks
                         if block.index not in optimized and block.index not in rejected]
            if not remaining:
                break
            for block in remaining:
                try:
                    optimized[block.index] = await self._optimize_single_text(
                        single_system, block.text, "补全中文字幕"
                    )
                except PipelineError as error:
                    self._recovery_errors[block.index] = error
                    if not error.descriptor.retryable:
                        rejected.add(block.index)
                    print(f"       [warn] LLM 优化第 {block.index} 行补全失败: {error}")
        print(f"       [llm] 优化缺行补全后已获得 {len(optimized)}/{len(blocks)} 行")


    async def polish(self, blocks: Sequence[TextRow], summary: str) -> dict[int, str]:
        system = build_polish_system(
            summary, tts_backend=self.cfg.tts_backend,
            tts_model=self.cfg.tts_model)
        wanted = {b.index for b in blocks}
        keys = {b.index: cache_key({"operation": "optimize-row", "text": b.text,
                                   "index": b.index, "summary": summary, "version": 1, "settings": public_settings(asdict(self.cfg))})
                for b in blocks}
        optimized = {}
        if self.cache:
            optimized = {b.index: value for b in blocks
                         if isinstance(value := self.cache.load(keys[b.index]), str) and value.strip()}
        pending = [b for b in blocks if b.index not in optimized]
        raw_texts, _ = await self._run_batched(
            system, pending, allow_invalid_batches=True)
        for raw in raw_texts:
            optimized.update(_parse_lines(raw, wanted))
        missing = wanted - set(optimized)
        self._recovery_errors = {}
        if missing and self.cfg.max_retries > 0:
            await self._recover_missing_optimized_rows(system, blocks, optimized)
            missing = wanted - set(optimized)
        if self.cache:
            for idx, text in optimized.items():
                self.cache.save(keys[idx], text)
        if missing:
            errors = [self._recovery_errors[idx] for idx in sorted(missing)
                      if idx in self._recovery_errors]
            self._raise_incomplete(keys, missing, "LLM_OPTIMIZE_INCOMPLETE",
                                   errors or self._batch_errors)
        self._failed_batches = 0
        return optimized
