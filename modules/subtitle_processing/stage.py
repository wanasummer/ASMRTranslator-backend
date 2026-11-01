"""Subtitle transformation: copied blocks in, translated/polished blocks out."""
import asyncio
from copy import copy
from ..error import LLMError, PipelineError
from ..llm import LLMConfig, is_configured
from ..llm.factory import LLMCapabilities
from .contracts import SubtitleStageInput, SubtitleStageResult
from .timeline import _build_sentence_blocks

class LLMSubtitleStage:
    def __init__(self, config: LLMConfig, capabilities: LLMCapabilities | None):
        self.cfg = config
        self.capabilities = capabilities
        if capabilities is not None:
            self.summarizer = capabilities.summary
            self.translator = capabilities.translation
            self.polisher = capabilities.polish

    def run(self, value: SubtitleStageInput) -> SubtitleStageResult:
        return asyncio.run(self.run_async(value))

    async def run_async(self, value: SubtitleStageInput) -> SubtitleStageResult:
        if value.source not in {'asr', 'subtitles'}:
            raise ValueError('Unknown subtitle source')
        blocks = [copy(block) for block in value.blocks]
        if not is_configured(self.cfg) or (value.source == 'subtitles' and not self.cfg.optimize_enabled):
            return SubtitleStageResult(tuple(blocks), {'enabled': False, 'translated': 0, 'rewritten': 0,
                'total': len(blocks), 'failed_batches': 0, 'summary_ok': False})
        if self.capabilities is None:
            raise ValueError('Configured subtitle stage requires LLM capabilities')
        if value.source == 'asr':
            blocks, stats = await self._translate_asr(blocks, value.preserve_timeline, value.preserve_source_timeline)
        else:
            stats = await self._polish_existing(blocks)
        return SubtitleStageResult(tuple(blocks), stats)

    async def _polish_existing(self, blocks) -> dict:
        """Run both phases and rewrite block.text in place, or fail atomically."""
        print(f"       [llm] 大模型字幕优化已启用 ({self.cfg.provider}: {self.cfg.model})")
        summary_error = None
        try:
            summary = await self.summarizer.summarize(blocks)
        except LLMError as error:
            summary_error = error
            print(f"       [llm-summary-failed] 摘要不可用，继续逐片处理: {error}")
            summary = ""
        print("       [llm] 整体情节分析完成")
        print(f"       [llm-summary] 剧情摘要：{summary}")
        try:
            optimized = await self.polisher.polish(blocks, summary)
        except PipelineError:
            if summary_error:
                raise summary_error
            raise
        rewritten = 0
        for b in blocks:
            new_text = optimized.get(b.index)
            if new_text is not None and new_text != b.text:
                b.text = new_text
                rewritten += 1
        failed = getattr(self.polisher, "_failed_batches", 0)
        msg = f"       [llm] 优化字幕：{rewritten}/{len(blocks)} 段已改写"
        if failed:
            msg += f"（{failed} 批失败已保留原文）"
        print(msg)
        return {"enabled": True, "rewritten": rewritten, "total": len(blocks),
                "failed_batches": failed, "summary_ok": summary_error is None}


    async def _translate_asr(
        self,
        slice_blocks,
        preserve_timeline=False,
        preserve_source_timeline=False,
    ):
        """Translate+split ASR slices into sentence-level Chinese blocks.

        Runs the plot summary and translate-split phase.  When subtitle
        optimization is enabled it then runs phase 2 to polish the sentence
        blocks; otherwise the translated blocks are returned directly. Enabled
        stages fail atomically when their contract cannot be fulfilled.

        Returns ``(blocks, stats)``.
        """
        feature = "字幕优化" if self.cfg.optimize_enabled else "翻译"
        print(f"       [llm] 大模型{feature}已启用 ({self.cfg.provider}: {self.cfg.model})")
        translatable = [
            block for block in slice_blocks
            if not block.skip_tts and (block.text or "").strip()
        ]
        if not translatable:
            print("       [llm] 日语清洗后没有需要翻译的字幕，全部片段保留原声")
            blocks = _build_sentence_blocks(
                slice_blocks, {}, preserve_timeline, preserve_source_timeline
            )
            return blocks, {"enabled": True, "translated": 0,
                            "untranslated": 0, "rewritten": 0,
                            "total": len(blocks), "failed_batches": 0,
                            "summary_ok": False}
        summary_error = None
        try:
            summary = await self.summarizer.summarize(translatable)
        except LLMError as error:
            summary_error = error
            print(f"       [llm-summary-failed] 摘要不可用，继续逐片处理: {error}")
            summary = ""
        print("       [llm] 整体情节分析完成")
        print(f"       [llm-summary] 剧情摘要：{summary}")
        try:
            per_slice = await self.translator.translate(translatable, summary)
        except PipelineError:
            if summary_error:
                raise summary_error
            raise
        translated = sum(len(v) for v in per_slice.values())
        untranslated = len(translatable) - len(per_slice)
        blocks = _build_sentence_blocks(
            slice_blocks,
            per_slice,
            preserve_timeline,
            preserve_source_timeline,
        )
        msg = f"       [llm] 翻译切句：{translated} 句"
        if untranslated:
            msg += f"（{untranslated} 个切片未能翻译，已直接丢弃）"
        print(msg)
        rewritten = 0
        failed = getattr(self.polisher, "_failed_batches", 0)
        if self.cfg.optimize_enabled:
            polishable = [block for block in blocks if not block.skip_tts]
            optimized = await self.polisher.polish(polishable, summary)
            for b in polishable:
                new_text = optimized.get(b.index)
                if new_text is not None and new_text != b.text:
                    b.text = new_text
                    rewritten += 1
            print(f"       [llm] 优化字幕：{rewritten}/{len(polishable)} 段已改写")
        else:
            print("       [llm] 字幕优化未启用，跳过润色与情绪标签")
        return blocks, {"enabled": True, "translated": translated,
                        "untranslated": untranslated, "rewritten": rewritten,
                        "total": len(blocks), "failed_batches": failed,
                        "summary_ok": summary_error is None}
