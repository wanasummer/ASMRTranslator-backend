"""Japanese-to-Chinese translation with explicit source ownership."""
import re
from dataclasses import asdict
from typing import Sequence
from ...contracts import TextRow
from ....error import LLMError, PipelineError
from ....task_cache import cache_key, public_settings
from ....text_normalization import _sanitize_symbols
from ..base import BatchedCapability
from ...tts_style import build_polish_system
from .prompts import _TRANSLATE_SPLIT_SYSTEM, _TRANSLATE_SINGLE_SYSTEM
from .parsing import _format_translation_request, _parse_translation_response, _suspicious_translation_ids

class TranslationCapability(BatchedCapability):
    parse_rows = staticmethod(_parse_translation_response)

    async def optimize_from_japanese(self, block: TextRow, japanese: str) -> str:
        """Regenerate exactly one Chinese row from edited Japanese, every call fresh."""
        if not japanese.strip():
            raise ValueError("当前片段没有日语原文，请先补入日语字幕")
        system = build_polish_system("", self.cfg.tts_backend, self.cfg.tts_model)
        system += "\n本次只处理一条日语原文。重新翻译成中文并优化。覆盖前面的批量输出格式：只输出优化后的纯中文文本，不要序号、制表符、代码块、解释，不要拆分或沿用旧译文。"
        return await self._optimize_single_text(system, japanese, "日语翻译成中文")


    async def _translate_split_once(self, blocks, summary, wanted) -> dict[int, list[str]]:
        """POST the translate-split prompt for ``blocks`` and parse the replies.

        Returns ``{slice_idx: [sentences]}``; rows whose slice is not in ``wanted``
        or whose content is empty are dropped.
        """
        # Use replacement instead of str.format because the prompt contains a
        # literal JSON example with braces.
        system = _TRANSLATE_SPLIT_SYSTEM.replace("{summary}", summary)
        raw_texts, _ = await self._run_batched(
            system, blocks, formatter=_format_translation_request, parser=_parse_translation_response,
            incomplete_code="LLM_TRANSLATION_INCOMPLETE",
            allow_invalid_batches=True,
        )
        per_slice: dict[int, list[str]] = {}
        for raw in raw_texts:
            for idx, sentences in _parse_translation_response(raw, wanted).items():
                per_slice.setdefault(idx, []).extend(sentences)
        return per_slice


    async def _translate_single(self, block, summary: str) -> list[str]:
        """Translate ONE slice into Chinese sentences with a numbering-free prompt.

        A single-slice request carries the raw text with no row index, so the
        model cannot renumber (e.g. start at 0 and steal another slice's index);
        every output line is attributed to ``block`` unambiguously. A leading
        ``<index>.`` / ``<index>.<n>`` prefix the model emits anyway is stripped
        when it matches ``block.index``; a line numbered with any OTHER index
        means the reply is unreliable, so the whole attempt counts as failed.
        Used for retrying slices the batched pass missed.
        """
        system = _TRANSLATE_SINGLE_SYSTEM.format(summary=summary)
        raw = await self.runtime.complete(
            system, block.text.strip(),
            max_tokens=max(self.cfg.max_tokens, 512 + 96),
        )
        text = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
        source = block.text.strip()
        sentences = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln == source:
                continue
            m = re.match(r"^(\d+)(?:\.\d+)?\s*[\.、．：\t]\s*", ln)
            if m:
                if int(m.group(1)) != block.index:
                    return []
                ln = ln[m.end():].strip()
            ln = _sanitize_symbols(ln)
            if ln:
                sentences.append(ln)
        return sentences


    async def translate(self, blocks: Sequence[TextRow], summary: str) -> dict[int, list[str]]:
        """Translate each ASR slice into Chinese sentences (``{slice: [sentences]}``).

        The first pass batches all slices with a numbered output format. Slices
        that come back with no sentences are retried one-by-one with a
        numbering-free prompt, so a missed slice gets a real second chance and
        cannot be lost to a misattributed/renumbered row. Slices still empty
        after ``cfg.max_retries`` retries are logged with a visible warning.
        """
        wanted = {b.index for b in blocks}
        keys = {b.index: cache_key({"operation": "translate-slice", "text": b.text,
                                   "index": b.index, "summary": summary, "version": 1, "settings": public_settings(asdict(self.cfg))})
                for b in blocks}
        per_slice = {}
        if self.cache:
            per_slice = {b.index: value for b in blocks
                         if isinstance(value := self.cache.load(keys[b.index]), list)
                         and value and all(isinstance(s, str) and s.strip() for s in value)}
        pending = [b for b in blocks if b.index not in per_slice]
        if pending:
            per_slice.update(await self._translate_split_once(pending, summary, wanted - set(per_slice)))
        if self.cache:
            for idx, sentences in per_slice.items():
                self.cache.save(keys[idx], sentences)
        missing = wanted - set(per_slice)
        rejected = set()
        recovery_errors = {}
        for _ in range(self.cfg.max_retries):
            if not missing:
                break
            for idx in sorted(missing - rejected):
                block = next(b for b in blocks if b.index == idx)
                try:
                    sentences = await self._translate_single(block, summary)
                except LLMError as e:
                    recovery_errors[idx] = e
                    if not e.descriptor.retryable:
                        rejected.add(idx)
                    print(f"       [warn] LLM 切片 {idx} 翻译请求失败: {e}")
                    continue
                if sentences:
                    per_slice[idx] = sentences
                    if self.cache:
                        self.cache.save(keys[idx], sentences)
            missing = wanted - set(per_slice)
        if missing:
            errors = [recovery_errors[idx] for idx in sorted(missing) if idx in recovery_errors]
            self._raise_incomplete(keys, missing, "LLM_TRANSLATION_INCOMPLETE",
                                   errors or self._batch_errors)

        # The normal path remains one whole-batch translation.  Only a
        # suspicious adjacent pair is verified separately, preventing a model
        # from borrowing the next slice's content while avoiding the token cost
        # of translating every slice one by one.
        suspicious = _suspicious_translation_ids(blocks, per_slice)
        if suspicious:
            print(f"       [llm] 检测到疑似跨切片重复，单独复核：{sorted(suspicious)}")
            block_by_index = {b.index: b for b in blocks}
            for idx in sorted(suspicious):
                try:
                    verified = await self._translate_single(
                        block_by_index[idx], summary
                    )
                except LLMError as e:
                    print(f"       [warn] LLM 切片 {idx} 独立复核失败，保留整体译文: {e}")
                    continue
                if verified:
                    per_slice[idx] = verified
                    if self.cache:
                        self.cache.save(keys[idx], verified)
        return per_slice
