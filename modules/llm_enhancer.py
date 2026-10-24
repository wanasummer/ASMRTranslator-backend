"""LLM subtitle enhancement — two-phase prompt engineering for TTS-friendly text.

When configured, the pipeline asks an external LLM (Anthropic Messages API or an
OpenAI-compatible chat-completions endpoint) to:

  Phase 1: read the whole subtitle list and produce a short plot summary;
  Phase 2: for every block, fix awkward wording, add proper Chinese punctuation,
           and prepend a single English emotion tag ([whispering] [angry] ...)
           when the scene calls for one.

The rewritten text is written back into each block's `text` field, which is the
single field that flows into every segment's TTS synthesis (see pipeline.py).
Emotion tags are kept in the stored content but stripped at synthesis time for
backends that do not support them (see tts_engine.prepare_text_for_synthesis).

Fully offline-testable: the only external dependency is httpx, and tests mock
`httpx.AsyncClient`.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import NoReturn, Optional

import httpx

from .network import NetworkPolicy, async_client

from .error import LLMError, PipelineError
from .task_cache import JsonResultCache, cache_key, public_settings


@dataclass
class LLMConfig:
    """Connection settings for the subtitle-enhancement LLM."""

    provider: str = ""            # "openai" | "anthropic"
    base_url: str = ""            # "" → provider default
    api_key: str = ""
    model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    batch_size: int = 80          # phase-2 rows per API call
    timeout: float = 120.0
    # TTS backend context: drives which emotion/voice tags the phase-2 prompt
    # may emit — fish-audio → [whispering]; minimax-tts + speech-2.8 → (laughs);
    # anything else → no tags at all (stripped/unsupported).
    tts_backend: str = ""
    tts_model: str = ""
    # Front-end "字幕优化" switch.  Translation can remain enabled while this
    # is false; in that mode ASR text is translated/split but never polished or
    # decorated with phase-2 emotion tags.
    optimize_enabled: bool = True
    max_retries: int = 2

    _DEFAULT_BASE: dict[str, str] = field(default_factory=lambda: {
        "anthropic": "https://api.anthropic.com",
        "openai": "https://api.openai.com/v1",
    })

    def resolved_base(self) -> str:
        return (self.base_url or self._DEFAULT_BASE.get(self.provider, "")).rstrip("/")


def is_configured(cfg: LLMConfig) -> bool:
    return cfg.provider in ("openai", "anthropic") and bool(cfg.api_key) and bool(cfg.model)


_TAG_RE = re.compile(r"\[[A-Za-z]+\]")
_SPACE_RE = re.compile(r"\s+")
# Emotion tags the phase-2 prompt may emit. Also used to recognize a leading
# bare-word tag the model occasionally writes without brackets ("whispering<TAB>
# 文本") so it can be normalized — while real English words ("OK 好的") survive.
_ALLOWED_TAGS = frozenset({
    # 声音/表现方式
    "whispering", "softly", "laughing", "sighing", "pause",
    # 情绪类
    "angry", "sad", "happy", "excited", "surprised",
    "fearful", "anxious", "embarrassed", "gentle",
    # 关系/表达意图类
    "comforting", "affectionate", "playful", "teasing",
    "hesitant", "serious",
})
_BARE_TAG_RE = re.compile(r"^\s*([A-Za-z]+)\s+")


def strip_emotion_tags(text: str) -> str:
    """Remove English emotion tags while keeping the words.

    Handles bracket form ([whispering] 你好) and a leading bare-word tag
    (whispering 你好) that the model sometimes emits unbracketed. Only words in
    _ALLOWED_TAGS are removed, so real English content is preserved.
    """
    stripped = _TAG_RE.sub("", text)
    m = _BARE_TAG_RE.match(stripped)
    if m and m.group(1).lower() in _ALLOWED_TAGS:
        stripped = stripped[m.end():]
    return _SPACE_RE.sub(" ", stripped).strip()


def _normalize_tags(text: str) -> str:
    """Rewrite a leading bare emotion tag into canonical bracket form.

    "whispering<TAB>这里是..." -> "[whispering] 这里是...". Applied to parsed
    content so stored block.text — and therefore the exported VTT and Fish
    Audio input — always uses the bracket form.
    """
    m = _BARE_TAG_RE.match(text)
    if m and m.group(1).lower() in _ALLOWED_TAGS:
        rest = text[m.end():].lstrip()
        if rest:
            return f"[{m.group(1).lower()}] {rest}".rstrip()
    return text

_WAVE_DASHES = frozenset({"\uff5e", "\u301c", "\u223c"})

_SPACE_COLLAPSE_RE = re.compile(r"[\s\u3000]+")


def _sanitize_symbols(text: str) -> str:
    """Drop symbol/emoji chars but convert all wave-dash variants into standard '~'."""
    if not text:
        return text
    out = []
    for ch in text:
        # 1. 如果是变体波浪线或标准半角波浪线，统一替换为标准半角 '~'
        if ch in _WAVE_DASHES or ch == "~":
            out.append("~")
            continue

        # 2. 保留字母、数字、标记、标点和空白字符（剔除 Emoji 与其他特殊符号）
        if unicodedata.category(ch)[0] in ("L", "N", "M", "P", "Z"):
            out.append(ch)

    return _SPACE_COLLAPSE_RE.sub(" ", "".join(out)).strip()


# ---- Prompt engineering (Chinese; the output content is Chinese) ----

_PHASE1_SYSTEM = (
    "你是 ASMR 剧情分析助手。下面是一段 ASMR 音频的字幕，每行格式为："
    "序号<TAB>说话人<TAB>文本。请通读全部字幕，输出一段简短的中文情节摘要，"
    "说明：大致场景、主要人物、情绪氛围与走向变化。只输出摘要本身，"
    "不要编号，不要复述台词。"
)

_PHASE2_SYSTEM = """你是专业的 ASMR 字幕润色与语气标注专家。

这是该 ASMR 音频的整体剧情摘要：
{summary}

请逐条处理下面的字幕。对每一条字幕执行以下操作：

1. 修复语义不通顺、明显笔误、错别字或明显的口语识别错误。
2. 根据上下文补充自然、符合中文习惯的标点符号。
3. 根据剧情、上下文和说话人的表达意图，为该句选择一个最合适的英文情绪/语气标签。
4. 不改变原句的说话人身份、核心含义、事件关系或剧情信息。
5. 不合并字幕，不拆分字幕，不重新排列字幕。

【标签选择原则】

标签必须反映“这一句话应该以什么状态说出来”，而不是仅仅根据某个词判断。

可选标签及含义：

情绪类：
[angry] 明显生气、责备、恼怒、斥责
[sad] 悲伤、失落、难过、低落
[happy] 明显开心、愉悦、欢快
[excited] 明显兴奋、激动、期待感强
[surprised] 对意外情况产生明显惊讶
[fearful] 害怕、恐惧、受到惊吓
[anxious] 紧张、不安、担忧
[embarrassed] 害羞、尴尬、不好意思

关系/表达意图类：
[comforting] 安慰、哄慰、让对方安心
[affectionate] 亲昵、宠溺、温柔地表达亲近
[playful] 轻松、俏皮、带玩笑感
[teasing] 故意逗弄、调侃、戏弄对方
[hesitant] 犹豫、迟疑、欲言又止
[serious] 严肃、郑重、认真

声音/表现方式：
[whispering] 明确使用耳语、低声耳语或气声表达
[softly] 明显轻柔、轻声表达，但重点不是安慰或亲昵
[laughing] 包含笑声，或明显是在笑着说话
[sighing] 包含叹气、长叹等明显叹息
[pause] 该字幕主要表现明显停顿、迟疑或停顿效果

【标签优先级】

如果一句话同时符合多个标签，只选择一个最主要、最能指导语音表现的标签。

优先级：

1. [laughing] / [sighing] / [pause]
2. 强烈情绪：[angry] / [sad] / [fearful] / [surprised] / [excited]
3. 明确的互动意图：[comforting] / [affectionate] / [teasing] / [playful]
4. 特殊状态：[anxious] / [embarrassed] / [hesitant] / [serious]
5. 发声方式：[whispering] / [softly]
6. 如果没有明显特征，则不添加标签。

【重要判断规则】

* 不要因为句子中出现“笑”“哭”“害怕”“惊讶”等字面词语就机械添加对应标签，必须结合上下文判断。
* [whispering] 表示“怎么说”，不是“什么情绪”。
* [softly] 表示轻柔的表达方式，不等于温柔、安慰或亲昵。
* [comforting] 用于安慰对方，不要仅因为语气温柔就使用。
* [affectionate] 用于明显带有亲昵、宠溺、亲近意味的表达。
* [teasing] 和 [playful] 用于互动中的调侃、逗弄或俏皮表达。
* [sad]、[angry]、[fearful] 等强情绪必须有足够的上下文依据，不要过度标注。
* 普通陈述句、普通问句、普通回应不要强行添加情绪标签。
* 无法确定时，不添加标签，并保持原句内容。

【标签格式】

有标签：
[标签] 优化后的文本

无标签：
优化后的文本

标签必须是英文方括号格式，例如：
[whispering] 你还醒着吗？
[comforting] 别怕，我会陪着你的。
[teasing] 原来你这么容易害羞啊。

绝对不要输出裸标签，例如：
whispering 你还醒着吗？

【输出格式】

必须逐行输出，每行严格使用：

序号<TAB>优化后的文本

例如：

1<TAB>[whispering] 你还醒着吗？
2<TAB>[comforting] 别害怕，我就在这里。
3<TAB>你今天过得怎么样？

不要输出任何解释、分析、分类说明或额外文字。"""

# MiniMax 语气词标签（speech-2.8 系列支持）——插在文本中对应位置，例如
# "（吸气）(inhale) 你还醒着吗？" 或 "哈哈(laughs) 你说得对。"
_MINIMAX_TAGS = (
    "(laughs)（笑声）", "(chuckle)（轻笑）", "(coughs)（咳嗽）",
    "(clear-throat)（清嗓子）", "(groans)（呻吟）", "(breath)（正常换气）",
    "(pant)（喘气）", "(inhale)（吸气）", "(exhale)（呼气）",
    "(gasps)（倒吸气）", "(sniffs)（吸鼻子）", "(sighs)（叹气）",
    "(snorts)（喷鼻息）", "(burps)（打嗝）", "(lip-smacking)（咂嘴）",
    "(humming)（哼唱）", "(hissing)（嘶嘶声）", "(emm)（嗯）", "(sneezes)（喷嚏）",
)

_PHASE2_NO_TAG = """你是专业的 ASMR 字幕润色专家。

这是该 ASMR 音频的整体剧情摘要：
{summary}

请逐条处理下面的字幕。对每一条字幕执行以下操作：

1. 修复语义不通顺、明显笔误、错别字或明显的口语识别错误。
2. 根据上下文补充自然、符合中文习惯的标点符号。
3. 不改变原句的说话人身份、核心含义、事件关系或剧情信息。
4. 不合并字幕，不拆分字幕，不重新排列字幕。

【重要】当前 TTS 引擎不支持任何情绪/语气标签，因此：
* 不要添加任何标签、标记、括号注释或附加说明。
* 只输出润色后的纯文本，保持原句完整。

【输出格式】

必须逐行输出，每行严格使用：

序号<TAB>优化后的文本

例如：

1<TAB>你还醒着吗？
2<TAB>别害怕，我就在这里。
3<TAB>你今天过得怎么样？

不要输出任何解释、分析、分类说明或额外文字。"""

# MiniMax speech-2.8 系列的 phase-2 规范：用官方语气词标签替换方括号情绪标签。
_PHASE2_MINIMAX28 = """你是专业的 ASMR 字幕润色与语气标注专家。

这是该 ASMR 音频的整体剧情摘要：
{summary}

请逐条处理下面的字幕。对每一条字幕执行以下操作：

1. 修复语义不通顺、明显笔误、错别字或明显的口语识别错误。
2. 根据上下文补充自然、符合中文习惯的标点符号。
3. 根据剧情、上下文和说话人的表达意图，在文本中的对应位置插入最合适的语气词标签。
4. 不改变原句的说话人身份、核心含义、事件关系或剧情信息。
5. 不合并字幕，不拆分字幕，不重新排列字幕。

【语气词标签】

目标 TTS 引擎（MiniMax speech-2.8）支持以下语气词标签，必须插在语气
发生的位置（如笑声处、叹气处），而不是句首：

{tags}

【标签选择原则】

* 标签只反映“这一句话里实际出现的声音效果”（笑声、叹气、吸气等），
  不要仅因为某个词出现就机械添加，必须结合上下文判断。
* 只有实际有对应声音效果时才插入标签；普通陈述句、普通问句不要加。
* 不要添加任何情绪词标签（如 happy/angry/sad/whispering 等），
  这些不在支持列表中。
* 无法确定时，不添加标签，并保持原句内容。

【输出格式】

必须逐行输出，每行严格使用：

序号<TAB>优化后的文本

例如：

1<TAB>(inhale) 你还醒着吗？
2<TAB>别怕，我会陪着你的。
3<TAB>哈哈哈(laughs) 原来你这么容易害羞啊。
4<TAB>(sighs) 这一天终于结束了。

不要输出任何解释、分析、分类说明或额外文字。"""


TTS_READABILITY_RULES = """
【面向 TTS 朗读的统一优化规则】
清除 ASR 窗口重叠导致的重复词句、机械叠词和断句残片，但保留有意的强调、
语气词、笑声及有意义的重复，不可删除相邻片段各自真实说出的内容。
去除字幕时间戳、说话人前缀、HTML/Markdown、装饰符号、Emoji、不可见控制符
及不适合朗读的舞台说明；标点使用自然的逗号、句号、问号、叹号和必要停顿，
不要堆叠波浪号、星号、括号或省略号。数字、日期、单位、缩写和特殊符号在
含义明确时改为自然的中文读法，不可猜测不明确的专名、数字或含义。
修复断裂句、错别字与生硬直译，使口语自然、句子易朗读，保留原有语气、
人物关系、专名及核心信息，不扩写剧情，不凭空添加内容，不改变片段边界。
情绪和声音标签仅按当前引擎明确支持的规则输出，不支持的标签不应被朗读。
"""


def build_phase2_system(summary: str, tts_backend: str = "",
                        tts_model: str = "") -> str:
    """Pick the phase-2 system prompt matching the TTS backend's tag support.

    - fish-audio: bracket emotion tags ([whispering]) are kept for the API.
    - minimax-tts + speech-2.8-hd/turbo: MiniMax voice-effect tags ((laughs)).
    - anything else: no tags (unsupported, would be read aloud / rejected).
    """
    if tts_backend == "fish-audio":
        return _PHASE2_SYSTEM.format(summary=summary) + TTS_READABILITY_RULES
    if tts_backend == "minimax-tts" and tts_model in (
            "speech-2.8-hd", "speech-2.8-turbo"):
        tags = "\n".join(f"* {t}" for t in _MINIMAX_TAGS)
        return _PHASE2_MINIMAX28.format(summary=summary, tags=tags) + TTS_READABILITY_RULES
    return _PHASE2_NO_TAG.format(summary=summary) + TTS_READABILITY_RULES

_TRANSLATE_SPLIT_SYSTEM = """你是面向中文语音合成（TTS）的日语 ASMR 字幕翻译器。

整体剧情摘要仅用于理解人物关系和语气，绝不能把摘要中出现、但当前切片
没有说出的内容补进译文：
{summary}

用户会提供一个 JSON 数组。每个对象包含：
- slice_id：音频切片的唯一整数编号。
- source_text：这个切片实际识别到的日语原文。

请分别翻译每一个切片，并严格遵守以下规则：

【内容归属】
1. 每个 slice_id 是封闭边界。该切片的译文只能来自自己的 source_text。
2. 不得借用、提前翻译或重复相邻切片的内容。即使一句话在边界处不完整，
   也只能翻译当前切片实际出现的部分，不得根据下一切片擅自补全。
3. 原文中的一项内容只翻译一次；不要用近义改写重复同一个意思。
4. 不合并不同切片，不移动内容，不改变人物关系、事件和说话意图。
5. 不漏译称呼、口语词、拟声词、犹豫、停顿和有实际含义的重复表达。

【适合 TTS】
1. 使用自然口语化中文，听起来像真人说话，不要生硬直译或书面腔。
2. 按语义和呼吸节奏拆成可独立朗读的句子。优先使用逗号、顿号、问号、
   叹号和省略号形成自然停顿；不要为了凑长度强行拆句。
3. 单句尽量简洁；过长句在自然停顿处分开，但短语气词和未说完的话要保留。
4. 不添加任何英文情绪标签、方括号标签、说话人标签、舞台说明或原文没有的
   解释。不要输出爱心、Emoji 或装饰符号。
5. 保持 ASMR 对话应有的亲近感和语气，但不能扩写剧情。

【输出格式】
只输出合法 JSON 数组，不要 Markdown 代码块，不要解释。每个输入对象必须且
只能对应一个输出对象，顺序和 slice_id 必须与输入完全一致：
[
  {"slice_id": 0, "sentences": ["第一句。", "第二句。"]},
  {"slice_id": 1, "sentences": ["这一片的译文。"]}
]

sentences 必须是非空字符串数组。绝对不要把 slice_id=1 的内容放进
slice_id=0，也不要为了让前一片读起来完整而使用后一片的信息。"""

_TRANSLATE_SINGLE_SYSTEM = (
    "你是面向中文 TTS 的日语 ASMR 字幕翻译器。整体剧情摘要只用于理解语气，"
    "不得用来补充原文没有的信息：\n{summary}\n\n"
    "下面只有一个音频切片。请只翻译实际出现的内容，不能根据上下文补全"
    "切片外的台词。使用自然口语中文，按语义和呼吸节奏切成适合独立朗读的"
    "句子，保留称呼、语气词、拟声词和未说完的话。不要添加英文情绪标签、"
    "说话人标签、解释、爱心或 Emoji。\n"
    "每行只输出一个中文句子，不要输出序号或额外文字。"
)


def _format_translation_request(blocks) -> str:
    """Serialize source slices with explicit, non-positional ownership."""
    return json.dumps([
        {"slice_id": b.index, "source_text": b.text}
        for b in blocks
    ], ensure_ascii=False)


def _parse_translation_response(raw: str, wanted: set[int]) -> dict[int, list[str]]:
    """Parse the structured translation protocol, with legacy text fallback."""
    if not raw:
        return {}
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(),
                  flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Compatibility with older/proxy models that ignore the JSON contract.
        return _parse_split_lines(raw, wanted)

    if not isinstance(payload, list):
        return {}
    result: dict[int, list[str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            slice_id = int(item.get("slice_id"))
        except (TypeError, ValueError):
            continue
        if slice_id not in wanted or slice_id in result:
            continue
        sentences = item.get("sentences")
        if not isinstance(sentences, list):
            continue
        cleaned = []
        for sentence in sentences:
            if not isinstance(sentence, str):
                continue
            sentence = _sanitize_symbols(sentence.strip())
            if sentence:
                cleaned.append(sentence)
        if cleaned:
            result[slice_id] = cleaned
    return result


def _translation_comparison_key(text: str) -> str:
    """Remove punctuation/spacing for conservative cross-slice comparison."""
    text = unicodedata.normalize("NFKC", text or "")
    return "".join(
        ch.casefold() for ch in text
        if unicodedata.category(ch)[0] in ("L", "N", "M")
    )


def _suspicious_translation_ids(blocks, per_slice: dict[int, list[str]]) -> set[int]:
    """Find adjacent translations likely contaminated across slice boundaries.

    A pair is only flagged when both Chinese results are substantial and quite
    similar while their source ASR texts are not.  The caller re-translates the
    flagged slices independently; nothing is deleted based on similarity alone,
    so genuine repeated dialogue remains intact when the verification agrees.
    """
    ordered = [b for b in blocks if b.index in per_slice]
    suspicious: set[int] = set()
    for previous, current in zip(ordered, ordered[1:]):
        left = _translation_comparison_key("".join(per_slice[previous.index]))
        right = _translation_comparison_key("".join(per_slice[current.index]))
        if min(len(left), len(right)) < 12:
            continue
        length_ratio = min(len(left), len(right)) / max(len(left), len(right))
        if length_ratio < 0.5:
            continue
        translated_similarity = SequenceMatcher(None, left, right).ratio()

        source_left = _translation_comparison_key(previous.text)
        source_right = _translation_comparison_key(current.text)
        source_similarity = SequenceMatcher(
            None, source_left, source_right
        ).ratio()
        if translated_similarity >= 0.55 and source_similarity < 0.45:
            suspicious.update((previous.index, current.index))
    return suspicious


def _parse_single_optimized_text(raw: str) -> str:
    """One request owns one fragment; tolerate plain text and legacy numbering."""
    text = re.sub(r"^\s*```(?:json|text)?\s*\n?|\n?```\s*$", "", raw.strip())
    if text.startswith("{"):
        try:
            value = json.loads(text)
            text = value.get("text", "") if isinstance(value, dict) else ""
        except ValueError:
            return ""
    if not isinstance(text, str):
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = re.compile(r"^\d+(?:\t|\.\s+|[、：:]\s*)(.+)$")
    # Multiple indexed records are not a valid single-fragment response.
    if len(lines) > 1 and any(numbered.match(line) for line in lines):
        return ""
    if len(lines) == 1:
        match = numbered.match(lines[0])
        if match:
            lines[0] = match.group(1)
    if any(re.match(r"^(?:以下是|以下为|分析[:：]|说明[:：])", line) for line in lines):
        return ""
    return _sanitize_symbols(_normalize_tags("\n".join(lines)))


def _format_rows(blocks) -> str:
    rows = []
    for b in blocks:
        speaker = f"\t{b.speaker}" if b.speaker else ""
        rows.append(f"{b.index}{speaker}\t{b.text}")
    return "\n".join(rows)


def _parse_lines(raw: str, wanted: set[int]) -> dict[int, str]:
    """Parse `index<TAB>content` output lines into {index: content}.

    Rows that cannot be parsed (or whose index is not wanted) are skipped so the
    caller keeps the original text for those blocks.
    """
    result: dict[int, str] = {}
    if not raw:
        return result
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
    fallback = re.compile(r"^(\d+)\s*[\.:、：]\s*(.+)$")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            idx_s, content = parts
        else:
            m = fallback.match(line)
            if not m:
                continue
            idx_s, content = m.group(1), m.group(2)
        try:
            idx = int(idx_s)
        except ValueError:
            continue
        content = _sanitize_symbols(_normalize_tags(content.strip()))
        if idx in wanted and content:
            result[idx] = content
    return result


def _parse_split_lines(raw: str, wanted_slices: set[int]) -> dict[int, list[str]]:
    """Parse ``sliceIdx.sentenceIdx<TAB>translation`` lines into {slice: [sentences]}.

    Lines with a dotted index map to ``(slice_idx, sentence_idx)``; a bare index
    maps to ``(slice_idx, 0)``. Rows that cannot be parsed, or whose slice index
    is not wanted, are skipped. Sentences are kept in ascending sentence order
    per slice.
    """
    result: dict[int, list[tuple[int, str]]] = {}
    if not raw:
        return {}
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
    # The content separator must be real whitespace — a bare dotted index like
    # "3.1" must not be re-read as index "3" + content "1".
    fallback = re.compile(r"^(\d+(?:\.\d+)?)\s+(.+)$")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            idx_s, content = parts
        else:
            m = fallback.match(line)
            if not m:
                continue
            idx_s, content = m.group(1), m.group(2)
        content = _sanitize_symbols(content.strip())
        if not content:
            continue
        if "." in idx_s:
            head, _, tail = idx_s.partition(".")
            try:
                slice_idx, sent_idx = int(head), int(tail)
            except ValueError:
                continue
        else:
            try:
                slice_idx = int(idx_s)
            except ValueError:
                continue
            sent_idx = 0
        if slice_idx not in wanted_slices:
            continue
        result.setdefault(slice_idx, []).append((sent_idx, content))
    return {idx: [c for _, c in sorted(sents)] for idx, sents in result.items()}


def _build_sentence_blocks(
    slice_blocks,
    per_slice: dict[int, list[str]],
    preserve_timeline=False,
    preserve_source_timeline=False,
):
    """Build sentence-level blocks by allocating each slice's time proportionally.

    For every slice, its ``[start_ms, end_ms]`` window is divided among its
    sentences by character count (the last sentence absorbs rounding remainder).
    Slices with no parsed Chinese sentences are DISCARDED — the raw (often
    Japanese) transcript is never kept as a fallback. The exception is a block
    already classified ``skip_tts`` by an upstream stage: it is preserved as an
    original-audio placeholder and is never sent to the LLM. Blocks are
    re-indexed globally, in order.

    With ``preserve_timeline``, return independent copies of each translated
    row on its parent's full timeline and carry its slice contract forward.
    The next alignment stage owns assigning real time boundaries. Missing
    translations remain original-audio placeholders in this mode.

    With ``preserve_source_timeline``, the previous detect stage already owns
    sentence boundaries. Multiple strings returned by the translator are joined
    back into that Japanese sentence's single timeline block instead of deriving
    new boundaries from Chinese punctuation.
    """
    from .vtt_parser import VTTBlock

    if preserve_source_timeline:
        from dataclasses import replace
        out = []
        for block in slice_blocks:
            sentences = per_slice.get(block.index, ()) if not block.skip_tts else ()
            if not sentences:
                out.append(replace(block, index=len(out), skip_tts=True))
            else:
                out.append(replace(
                    block,
                    index=len(out),
                    text="".join(sentences),
                    translation_available=True,
                ))
        return out

    if preserve_timeline:
        from dataclasses import replace
        out = []
        for block in slice_blocks:
            sentences = per_slice.get(block.index, ()) if not block.skip_tts else ()
            if not sentences:
                out.append(replace(block, index=len(out), skip_tts=True))
            else:
                for text in sentences:
                    out.append(replace(block, index=len(out), text=text, translation_available=True))
        return out

    out = []
    next_index = 0
    for b in slice_blocks:
        if b.skip_tts:
            # A previous, independent stage classified this slice as original-only.
            # Preserve its complete VAD window without exposing it to translation.
            out.append(VTTBlock(
                index=next_index, start_ms=b.start_ms, end_ms=b.end_ms,
                text=b.text, speaker=b.speaker,
                speaker_confidence=b.speaker_confidence, skip_tts=True,
            ))
            next_index += 1
            continue
        sentences = per_slice.get(b.index)
        if not sentences:
            # No Chinese translation for this slice — drop it outright rather
            # than keep the raw (often Japanese) transcript in the output.
            continue
        total_chars = sum(len(s) for s in sentences)
        if total_chars <= 0:
            continue
        span = b.end_ms - b.start_ms
        cursor = b.start_ms
        for i, s in enumerate(sentences):
            if i == len(sentences) - 1:
                end = b.end_ms
            else:
                end = min(b.end_ms, cursor + int(round(span * len(s) / total_chars)))
            out.append(VTTBlock(
                index=next_index, start_ms=cursor, end_ms=max(cursor, end),
                text=s, speaker=b.speaker,
                speaker_confidence=b.speaker_confidence,
            ))
            next_index += 1
            cursor = end
    return out


class LLMClient:
    """Minimal async client for Anthropic / OpenAI-compatible chat APIs."""

    def __init__(self, cfg: LLMConfig, cache_dir: Optional[str] = None):
        self.cfg = cfg
        self._network_policy = NetworkPolicy()
        self._result_cache = JsonResultCache(cache_dir) if cache_dir else None
        self._batch_errors: list[PipelineError] = []

    # ---- endpoint / request building ----

    def _endpoint(self) -> str:
        base = self.cfg.resolved_base()
        if self.cfg.provider == "anthropic":
            if base.endswith("/messages"):
                return base
            if base.endswith("/v1"):
                return base + "/messages"
            return base + "/v1/messages"
        # openai-compatible
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.provider == "anthropic":
            headers["x-api-key"] = self.cfg.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _openai_token_limit_key(self) -> str:
        """Return the Chat Completions token-limit field for this model.

        GPT-5 and o-series models reject the legacy ``max_tokens`` field. Keep
        using it for older and unknown OpenAI-compatible models so custom
        endpoints retain their existing request contract.
        """
        model = self.cfg.model.lower()
        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            return "max_completion_tokens"
        return "max_tokens"

    def _payload(self, system: str, user: str, max_tokens: Optional[int] = None) -> dict:
        token_limit = max_tokens or self.cfg.max_tokens
        payload: dict = {"model": self.cfg.model}
        if self.cfg.provider == "anthropic":
            payload["max_tokens"] = token_limit
            payload["system"] = system
            payload["messages"] = [{"role": "user", "content": user}]
            # DeepSeek's Anthropic-compatible endpoint enables thinking mode by
            # default; a long reasoning block can consume the whole max_tokens
            # budget so the final text block never appears. Disable thinking so
            # the entire budget goes to the actual answer. Inert on Anthropic's
            # own API (thinking is already off there).
            payload["thinking"] = {"type": "disabled"}
        else:
            payload[self._openai_token_limit_key()] = token_limit
            payload["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            payload["temperature"] = self.cfg.temperature
            if self._is_direct_deepseek():
                # DeepSeek defaults to thinking, which can exhaust the subtitle
                # budget before any answer appears. Keep the extension scoped
                # to its official endpoint so other compatible APIs accept it.
                payload["thinking"] = {"type": "disabled"}
        return payload

    def _is_direct_deepseek(self) -> bool:
        from urllib.parse import urlsplit

        return (
            self.cfg.provider == "openai"
            and urlsplit(self.cfg.resolved_base()).hostname == "api.deepseek.com"
        )

    def _is_direct_gemini(self) -> bool:
        from urllib.parse import urlsplit

        return (
            self.cfg.provider == "openai"
            and self.cfg.model.lower().startswith("gemini-")
            and urlsplit(self.cfg.resolved_base()).hostname == "generativelanguage.googleapis.com"
        )

    @staticmethod
    def _gemini_retry_delay(response, attempt: int) -> float:
        header = getattr(response, "headers", {}).get("retry-after", "")
        try:
            advised = max(0.0, float(header))
        except (TypeError, ValueError):
            advised = 0.0
        return min(60.0, max(2.0 ** (attempt + 1), advised) + random.uniform(0, 0.5))

    @staticmethod
    def _parse_response(provider: str, resp_json) -> str:
        if not isinstance(resp_json, dict):
            raise LLMError(f"LLM 返回了非 JSON 对象响应: {str(resp_json)[:200]}")
        if resp_json.get("error"):
            raise LLMError(f"LLM API 返回错误: {str(resp_json['error'])[:300]}")
        try:
            if provider == "anthropic":
                content = resp_json.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    # DeepSeek's Anthropic-compatible layer may put a thinking/
                    # reasoning block first; scan for the first text-bearing block.
                    for block in content:
                        if isinstance(block, dict):
                            if isinstance(block.get("text"), str) and block["text"].strip():
                                return block["text"]
                            if isinstance(block.get("content"), str) and block["content"].strip():
                                return block["content"]
                        elif isinstance(block, str) and block.strip():
                            return block
                # Some proxies return an OpenAI-shaped response even when asked
                # in Anthropic format — accept it as a fallback.
                choices = resp_json.get("choices") if isinstance(resp_json, dict) else None
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message", {})
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        return c
                    if isinstance(c, list) and c and isinstance(c[0], dict):
                        return c[0].get("text", "")
                raise LLMError(f"Anthropic 响应中没有可解析的文本: {str(resp_json)[:300]}")
            choices = resp_json["choices"]
            content = choices[0]["message"].get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                        return block["text"]
            finish_reason = choices[0].get("finish_reason")
            detail = "（输出 token 上限可能过低）" if finish_reason == "length" else ""
            raise LLMError(f"大模型未返回文本，finish_reason={finish_reason or '未知'}{detail}")
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"unexpected LLM response shape: {e}") from e

    def _request_key(self, system: str, user: str, max_tokens: Optional[int]) -> str:
        return cache_key({"settings": public_settings(asdict(self.cfg)),
                          "system": system, "user": user, "max_tokens": max_tokens})

    async def _post(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        key = self._request_key(system, user, max_tokens)
        if self._result_cache:
            cached = self._result_cache.load(key)
            if isinstance(cached, str) and cached.strip():
                print("       [cached-llm] 复用已完成请求")
                return cached
        try:
            result = await self._post_uncached(system, user, max_tokens)
            if self._result_cache:
                self._result_cache.save(key, result)
            return result
        except Exception as error:
            if self._result_cache:
                self._result_cache.fail(key, error)
            raise

    async def _post_uncached(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        url = self._endpoint()
        headers = self._headers()
        payload = self._payload(system, user, max_tokens)
        last_error: Optional[Exception] = None
        direct_gemini = self._is_direct_gemini()
        # Keep other OpenAI-compatible providers on the existing retry policy.
        gemini_limit = max(self.cfg.max_retries, 5) if direct_gemini else self.cfg.max_retries
        for attempt in range(gemini_limit + 1):
            try:
                async with async_client(client_factory=httpx.AsyncClient, network_policy=self._network_policy, timeout=self.cfg.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                gemini_transient = direct_gemini and response.status_code in (429, 503)
                retry_limit = gemini_limit if gemini_transient else self.cfg.max_retries
                if (response.status_code >= 500 or gemini_transient) and attempt < retry_limit:
                    last_error = self._http_error(response, url, attempt + 1)
                    delay = (self._gemini_retry_delay(response, attempt) if gemini_transient
                             else 2 ** attempt)
                    if gemini_transient:
                        print(f"       [llm-retry] Gemini HTTP {response.status_code}, "
                              f"{delay:.1f}s 后重试（{attempt + 1}/{gemini_limit}）")
                    await asyncio.sleep(delay)
                    continue
                if response.status_code != 200:
                    raise self._http_error(response, url, attempt + 1)
                response_data = response.json()
                try:
                    return self._parse_response(self.cfg.provider, response_data)
                except LLMError as exc:
                    if not str(exc).startswith("大模型未返回文本"):
                        raise
                    choices = response_data.get("choices") if isinstance(response_data, dict) else None
                    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
                    usage = response_data.get("usage") if isinstance(response_data, dict) else None
                    safe_usage = {
                        key: value for key, value in usage.items()
                        if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        and isinstance(value, int)
                    } if isinstance(usage, dict) else {}
                    print(f"       [llm-empty-response] model={self.cfg.model} "
                          f"attempt={attempt + 1} finish_reason={choice.get('finish_reason')!r} "
                          f"message_keys={sorted(message)} usage={safe_usage}")
                    if direct_gemini and attempt < self.cfg.max_retries:
                        last_error = exc
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            except LLMError:
                # 4xx / unparseable responses are deterministic — retrying won't help.
                raise
            except Exception as e:  # noqa: BLE001 — transport errors are retriable
                last_error = e
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break
        raise LLMError(
            f"LLM request failed after {self.cfg.max_retries + 1} tries: {last_error}"
        )

    def _http_error(self, response, url: str, attempt: int) -> LLMError:
        """Log bounded, redacted service diagnostics before constructing an error.

        Request headers and prompts are deliberately excluded. The pipeline tee
        persists this record, and the error reporter includes it without the
        500-character user-facing detail limit.
        """
        from urllib.parse import urlsplit, urlunsplit
        from .error import descriptor, safe_detail

        def redact(value, limit=32768):
            text = str(value or "")
            if self.cfg.api_key:
                text = text.replace(self.cfg.api_key, "[REDACTED]")
            return safe_detail(text, limit=limit)

        parsed_url = urlsplit(url)
        host = parsed_url.netloc.rsplit("@", 1)[-1]
        endpoint = urlunsplit((parsed_url.scheme, host, parsed_url.path, "", ""))
        body = response.text
        try:
            data = response.json()
        except (ValueError, TypeError):
            data = {}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        service_error = data.get("error", {}) if isinstance(data, dict) else {}
        service_error = service_error if isinstance(service_error, dict) else {}
        response_headers = getattr(response, "headers", {})
        request_ids = {key: redact(response_headers.get(key), 500)
                       for key in ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray")
                       if response_headers.get(key)}
        record = {
            "provider": self.cfg.provider, "model": redact(self.cfg.model, 500),
            "endpoint": redact(endpoint, 2000), "status": response.status_code,
            "attempt": attempt, "request_ids": request_ids,
            "error_type": redact(service_error.get("type"), 500),
            "error_code": redact(service_error.get("code"), 500),
            "error_param": redact(service_error.get("param"), 500),
            "error_message": redact(service_error.get("message"), 4000),
            "response_body": redact(body), "response_body_chars": len(body),
            "response_body_truncated": len(body) > 32768,
        }
        print("       [llm-http-error] " + json.dumps(record, ensure_ascii=False))
        detail = (f"{self.cfg.provider} LLM returned {response.status_code}: "
                  f"type={record['error_type']} code={record['error_code']} "
                  f"message={record['error_message']} body={record['response_body']}")
        if service_error.get("type") == "content_policy_violation" or service_error.get("code") == "content_policy_violation":
            error = LLMError(descriptor("LLM_CONTENT_REJECTED"), detail=detail)
        elif self._is_direct_gemini() and response.status_code == 429:
            error = LLMError(descriptor("LLM_GEMINI_RATE_LIMITED"), detail=detail)
        elif self._is_direct_gemini() and response.status_code == 503:
            error = LLMError(descriptor("LLM_SERVICE_BUSY"), detail=detail)
        else:
            error = LLMError(detail)
        error.context.update({"status": response.status_code, "backend": self.cfg.provider})
        if error.descriptor.code in ("LLM_CONTENT_REJECTED", "LLM_ACCESS_DENIED") and record["error_message"]:
            from dataclasses import replace
            error.descriptor = replace(
                error.descriptor,
                user_message=f"{error.descriptor.user_message} 服务返回：{redact(record['error_message'], 300)}",
            )
        return error

    async def check_connection(self) -> str:
        """Perform a real, minimal request through the production HTTP path."""
        return await self._post(
            "你是连接检测助手。",
            "只回复：好",
            max_tokens=256,
        )

    # ---- two-phase orchestration ----

    async def phase1_summarize(self, blocks) -> str:
        return await self._post(_PHASE1_SYSTEM, _format_rows(blocks))

    async def _run_batched(self, system: str, blocks, budget_per_row: int = 96,
                           formatter=None,
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
                raw = await self._post(system, user_text, max_tokens=budget)
                raw_texts.append(raw)
                wanted = {block.index for block in batch}
                parser = (_parse_translation_response if formatter is _format_translation_request
                          else _parse_lines)
                if self._result_cache and wanted - set(parser(raw, wanted)):
                    # A transport success with omitted/invalid rows is not a
                    # reusable completed request. Parsed valid rows survive in
                    # their own records; missing rows must get a new request.
                    from .error import descriptor
                    self._result_cache.fail(
                        self._request_key(system, user_text, budget),
                        LLMError(descriptor("LLM_RESPONSE_INVALID"),
                                 context={"missing_segments": sorted(wanted - set(parser(raw, wanted)))}),
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
            from .error import LlmError, descriptor
            raise LlmError(
                descriptor(incomplete_code),
                detail=f"{failed}/{len(batches)} LLM batches failed",
                context={"operation": "batch"},
            )
        return raw_texts, failed

    async def optimize_from_japanese(self, block, japanese: str) -> str:
        """Regenerate exactly one Chinese row from edited Japanese, every call fresh."""
        if not japanese.strip():
            raise ValueError("当前片段没有日语原文，请先补入日语字幕")
        system = build_phase2_system("", self.cfg.tts_backend, self.cfg.tts_model)
        system += "\n本次只处理一条日语原文。重新翻译成中文并优化。覆盖前面的批量输出格式：只输出优化后的纯中文文本，不要序号、制表符、代码块、解释，不要拆分或沿用旧译文。"
        return await self._optimize_single_text(system, japanese, "日语翻译成中文")

    async def optimize_chinese_text(self, block, chinese: str) -> str:
        if not chinese.strip():
            raise ValueError("中文字幕为空，无法优化")
        system = build_phase2_system("", self.cfg.tts_backend, self.cfg.tts_model)
        system += "\n本次只处理一条中文字幕。覆盖前面的批量输出格式：只输出优化后的中文纯文本，不要序号、制表符、代码块或解释，不要翻译成其他语言。"
        return await self._optimize_single_text(system, chinese, "优化中文文本")

    async def _optimize_single_text(self, system: str, text: str, operation: str) -> str:
        raw = await self._post(system, text, max_tokens=max(self.cfg.max_tokens, 512 + len(text) * 3))
        result = _parse_single_optimized_text(raw)
        if not result:
            from .error import LlmError, descriptor
            raise LlmError(descriptor("LLM_OPTIMIZE_INCOMPLETE"),
                           detail=f"{operation}未返回有效单段文本；响应片段：{raw[:800]!r}",
                           context={"operation": operation})
        return result

    async def optimize_japanese_text(self, block, japanese: str) -> str:
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
                    raw = await self._post(
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

    def _raise_incomplete(self, keys: dict[int, str], missing: set[int],
                          code: str, errors: list[PipelineError]) -> NoReturn:
        from .error import LlmError, descriptor, safe_detail
        error = errors[0] if errors else LlmError(
            descriptor(code), detail=f"missing segments after retries: {sorted(missing)}")
        label = "missing optimized rows after retries" if code == "LLM_OPTIMIZE_INCOMPLETE" else "missing slices after retries"
        error.detail = safe_detail(f"{label}: {sorted(missing)}; {error.detail}")
        error.context.update({"missing_segments": sorted(missing)})
        if self._result_cache:
            for idx in missing:
                self._result_cache.fail(keys[idx], error)
        raise error

    async def phase2_optimize(self, blocks, summary: str) -> dict[int, str]:
        system = build_phase2_system(
            summary, tts_backend=self.cfg.tts_backend,
            tts_model=self.cfg.tts_model)
        wanted = {b.index for b in blocks}
        keys = {b.index: cache_key({"operation": "optimize-row", "text": b.text,
                                   "index": b.index, "settings": public_settings(asdict(self.cfg))})
                for b in blocks}
        optimized = {}
        if self._result_cache:
            optimized = {b.index: value for b in blocks
                         if isinstance(value := self._result_cache.load(keys[b.index]), str) and value.strip()}
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
        if self._result_cache:
            for idx, text in optimized.items():
                self._result_cache.save(keys[idx], text)
        if missing:
            errors = [self._recovery_errors[idx] for idx in sorted(missing)
                      if idx in self._recovery_errors]
            self._raise_incomplete(keys, missing, "LLM_OPTIMIZE_INCOMPLETE",
                                   errors or self._batch_errors)
        self._failed_batches = 0
        return optimized

    async def _translate_split_once(self, blocks, summary, wanted) -> dict[int, list[str]]:
        """POST the translate-split prompt for ``blocks`` and parse the replies.

        Returns ``{slice_idx: [sentences]}``; rows whose slice is not in ``wanted``
        or whose content is empty are dropped.
        """
        # Use replacement instead of str.format because the prompt contains a
        # literal JSON example with braces.
        system = _TRANSLATE_SPLIT_SYSTEM.replace("{summary}", summary)
        raw_texts, _ = await self._run_batched(
            system, blocks, formatter=_format_translation_request,
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
        raw = await self._post(
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

    async def translate_split(self, blocks, summary: str) -> dict[int, list[str]]:
        """Translate each ASR slice into Chinese sentences (``{slice: [sentences]}``).

        The first pass batches all slices with a numbered output format. Slices
        that come back with no sentences are retried one-by-one with a
        numbering-free prompt, so a missed slice gets a real second chance and
        cannot be lost to a misattributed/renumbered row. Slices still empty
        after ``cfg.max_retries`` retries are logged with a visible warning.
        """
        wanted = {b.index for b in blocks}
        keys = {b.index: cache_key({"operation": "translate-slice", "text": b.text,
                                   "index": b.index, "settings": public_settings(asdict(self.cfg))})
                for b in blocks}
        per_slice = {}
        if self._result_cache:
            per_slice = {b.index: value for b in blocks
                         if isinstance(value := self._result_cache.load(keys[b.index]), list)
                         and value and all(isinstance(s, str) and s.strip() for s in value)}
        pending = [b for b in blocks if b.index not in per_slice]
        per_slice.update(await self._translate_split_once(pending, summary, wanted - set(per_slice)))
        if self._result_cache:
            for idx, sentences in per_slice.items():
                self._result_cache.save(keys[idx], sentences)
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
                    if self._result_cache:
                        self._result_cache.save(keys[idx], sentences)
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
                    if self._result_cache:
                        self._result_cache.save(keys[idx], verified)
        return per_slice

    async def enhance(self, blocks) -> dict:
        """Run both phases and rewrite block.text in place, or fail atomically."""
        print(f"       [llm] 大模型字幕优化已启用 ({self.cfg.provider}: {self.cfg.model})")
        summary_error = None
        try:
            summary = await self.phase1_summarize(blocks)
        except LLMError as error:
            summary_error = error
            print(f"       [llm-summary-failed] 摘要不可用，继续逐片处理: {error}")
            summary = ""
        print("       [llm] 整体情节分析完成")
        print(f"       [llm-summary] 剧情摘要：{summary}")
        try:
            optimized = await self.phase2_optimize(blocks, summary)
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
        failed = getattr(self, "_failed_batches", 0)
        msg = f"       [llm] 优化字幕：{rewritten}/{len(blocks)} 段已改写"
        if failed:
            msg += f"（{failed} 批失败已保留原文）"
        print(msg)
        return {"enabled": True, "rewritten": rewritten, "total": len(blocks),
                "failed_batches": failed, "summary_ok": summary_error is None}

    async def enhance_asr(
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
            summary = await self.phase1_summarize(translatable)
        except LLMError as error:
            summary_error = error
            print(f"       [llm-summary-failed] 摘要不可用，继续逐片处理: {error}")
            summary = ""
        print("       [llm] 整体情节分析完成")
        print(f"       [llm-summary] 剧情摘要：{summary}")
        try:
            per_slice = await self.translate_split(translatable, summary)
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
        failed = getattr(self, "_failed_batches", 0)
        if self.cfg.optimize_enabled:
            polishable = [block for block in blocks if not block.skip_tts]
            optimized = await self.phase2_optimize(polishable, summary)
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


def enhance_blocks(blocks, llm_kwargs: Optional[dict] = None, *, output_dir: Optional[str] = None) -> dict:
    """Sync entry point — rewrites `blocks` in place via the configured LLM.

    Returns a stats dict; returns ``{"enabled": False}`` when not configured or
    incomplete, in which case no network request is made.
    """
    cfg = LLMConfig(**(llm_kwargs or {}))
    if not is_configured(cfg) or not cfg.optimize_enabled:
        return {"enabled": False, "rewritten": 0, "total": len(blocks),
                "failed_batches": 0, "summary_ok": False}
    client = LLMClient(cfg, os.path.join(output_dir, "llm_results") if output_dir else None)
    return asyncio.run(client.enhance(blocks))


def enhance_asr_blocks(
    slice_blocks,
    llm_kwargs: Optional[dict] = None,
    *,
    preserve_timeline=False,
    preserve_source_timeline=False,
    output_dir: Optional[str] = None,
):
    """Sync entry point — translate+split ASR slices into sentence-level blocks.

    Returns ``(blocks, stats)``; when not configured or incomplete, returns the
    original slice blocks unchanged with ``{"enabled": False}``.
    """
    cfg = LLMConfig(**(llm_kwargs or {}))
    if not is_configured(cfg):
        return list(slice_blocks), {"enabled": False, "translated": 0,
                                    "rewritten": 0, "total": len(slice_blocks),
                                    "summary_ok": False}
    client = LLMClient(cfg, os.path.join(output_dir, "llm_results") if output_dir else None)
    return asyncio.run(client.enhance_asr(
        slice_blocks,
        preserve_timeline,
        preserve_source_timeline,
    ))
