"""Pure speech-text normalization; independent of LLM and TTS adapters."""
import re
import unicodedata

_TAG_RE = re.compile(r"\[[A-Za-z]+\]")


_SPACE_RE = re.compile(r"\s+")


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
