"""Single-fragment response parsing without I/O."""
import json
import re
from ..text_normalization import _sanitize_symbols, _normalize_tags

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
