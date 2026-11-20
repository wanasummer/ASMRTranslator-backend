"""Chinese subtitle row parsing."""
import re
from ....text_normalization import _sanitize_symbols, _normalize_tags

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
