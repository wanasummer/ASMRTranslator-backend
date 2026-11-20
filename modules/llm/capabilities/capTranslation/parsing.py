"""Translation ownership parsing and cross-slice verification policy."""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from ....text_normalization import _sanitize_symbols

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
