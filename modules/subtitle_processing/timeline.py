"""Pure subtitle time allocation; language models do not own audio timestamps."""

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
    from ..vtt_parser import VTTBlock

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
