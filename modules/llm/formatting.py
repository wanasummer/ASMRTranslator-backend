"""Pure source-row formatting shared by summary and polishing."""

def _format_rows(blocks) -> str:
    rows = []
    for b in blocks:
        speaker = f"\t{b.speaker}" if b.speaker else ""
        rows.append(f"{b.index}{speaker}\t{b.text}")
    return "\n".join(rows)
