"""Speech geometry at the ASR output boundary, independent of decode padding."""

from .models import SpeechRegion


def speech_bounds_ms(region: SpeechRegion) -> tuple[int, int]:
    """Prefer explicit speech contracts; legacy unpadded regions keep their span.

    Never subtract a fixed padding amount: clipping at zero and overlapping
    windows mean the actual context on either side need not be symmetric.
    """
    if region.asr_slice is not None:
        return region.asr_slice.speech_start_ms, region.asr_slice.speech_end_ms
    return (region.speech_start_ms if region.speech_start_ms is not None else round(region.start * 1000),
            region.speech_end_ms if region.speech_end_ms is not None else round(region.end * 1000))
