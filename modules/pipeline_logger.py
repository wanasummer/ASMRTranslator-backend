"""Logging utilities for the ASMR dubbing pipeline.

Provides:
- PipelineTee: duplicate stdout writes to both the original stream and a log file.
- print_progress: structured progress reporting for the Electron frontend.
"""

import io
import json
import sys
import threading


class PipelineTee(io.TextIOBase):
    """Duplicate writes to both an original stream and a log file.

    Usage:
        log_fp = open("pipeline.log", "w")
        sys.stdout = PipelineTee(sys.stdout, log_fp)
        # ... all prints go to both stdout and the log file ...
        sys.stdout = sys.stdout.close()
    """

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file
        self._original_available = True
        self._write_lock = threading.RLock()

    def write(self, data):
        # TextIOWrapper is not safe for concurrent writes. Audio build stages
        # log from worker threads, so serialize both destinations to prevent
        # buffer corruption and invalid UTF-8 reaching the desktop pipe.
        with self._write_lock:
            self._log.write(data)
            if self._original_available:
                try:
                    self._original.write(data)
                except (OSError, ValueError):
                    # The desktop host owns this pipe and may stop reading it. A
                    # failed UI transport must not fail the audio pipeline.
                    self._original_available = False
        return len(data)

    def flush(self):
        with self._write_lock:
            self._log.flush()
            if self._original_available:
                try:
                    self._original.flush()
                except (OSError, ValueError):
                    self._original_available = False

    def close(self):
        """Restore the original stream and close the log file."""
        with self._write_lock:
            self._log.close()
        return self._original


# ---- Structured progress reporting ----
# Uses a unique prefix so the Electron host can detect and forward as IPC events.
_PROGRESS_LOCK = threading.Lock()


def print_progress(stage: str, label: str, current: int, total: int):
    """Print a machine-readable progress line for the Electron frontend.

    The [PROGRESS] prefix is detected in electron/main.js and forwarded
    as a structured IPC event, while being filtered from the log display.

    Args:
        stage: Short identifier for the pipeline stage (e.g. "parsing", "tts").
        label: Human-readable label shown in the UI (e.g. "正在生成翻译片段").
        current: Current progress count.
        total: Total expected count.
    """
    msg = json.dumps({
        "stage": stage,
        "label": label,
        "current": current,
        "total": total,
    }, ensure_ascii=False)
    with _PROGRESS_LOCK:
        print(f"[PROGRESS]{msg}")
        sys.stdout.flush()
