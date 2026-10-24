"""Machine-wide resource budgets for one pipeline run.

The pipeline uses one pool per processing stage. Values in this module are
therefore global per process, never multiplied by a chunk count.
"""

from __future__ import annotations

import ctypes
import os


def cpu_count() -> int:
    """Return the number of logical processors available to this process."""
    return os.cpu_count() or 4


def _windows_memory() -> tuple[int | None, int | None]:
    """Return (total, available) physical memory in bytes on Windows."""
    if os.name != "nt":
        return None, None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except (AttributeError, OSError):
        pass
    return None, None


def _positive(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


class PoolConfig:
    """Global resource budget for a pipeline process.

    Explicit positive values override automatic sizing. Zero/``None`` means
    automatic. ``requested_chunks`` remains accepted for CLI compatibility,
    but the optimized pipeline is staged globally and no longer creates nested
    per-chunk pools.
    """

    def __init__(
        self,
        requested_chunks: int = 0,
        split_workers: int | None = None,
        build_workers: int | None = None,
        tts_concurrency: int | None = None,
        tts_backend: str = "edge-tts",
    ):
        del requested_chunks
        logical = cpu_count()
        total_mem, available_mem = _windows_memory()
        try:
            active_pipelines = max(1, int(os.environ.get("ASMR_ACTIVE_PIPELINES", "1") or 1))
        except ValueError:
            active_pipelines = 1

        # The source is normalized by one ffmpeg process, then sliced from PCM.
        # A former CLI option may still provide a value; it is intentionally no
        # longer allowed to create parallel decoder processes.
        del split_workers
        self.split_workers = 1

        cpu_budget = max(1, min(8, logical // 2))
        if available_mem is None:
            memory_budget = 4
        else:
            available_gib = available_mem / (1024 ** 3)
            if available_gib < 2.5:
                memory_budget = 2
            elif available_gib < 4:
                memory_budget = 3
            elif available_gib < 8:
                memory_budget = 4
            else:
                memory_budget = 8
        auto_build = max(1, min(cpu_budget, memory_budget) // active_pipelines)
        self.build_workers = _positive(build_workers) or auto_build

        backend_caps = {
            "indextts-2.5": 1,
            "fish-audio": 6,
            "minimax-tts": 8,
            "edge-tts": 8,
        }
        auto_tts = max(1, backend_caps.get(tts_backend, 6) // active_pipelines)
        requested_tts = _positive(tts_concurrency)
        self.tts_concurrency = min(
            requested_tts or auto_tts, backend_caps.get(tts_backend, 8)
        )

        self.chunk_workers = 1
        self.logical_cpus = logical
        self.total_memory = total_mem
        self.available_memory = available_mem
        self.active_pipelines = active_pipelines

    @staticmethod
    def _gib(value: int | None) -> str:
        return "unknown" if value is None else f"{value / (1024 ** 3):.1f}GiB"

    def __repr__(self) -> str:
        return (
            "PoolConfig(stage=global, split=1, "
            f"build={self.build_workers}, tts={self.tts_concurrency}, "
            f"logical_cpu={self.logical_cpus}, "
            f"memory={self._gib(self.available_memory)}/{self._gib(self.total_memory)}, "
            f"active_tasks={self.active_pipelines})"
        )
