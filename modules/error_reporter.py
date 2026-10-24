"""Best-effort asynchronous uploads for backend error diagnostics.

Reporting is deliberately isolated from the pipeline error contract: a broken or
retired collection endpoint must never replace, delay the emission of, or add
details to the error shown to the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
from typing import Mapping, Protocol
from urllib.request import Request, urlopen

from .app_version import resolve_app_version
from .error import PipelineError

DEFAULT_LOG_UPLOAD_URL = ""
_MAX_PIPELINE_LOG_CHARS = 750_000
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(authorization|api[-_ ]?key|token|secret|password)"
        r"(\s*[:=]\s*)([^\s,;&\"'}]+)"
    ),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)([?&](?:key|api_key|token|secret)=)[^&#\s]+"),
)


class LogUploader(Protocol):
    """Transport boundary used by the error reporting worker."""

    def upload(self, report: str) -> None: ...


@dataclass(frozen=True)
class HttpLogUploader:
    endpoint: str
    timeout_seconds: float = 2.5

    def upload(self, report: str) -> None:
        headers = {"Content-Type": "text/plain; charset=utf-8"}
        request = Request(
            self.endpoint,
            data=report.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"log endpoint returned HTTP {response.status}")


def redact_log_text(value: str) -> str:
    """Remove common credential shapes without truncating diagnostic logs."""
    text = value
    text = _SECRET_PATTERNS[1].sub("Bearer [REDACTED]", text)
    text = _SECRET_PATTERNS[0].sub(r"\1\2[REDACTED]", text)
    return _SECRET_PATTERNS[2].sub(r"\1[REDACTED]", text)


def build_error_report(
    error: PipelineError,
    traceback_text: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Create the UTF-8 text accepted by the server's ``POST /log`` route."""
    env = os.environ if environ is None else environ
    payload = error.to_payload()
    header = {
        "reportedAt": datetime.now(timezone.utc).isoformat(),
        "application": "asmr-translator",
        "appVersion": resolve_app_version(env),
        "error": payload,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packaged": _is_packaged(),
        },
        "task": {
            "assetId": env.get("ASMR_ASSET_ID", ""),
            "exportName": env.get("ASMR_EXPORT_NAME", ""),
        },
    }
    sections = [json.dumps(header, ensure_ascii=False, indent=2)]
    if traceback_text.strip():
        sections.extend(("\n--- traceback ---", traceback_text.rstrip()))

    log_path = getattr(error, "log_path", "")
    if log_path:
        try:
            pipeline_log = Path(log_path).read_text(encoding="utf-8", errors="replace")
            if len(pipeline_log) > _MAX_PIPELINE_LOG_CHARS:
                pipeline_log = pipeline_log[-_MAX_PIPELINE_LOG_CHARS:]
                pipeline_log = "[earlier pipeline log omitted]\n" + pipeline_log
            sections.extend((f"\n--- pipeline log: {log_path} ---", pipeline_log.rstrip()))
        except OSError as read_error:
            sections.extend(("\n--- pipeline log unavailable ---", str(read_error)))
    return redact_log_text("\n".join(sections))


def _build_log_defaults() -> dict[str, str]:
    """Read optional values embedded by CI, never a user configuration file."""
    if not _is_packaged():
        return {}
    try:
        payload = json.loads(Path(__file__).with_name("_log_upload_defaults.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items()
            if key == "ASMR_LOG_UPLOAD_URL" and isinstance(value, str)}


def report_error_async(
    error: PipelineError,
    traceback_text: str,
    *,
    uploader: LogUploader | None = None,
    environ: Mapping[str, str] | None = None,
) -> threading.Thread | None:
    """Start a bounded best-effort upload after the user error has been emitted.

    The non-daemon worker is allowed to finish during interpreter shutdown so an
    error report is not silently discarded. The HTTP transport itself has a short
    timeout, while the already-emitted frontend error remains immediately visible.
    """
    env = os.environ if environ is None else environ
    if env.get("ASMR_DISABLE_LOG_UPLOAD", "").strip().lower() in {"1", "true", "yes"}:
        return None
    defaults = _build_log_defaults()
    endpoint = env.get("ASMR_LOG_UPLOAD_URL", defaults.get("ASMR_LOG_UPLOAD_URL", DEFAULT_LOG_UPLOAD_URL)).strip()
    if not endpoint and uploader is None:
        return None
    def _worker() -> None:
        try:
            selected = uploader or HttpLogUploader(
                endpoint=endpoint,
            )
            report = build_error_report(error, traceback_text, environ=env)
            selected.upload(report)
        except Exception as upload_error:  # reporting must never affect users
            if _is_development(env):
                print(f"[log-upload] {upload_error}", file=sys.stderr, flush=True)

    try:
        worker = threading.Thread(target=_worker, name="error-log-upload", daemon=False)
        worker.start()
        return worker
    except Exception as start_error:  # even thread creation is best-effort
        if _is_development(env):
            print(f"[log-upload] {start_error}", file=sys.stderr, flush=True)
        return None


def _is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False) or "__compiled__" in globals())


def _is_development(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    explicit = env.get("ASMR_ENV", "").strip().lower()
    if explicit:
        return explicit in {"dev", "development", "local"}
    return not _is_packaged()
