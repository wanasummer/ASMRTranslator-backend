"""Single runtime source for the application version."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from typing import Mapping

_DISTRIBUTION_NAME = "asmr-translator"
_FALLBACK_VERSION = "1.5.7"


def resolve_app_version(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the version for source, installed, and packaged runtimes."""
    env = os.environ if environ is None else environ
    override = env.get("ASMR_APP_VERSION", "").strip()
    if override:
        return override
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION


APP_VERSION = resolve_app_version()
