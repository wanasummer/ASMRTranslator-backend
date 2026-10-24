"""Prepare CI log defaults for embedding inside the executable; values stay off stdout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.io import write_json


def prepare_log_defaults(output: Path, environ: Mapping[str, str]) -> Path:
    values = {"ASMR_LOG_UPLOAD_URL": environ.get("ASMR_LOG_UPLOAD_URL", "").strip()}
    if any("\r" in value or "\n" in value for value in values.values()):
        raise ValueError("Log build variables must be single-line values")
    endpoint = values["ASMR_LOG_UPLOAD_URL"]
    if endpoint:
        try:
            url = urlsplit(endpoint)
            valid = url.scheme in {"http", "https"} and bool(url.hostname) and url.username is None and url.password is None
            url.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Invalid log upload endpoint")
    required = environ.get("ASMR_REQUIRE_LOG_UPLOAD", "").strip().lower() in {"1", "true", "yes"}
    disabled = environ.get("ASMR_DISABLE_LOG_UPLOAD", "").strip().lower() in {"1", "true", "yes"}
    if required and (not endpoint or disabled):
        raise ValueError("Release build requires an enabled log upload endpoint")
    write_json(str(output), values)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = prepare_log_defaults(args.output, os.environ)
    except (OSError, UnicodeError, ValueError):
        print("Cannot prepare log defaults; check build environment variables", file=sys.stderr)
        return 1
    print(f"Log build defaults prepared: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
