"""Tests for main.py CLI argument contracts and frontend/factory parity."""

import io
import os
import re
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

MODELS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODELS_ROOT.parent
sys.path.insert(0, str(MODELS_ROOT))

# CLI import loads the user dotenv; keep it out of other tests.
with mock.patch.dict(os.environ):
    import main as cli_main
from modules.asr.factory import get_available_asr_backends


def _parse_ts_union(ts_text: str, type_name: str) -> list[str]:
    match = re.search(rf"export\s+type\s+{re.escape(type_name)}\s*=\s*([^;]+);", ts_text)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


class MainCliTests(unittest.TestCase):
    def test_asr_backend_accepts_all_registered_backends_including_deepseek(self) -> None:
        parser = cli_main.build_parser()
        backends = get_available_asr_backends()
        self.assertIn("deepseek-asr", backends)

        for backend in backends:
            with self.subTest(backend=backend):
                args = parser.parse_args(["audio.wav", "--asr-backend", backend])
                self.assertEqual(args.asr_backend, backend)

    def test_asr_backend_rejects_unknown_backend(self) -> None:
        parser = cli_main.build_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                parser.parse_args(["audio.wav", "--asr-backend", "nonexistent-asr"])
        self.assertEqual(ctx.exception.code, 2)

    def test_desktop_ts_cli_contracts_match_main_parser(self) -> None:
        desktop_ts = REPO_ROOT / "desktop-tauri" / "src" / "lib" / "types" / "desktop.ts"
        if not desktop_ts.is_file():
            self.skipTest("desktop-tauri/src/lib/types/desktop.ts not present")

        ts_text = desktop_ts.read_text(encoding="utf-8")
        parser = cli_main.build_parser()

        asr_engines = _parse_ts_union(ts_text, "AsrEngine")
        self.assertTrue(asr_engines)
        for engine in asr_engines:
            with self.subTest(asr_engine=engine):
                args = parser.parse_args(["audio.wav", "--asr-backend", engine])
                self.assertEqual(args.asr_backend, engine)

        tts_backends = _parse_ts_union(ts_text, "TtsBackend")
        self.assertTrue(tts_backends)
        for backend in tts_backends:
            with self.subTest(tts_backend=backend):
                args = parser.parse_args(["audio.wav", "--tts-backend", backend])
                self.assertEqual(args.tts_backend, backend)

        sync_modes = _parse_ts_union(ts_text, "SyncMode")
        self.assertTrue(sync_modes)
        for mode in sync_modes:
            with self.subTest(sync_mode=mode):
                args = parser.parse_args(["audio.wav", "--sync-mode", mode])
                self.assertEqual(args.sync_mode, mode)


if __name__ == "__main__":
    unittest.main()
