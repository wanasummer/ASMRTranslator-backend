"""Clean-room packaged sidecar E2E test suite.

All `test_*.py` files in this package are automatically discovered and executed:
- Against the compiled Nuitka `resources/main.exe` when `ASMR_PACKAGED_RESOURCES`
  points to an installed or staged `resources/` directory (e.g. in GitHub Actions
  after silently installing the NSIS installer).
- Against `main.py` in a clean-room subprocess when run locally during development.
"""
