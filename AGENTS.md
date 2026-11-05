# Backend Agent Guide

## Scope

This file governs the Python backend rooted at `main.py`, including `modules/`,
`tests/`, backend dependencies, and Python sidecar packaging. Frontend code under
`desktop-tauri/src/` follows its own framework conventions.

## Commit Rules
- follow Conventional Commits Rules

## Architecture Rules

- Treat every pipeline stage as a replaceable module with an explicit input and
  return value. A stage may consume only the previous stage's returned contract,
  configuration passed to it, and injected adapters.
- Keep orchestration in `modules/pipeline.py`; keep domain behavior out of it.
  Orchestration selects implementations, passes outputs forward, logs summaries,
  and handles top-level degradation only.
- Define extension points with an abstract base class or Protocol, a stable data
  contract, and a factory/registry. Adding an implementation must not require a
  conditional chain in `run_pipeline`.
- Separate transformation from persistence. Transformers return structured values;
  dedicated writers persist artifacts and audit data.
- Keep dependency direction aligned with pipeline order. Upstream modules must not
  import or invoke downstream modules.
- Prefer returning new values over mutating caller-owned collections. When legacy
  mutable block models are unavoidable, copy blocks at a stage boundary and state
  mutation behavior in the contract.
- Preserve stable identifiers, ordering, and metadata across stage boundaries unless
  a stage contract explicitly owns changing them.
- Keep Qwen Force Align and the FSMN/MeCab joint detector as separate intermediate
  pipelines, selected by the detect-stage factory. Qwen owns its configuration and
  word-timestamp sentence policy; FSMN owns speech atoms, mora estimates, and joint
  partitioning. Both consume `DetectStageInput` and return `DetectStageResult` to
  the common cleanup/translation/export flow. Do not route Qwen through the FSMN
  aligner registry or pass FSMN sentence evidence into Qwen's partitioner.
- Include the selected intermediate pipeline's effective settings and policy
  version in its cache identity when changing its behavior.
- MOSS consumes project VAD slices sequentially and returns native ASR blocks.
  Its dedicated `moss-native` detect stage refines only within each native block,
  preserving outer timestamps, VAD lineage, overlapping speakers, and exact text.
  Punctuation owns candidate boundaries; Qwen word alignment supplies internal
  timestamps only when splitting is needed. Do not route MOSS through the Qwen
  sentence pipeline, FSMN/MeCab, or generic Japanese cleanup. Alignment failure
  explicitly retains the native block. Keep raw ASR and detected output separate
  in the project database and include effective MOSS policy in cache identity.

## Code Conventions

- Keep reusable LLM transport/providers in `modules/llm/`, with one capability
  per `capabilities/capXxx/` directory and registration in `llm/factory.py`.
  Capabilities consume typed text rows and an injected text-generation port;
  they must not import pipeline, audio, or subtitle domain modules. Subtitle
  timing and business orchestration belong in `modules/subtitle_processing/`.
- Keep subtitle and text cleaning policies in `modules/cleaning/`, with one
  independently toggleable capability per `capabilities/capXxx/` directory
  (`capJapaneseClean`, `capInterjectionSkip`) backed by its own `Config`.

- Keep public contracts typed and explicit. Use dataclasses, TypedDicts, Protocols,
  or abstract base classes where they make boundaries clearer.
- Keep functions focused. Extract reusable policy and transformation logic from
  orchestration and I/O code.
- Prefer dependency injection for network clients, filesystem adapters, clocks, and
  other external dependencies so modules remain independently testable.
- Create service HTTP clients through `modules/network.py`. Keep proxy discovery,
  direct-first routing, and connection fallback in that shared layer; adapters
  only supply request data and an instance-owned policy for route reuse.
- Avoid hidden global state. Pass configuration explicitly and validate it at the
  boundary where it enters the system.
- Preserve backward compatibility for serialized contracts unless a migration is
  provided and tested.
- Use clear domain-specific exceptions internally instead of leaking transport or
  library exceptions across module boundaries.

## Diagnostics and Failure Behavior

- Persist project-generated non-audio data in one project SQLite database using
  `modules/storage.py` and the shared `storage_schema.sql`. Audio remains on disk.
  Do not introduce JSON/VTT/log sidecars, legacy-file fallbacks, or SQL in pipeline
  orchestration. Include the SQL resource explicitly in the Nuitka package.

- Save raw inputs before transforming them. Save cleaned/optimized artifacts and
  structured audit documents in the same project database when a transformation
  can discard information.
- Log counts, stage names, and artifact paths; never log secrets or credentials.
- Make fallback and degradation behavior explicit rather than silently changing a
  stage's contract.
- Raise typed domain errors inside adapters. Convert them to user-facing pipeline
  errors only at orchestration boundaries.
- Preserve UTF-8 text in structured and text artifacts and use shared I/O helpers.

## Dependencies and Packaging

- Runtime dependencies belong in `pyproject.toml`; regenerate `uv.lock` with
  `uv lock` and verify with `uv sync --frozen`.
- A package containing runtime data needs a corresponding Nuitka inclusion in
  `scripts/build-nuitka.js` or `nuitka-package.config.yml`.
- Development and packaged backends must resolve runtime data and executables in
  the same explicit way. Do not depend on undeclared machine-global resources.
- Keep Python compatible with the version range declared in `pyproject.toml`.

## Tests and Verification

- Put pure module tests in a matching `tests/test_<module>.py` file.
- Test contracts and stage boundaries in addition to happy-path text examples.
- Network calls, external APIs, subprocesses, and native integrations must be mocked
  in unit tests. Bundled deterministic runtime data may be exercised locally.
- Run targeted tests while iterating, then run the full backend suite before handoff:

  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests
  ```

- Also run `python -m py_compile` for new modules and `git diff --check`.

## Adding a New Stage or Implementation

1. Define the stage's input/output contract without referencing the orchestrator.
2. Define an ABC or Protocol for interchangeable implementations.
3. Implement one concrete adapter and register it in a factory.
4. Add an independent artifact writer if diagnostics are needed.
5. Wire exactly one factory call and one returned value into `pipeline.py`.
6. Add unit tests for the implementation and an integration test proving the next
   stage receives only the returned output.
7. Update this guide only when code conventions, architectural boundaries, or
   verification requirements change.
