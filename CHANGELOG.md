# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning. While the project is pre-1.0, minor
versions may include API changes as the experiment framework settles.

## [0.9.0] - 2026-05-21

### Added

- Added folder-backed JSON/YAML scenario datasets, JSONL datasets, CSV
  `scenario_path`, and CSV `turns_json` expansion for multi-turn eval inputs.
- Added named model variants for multi-agent workflows, with `ctx.model(role)`
  access to role-specific `ModelConfig` entries.
- Added conversation turn recording helpers, seeded user/assistant/action turns,
  tool-call recording, `turns.csv`, `tool_calls.csv`, and tool-call built-in
  evaluators.

## [0.8.0] - 2026-05-11

### Added

- Added first-class Realtime workflow support through `ctx.realtime`, including
  text and audio helpers for `gpt-realtime-2`.
- Parsed Realtime tool calls into `RealtimeRunResult.tool_calls` and
  `TaskOutput.value["tool_calls"]` for scoring.
- Added Realtime text and voice-agent smoke examples, plus viewer playback for
  audio media.

## [0.7.0] - 2026-05-04

### Changed

- `TaskOutput` is now the required workflow and step output contract.
- Workflows should import provider SDKs directly instead of relying on Prism to
  proxy OpenAI calls.

### Added

- Added `TaskOutput.media`, `MediaArtifact`, run-local `media/` storage,
  compact media columns in CSV outputs, and viewer media previews.
- Added `ctx.media.from_base64(...)`, `ctx.media.from_bytes(...)`, and
  `ctx.media.from_path(...)` helpers for generated outputs.

## [0.6.8] - 2026-04-16

### Added

- `prism run <experiment_file>` discovers and runs module-level `Experiment`
  instances, so eval files no longer need an explicit `exp.run()` block.

## [0.6.7] - 2026-04-16

### Added

- Compact inline `data:` URLs in captured raw payloads by default with
  `redact_raw_data_urls=True`.

## [0.6.0] - 2026-04-16

### Changed

- Rebranded the package to Prism Evals.
- Renamed the Python distribution to `prism-evals` and the import package to
  `prism_evals`.
- Added `prism`, `prism-evals`, and `pe` console scripts.
- Updated the local viewer, scaffolded instructions, docs, examples, and tests
  to use Prism Evals naming.

## [0.5.1] - 2026-04-15

### Fixed

- Bundle the local viewer with Python wheels so installed packages can launch `prism view`.
- Install viewer npm dependencies automatically on first launch when they are missing.

## [0.5.0] - 2026-04-15

### Added

- `prism view <runs_dir>` opens a read-only local Next.js viewer for parent directories of eval runs.
- Viewer pages for all runs, run detail, score matrices, artifact downloads, and lane comparisons across `run + model_key` pairs.

## [0.4.5] - 2026-04-15

### Added

- `Experiment(..., artifacts=[...])` can copy prompt files, configs, and other user files into the run output folder.

## [0.4.4] - 2026-04-15

### Added

- Console output now includes a model-column score pivot table by eval key.

## [0.4.3] - 2026-04-15

### Added

- Console token usage now shows average per item run and totals for input, cached, output, reasoning, and total tokens.

## [0.4.2] - 2026-04-15

### Fixed

- Declared the `src/prism_evals` package for Hatchling wheel builds so `prism-evals` installs correctly from Git tags.

## [0.4.1] - 2026-04-15

### Added

- Timestamp-prefixed output directories by default, with `timestamp_output_dir=False` to keep stable experiment folders.

## [0.4.0] - 2026-04-15

### Added

- Multi-step workflows via `ctx.step(...)`, with step-owned outputs, evals, generations, usage, latency, and errors.
- Step selectors: `step(...)` and `step_text(...)`.
- `steps.csv` artifact and scoped score rows in `scores.csv`.

### Changed

- Renamed public terminology from row/execution/task to item/item-run/workflow.
- Replaced `exp.task = callable` with `exp.workflow = callable`.
- Replaced the dataset selector `row(...)` with `item(...)`.
- Renamed `ExecutionRecord` to `ItemRunRecord` and public result fields to item/item-run names.

## [0.3.0] - 2026-04-15

### Changed

- Replaced task decorators with direct task assignment via `exp.task = callable`.
- Removed eval decorator registration in favor of `exp.eval("key", evaluator)`.
- Task callables may now be sync or async, including callable task objects.

## [0.2.0] - 2026-04-15

### Added

- Unified `exp.eval("key", evaluator)` registration for custom functions and built-in evaluators.
- Built-in deterministic evaluators for equality, approximate equality, containment, regex, non-empty values, length bounds, and JSON path checks.
- Selector helpers: `row(...)`, `out(...)`, and `text(...)`.

## [0.1.0] - 2026-04-15

### Added

- Initial Prism Evals Python package with decorator-style experiments.
- `Experiment`, `ModelConfig`, `TaskOutput`, and `EvalResult` public APIs.
- OpenAI Responses API wrapper for automatic raw request/response capture.
- Token usage capture for input, cached, output, reasoning, and total tokens.
- Per-call latency and per-execution duration tracking.
- CSV dataset loading with row/model/repetition execution matrix.
- Bounded async concurrency, retries, resume support, and fail-fast option.
- Local artifacts: `manifest.json`, `results.jsonl`, `results.csv`, and `scores.csv`.
- Rich terminal progress and summary tables.
- Example QA experiment and pytest coverage.
