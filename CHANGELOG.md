# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning. While the project is pre-1.0, minor
versions may include API changes as the experiment framework settles.

## [0.4.3] - 2026-04-15

### Added

- Console token usage now shows average per item run and totals for input, cached, output, reasoning, and total tokens.

## [0.4.2] - 2026-04-15

### Fixed

- Declared the `src/evals` package for Hatchling wheel builds so `pt-evals` installs correctly from Git tags.

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

- Initial `evals` Python package with decorator-style experiments.
- `Experiment`, `ModelConfig`, `TaskOutput`, and `EvalResult` public APIs.
- OpenAI Responses API wrapper for automatic raw request/response capture.
- Token usage capture for input, cached, output, reasoning, and total tokens.
- Per-call latency and per-execution duration tracking.
- CSV dataset loading with row/model/repetition execution matrix.
- Bounded async concurrency, retries, resume support, and fail-fast option.
- Local artifacts: `manifest.json`, `results.jsonl`, `results.csv`, and `scores.csv`.
- Rich terminal progress and summary tables.
- Example QA experiment and pytest coverage.
