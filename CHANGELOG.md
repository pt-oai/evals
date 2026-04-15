# Changelog

All notable changes to this project will be documented in this file.

This project follows Semantic Versioning. While the project is pre-1.0, minor
versions may include API changes as the experiment framework settles.

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

