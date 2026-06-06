<!-- prism-evals instructions begin -->
# Prism Evals

This repo uses `prism-evals` for local executable OpenAI API
experiments. The package distribution is named `prism-evals`; the Python import is
`prism_evals`.

Common commands:

```bash
prism init
prism run path/to/experiment.py
prism view runs/
pe run path/to/experiment.py
pe view runs/
```

## What An Eval Experiment Is

An experiment is a normal Python file that configures:

- A CSV dataset, JSONL file, or folder of JSON/YAML scenario files.
- One or more `ModelConfig` entries or named model variants.
- A workflow callable assigned to `exp.workflow`.
- Item-level evals registered with `exp.eval(...)`.
- Optional step-level evals inside `ctx.step(...)`.

Run an experiment with the Prism CLI:

```bash
prism run path/to/experiment.py
```

Direct `python path/to/experiment.py` execution is still supported if the file
includes an explicit `exp.run()` block.

Experiment files import the public API from `prism_evals`:

```python
from prism_evals import Experiment, ModelConfig, TaskOutput
```

## Where To Make Changes

When changing eval behavior, edit the experiment Python files first. Those files
are the source of truth for datasets, model choices, workflow logic, scoring
rules, concurrency, retries, and output settings.

Common change points:

- Change prompts, tool calls, multi-step logic, or response parsing in the
  workflow assigned to `exp.workflow`.
- Change model coverage by editing `exp.model(...)`, `exp.models(...)`, or
  `exp.variant(...)` for multi-agent role/model configurations.
- Change pass/fail or scoring logic by editing `exp.eval(...)` or the `evals=`
  list passed to `ctx.step(...)`.
- Set `exp.pass_condition` when item-level eval scores should determine the item
  run's pass/fail status.
- Change data coverage by editing the CSV/JSONL file or scenario folder
  referenced by `Experiment(dataset=...)`.
- Change output placement by editing `Experiment(output_dir=...)`.
- Change resume behavior with `resume=True` or `resume=False`.
- Change repeated sampling with `repetitions=...`.
- Change parallelism with `concurrency=...`.

If prompts, rubrics, schemas, or other files are part of the experiment, keep
them near the experiment file and pass them through `artifacts=[...]` so each run
copies them into the run directory.

## Paths Are Relative To The Experiment File

Relative `dataset`, `output_dir`, and `artifacts` paths are resolved relative to
the Python file that creates `Experiment(...)`, not necessarily the process
working directory.

For example, in `experiments/qa.py`:

```python
Experiment(
    name="qa",
    dataset="datasets/qa.csv",
    output_dir="runs",
    artifacts=["prompts/system.md"],
)
```

This reads `experiments/datasets/qa.csv`, writes under `experiments/runs/`, and
copies `experiments/prompts/system.md` into the run artifacts.

## Result Storage

Each `Experiment` instance chooses one run directory. With default settings, the
timestamp is created when `Experiment(...)` is constructed:

```text
<output_dir>/<YYYYMMDD-HHMMSS>_<experiment_name>/
```

If `timestamp_output_dir=False`, the run directory is:

```text
<output_dir>/<experiment_name>/
```

Run directories contain:

- `manifest.json`: run metadata, dataset hash, experiment file hash, settings,
  model configs, git commit, copied artifact metadata, and output path.
- `results.jsonl`: append-only item-run records. This is the most complete
  machine-readable output.
- `results.csv`: one row per item/model/repetition with flattened item fields,
  final output text, usage, errors, item-level scores, and step score columns.
- `scores.csv`: one row per score, including both item-level and step-level
  scores.
- `steps.csv`: one row per recorded workflow step, including step output text,
  usage, errors, media columns, and step scores.
- `turns.csv`: one row per recorded conversation turn.
- `tool_calls.csv`: one row per recorded tool call.
- `artifacts/`: optional copied prompt/rubric/schema files listed in
  `artifacts=[...]`.
- `media/`: generated outputs saved with `ctx.media`.

Treat run outputs as generated artifacts unless this repo explicitly chooses to
version selected reports.

## How To Read Results

Start with `manifest.json` to confirm the experiment file, dataset, model
configs, and settings that produced the run.

Use `results.csv` for quick spreadsheet-style inspection. Useful columns include:

- `item_id`, `item_index`, `model_key`, `repetition`, and `status`.
- `output_text` for the final workflow output.
- `media_count`, `media_paths_json`, and `primary_media_path` for generated
  output files.
- `score:<eval_key>` and `score_error:<eval_key>` for item-level evals.
- `step:<step_key>.score:<eval_key>` for step evals flattened into the item row.
- `input_tokens`, `output_tokens`, `reasoning_tokens`, `total_tokens`, and
  `latency_s` for usage and timing.
- `error_type` and `error_message` for failed item runs.

Use `scores.csv` when comparing eval metrics across models, repetitions, or
steps. Filter by:

- `scope=item_run` for final-output evals.
- `scope=step` and `step_key=<name>` for step evals.
- `score_key=<eval>` for a specific metric.

Eval scores are informational by default. A synchronous or asynchronous
`exp.pass_condition(scores)` callable can combine item-level boolean and numeric
scores into an overall result. A false result marks the item run as `failed`
with `PassConditionFailed`; step-level eval scores are not included. Evaluator
errors have a `None` score, so conditions should handle that case when
`fail_fast=False`.

Use `steps.csv` when debugging multi-step workflows. It shows each step's
status, output text, media paths, token usage, latency, response ID, and
`scores_json`.

Use `results.jsonl` when full record structure matters. It preserves nested
records for items, final `TaskOutput` values, evals, optional legacy generation
records, steps, errors, usage, and media metadata.

## Resume Behavior

`resume=True` skips item/model/repetition records that already have
`status == "success"` in the current run directory's `results.jsonl`.

The item-run identity is based on experiment name, dataset hash, per-item hash,
item id/index, model or variant key, and repetition. Changing dataset contents or
a scenario file creates different item-run IDs.

For a clean rerun, launch a fresh process with `timestamp_output_dir=True`,
change the output directory or experiment name, or delete the old generated run
directory.

## Datasets

Datasets can be CSV files, JSONL files, structured JSON/YAML scenario files, or
folders containing one JSON/YAML scenario file per item.

- Every item is passed to the workflow as `item`; CSV values remain strings,
  while JSON/YAML items preserve nested lists and objects.
- If the CSV has an `id` column, that value is used as `item_id`.
- If `id` is missing or blank, the zero-based row index is used as `item_id`.
- Empty CSV values are normalized to empty strings.
- If `dataset` is a directory, Prism recursively reads `*.json`, `*.yaml`, and
  `*.yml` files in stable path order. Files beginning with `_` are ignored.
- Scenario turns support shorthand keys such as `user`, `assistant_seed`,
  `assistant_expect`, and `action`.

Keep dataset columns explicit and stable. Evals often use selectors such as
`item("expected")`, so renaming columns can silently change scoring behavior.

CSV can point to scenario files with `scenario_path`, or store compact turn lists
in `turns_json`.

## Workflows And Steps

A workflow receives `(item, model, ctx)` and may be sync or async. It must
return `TaskOutput`.

Import and use the OpenAI SDK directly inside experiment files. Prism owns eval
orchestration and output storage, not provider SDK calls.

Use `ctx.step("step_key", callable_or_value, evals=[...])` for multi-step
workflows. Step callables must return `TaskOutput`. Step outputs and step evals
are written to both `results.jsonl` and the flattened CSV files.

Use `ctx.conversation(...)`, `ctx.user(...)`, `ctx.assistant_seed(...)`,
`ctx.action_seed(...)`, and `ctx.turn(...)` for multi-turn scenarios. Seeded
turns record context only; generated turns also write a normal step with key
`turn:<turn_id>`.

Use `ctx.record_tool_call(...)` to capture app/tool invocations that matter for
scoring. Built-ins such as `ToolCalled`, `ToolNotCalled`, and `ToolArgsEqual`
can score those calls.

Return `TaskOutput(text=..., value=..., media=[...])` for display text,
structured data, and generated media. Built-in selectors such as `text()`,
`out("path")`, `step("step_key.path")`, and `step_text("step_key")` depend on
that structure. Use `ctx.media.from_base64(...)`, `ctx.media.from_bytes(...)`,
or `ctx.media.from_path(...)` to save generated outputs into `media/`.

Use `ctx.realtime.run_text(...)` or `ctx.realtime.run_audio(...)` for Realtime
API evals. Realtime helpers still return data through `TaskOutput`, and audio
outputs should be attached as run-local media.
<!-- prism-evals instructions end -->
