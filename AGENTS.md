# AGENTS.md

This repo provides Prism Evals, a local Python framework for executable OpenAI
API experiments. The package distribution is `prism-evals`, and the Python
import is `prism_evals`.

Use this file as the quick orientation for LLM agents working in this package
repo, and as a template for consuming repos that install Prism Evals.

To seed Prism Evals instructions into a consuming repo, install the package and
run this from the consuming repo root:

```bash
python -m prism_evals init
```

The installed console scripts also work:

```bash
prism init
prism-evals init
pe init
```

## What An Eval Experiment Is

An experiment is a normal Python file that configures:

- A CSV dataset.
- One or more `ModelConfig` entries.
- A workflow callable assigned to `exp.workflow`.
- Item-level evals registered with `exp.eval(...)`.
- Optional step-level evals inside `ctx.step(...)`.

Run an experiment with the Prism CLI:

```bash
prism run path/to/experiment.py
```

The minimal shape is:

```python
from openai import AsyncOpenAI

from prism_evals import Experiment, ModelConfig, TaskOutput

exp = Experiment(name="my_eval", dataset="datasets/my_eval.csv", output_dir="runs")
exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))
client = AsyncOpenAI()

async def workflow(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["prompt"],
    )
    return TaskOutput(text=response.output_text)

exp.workflow = workflow
```

Direct `python path/to/experiment.py` execution is still supported if the file
includes an explicit `exp.run()` block.

## Where To Make Changes

When changing eval behavior in a consuming repo, edit the experiment Python files
first. Those files are the source of truth for datasets, model choices,
workflow logic, scoring rules, concurrency, retries, and output settings.

Common change points:

- Change prompts, tool calls, multi-step logic, or response parsing in the
  workflow assigned to `exp.workflow`.
- Change model coverage by editing `exp.model(...)` or `exp.models(...)`.
- Change pass/fail or scoring logic by editing `exp.eval(...)` or the `evals=`
  list passed to `ctx.step(...)`.
- Change data coverage by editing the CSV dataset referenced by
  `Experiment(dataset=...)`.
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
- `artifacts/`: optional copied prompt/rubric/schema files listed in
  `artifacts=[...]`.
- `media/`: generated outputs saved with `ctx.media`.

`runs/` and `examples/runs/` are ignored by git in this repo. Treat run outputs
as generated artifacts unless the consuming repo explicitly chooses to version
selected reports.

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

Use `steps.csv` when debugging multi-step workflows. It shows each step's
status, output text, media paths, token usage, latency, response ID, and
`scores_json`.

Use `results.jsonl` when full record structure matters. It preserves nested
records for items, final `TaskOutput` values, evals, optional legacy generation
records, steps, errors, usage, and media metadata.

## Resume Behavior

`resume=True` skips item/model/repetition records that already have
`status == "success"` in the current run directory's `results.jsonl`.

The item-run identity is based on experiment name, dataset hash, item id/index,
model key, and repetition. Changing the dataset contents changes the dataset hash
and creates different item-run IDs.

For a clean rerun, launch a fresh process with `timestamp_output_dir=True`,
change the output directory or experiment name, or delete the old generated run
directory.

## Datasets

Datasets are CSV files read with a header row.

- Every row is passed to the workflow as `item`, a `dict[str, str]`.
- If the CSV has an `id` column, that value is used as `item_id`.
- If `id` is missing or blank, the zero-based row index is used as `item_id`.
- Empty CSV values are normalized to empty strings.

Keep dataset columns explicit and stable. Evals often use selectors such as
`item("expected")`, so renaming columns can silently change scoring behavior.

## Workflows And Steps

A workflow receives `(item, model, ctx)` and may be sync or async. It must
return `TaskOutput`.

Import and use the OpenAI SDK directly inside experiment files. Prism owns eval
orchestration and output storage, not provider SDK calls.

Use `ctx.step("step_key", callable_or_value, evals=[...])` for multi-step
workflows. Step callables must return `TaskOutput`. Step outputs and step evals
are written to both `results.jsonl` and the flattened CSV files.

Return `TaskOutput(text=..., value=..., media=[...])` for display text,
structured data, and generated media. Built-in selectors such as `text()`,
`out("path")`, `step("step_key.path")`, and `step_text("step_key")` depend on
that structure. Use `ctx.media.from_base64(...)`, `ctx.media.from_bytes(...)`,
or `ctx.media.from_path(...)` to save generated images into `media/`.

## Viewer UI Preferences

When editing the local Next.js viewer, optimize for dense eval inspection rather
than marketing-style presentation:

- Use the full viewport width for run tables, compare views, and charts.
- Keep visible copy short and product-facing; avoid implementation details in
  the UI.
- Put related controls in the same toolbar row when space allows, with secondary
  controls aligned to the right.
- Prefer familiar icon-only controls for common actions such as close and remove.
  Use a small, light `×` with an accessible label rather than bordered text
  buttons.
- Keep chart previews compact. Avoid legends or section labels in small cards
  when the surrounding context already names the metric.
- For run trend charts, show runs chronologically left to right so the newest
  run is on the right.
- Preserve user filters across summaries and charts; model filters should affect
  both the table and chart previews.

## Testing Changes

For this package repo:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

For a consuming repo, run the smallest relevant experiment first, then inspect
the generated run directory before broadening concurrency, model coverage, or
dataset size.
