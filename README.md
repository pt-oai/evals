# Prism Evals

Prism Evals is a local Python framework for executable OpenAI API experiments.

The package distribution is `prism-evals`, the Python import is `prism_evals`,
and the primary CLI is `prism`. The short CLI alias is `pe`.

Experiments are plain Python files. Define the dataset, models, workflow, and evals in one place, then run them with the CLI:

```bash
prism run examples/qa_smoke.py
```

## Quick Start

From the repo where you want to write and run evals:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "prism-evals @ git+ssh://git@github.com/pt-oai/evals.git@v0.7.0"
prism init
# or: prism-evals init
# or: pe init
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="..."
```

Create a tiny dataset:

```bash
mkdir -p datasets
cat > datasets/qa.csv <<'CSV'
id,question,expected
item-1,What is 2 + 2?,4
item-2,Name the color of a clear daytime sky.,blue
CSV
```

Create an eval:

```bash
cat > qa_smoke.py <<'PY'
from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, TaskOutput, item, text

exp = Experiment(
    name="qa_smoke",
    dataset="datasets/qa.csv",
    output_dir="runs",
    concurrency=5,
    resume=True,
)

exp.model(
    ModelConfig(
        key="gpt5_low",
        model="gpt-5",
        params={"reasoning": {"effort": "low"}},
    )
)

client = AsyncOpenAI()

async def answer(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["question"],
    )
    return TaskOutput(text=response.output_text)

exp.workflow = answer
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected answer appears",
)
exp.eval("brevity", LengthBetween(value=text(), max_len=200))
PY
```

Run the eval and open the local viewer:

```bash
prism run qa_smoke.py
prism view runs/
```

The init command creates `AGENTS.md` if it does not exist. If `AGENTS.md`
already exists, it appends a marked Prism Evals section unless that section is
already present. Use `--force` to overwrite the file.

For local development on this package checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Terminology

- **Experiment**: the configured eval definition.
- **Experiment run**: one call to `exp.run()`, with one `run_id` and one output directory.
- **Item**: one dataset record. For CSV, this is one row.
- **Item run**: one `item x model x repetition` attempt.
- **Workflow**: the user-defined callable assigned to `exp.workflow`. It must return `TaskOutput`.
- **Step**: a named unit inside a workflow. Step callables must return `TaskOutput`.
- **Media**: generated output files saved through `ctx.media` and referenced from `TaskOutput.media`.
- **Eval**: one named score attached to either a step or the final item-run output.

## Basic QA Eval

Create a CSV dataset:

```csv
id,question,expected
item-1,What is 2 + 2?,4
item-2,Name the color of a clear daytime sky.,blue
```

Create an experiment file:

```python
from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, TaskOutput, item, text

exp = Experiment(
    name="qa_smoke",
    dataset="datasets/qa.csv",
    output_dir="runs",
    concurrency=5,
    resume=True,
)

exp.model(
    ModelConfig(
        key="gpt5_low",
        model="gpt-5",
        params={"reasoning": {"effort": "low"}},
    )
)

client = AsyncOpenAI()

async def answer(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["question"],
    )
    return TaskOutput(text=response.output_text)

exp.workflow = answer
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected answer appears",
)
exp.eval("brevity", LengthBetween(value=text(), max_len=200))
```

Run the experiment with `prism run qa_smoke.py`. Direct `python qa_smoke.py`
execution is still supported if the file includes an explicit `exp.run()` block.

## Multi-Step Workflow

Use `ctx.step(...)` when one dataset item needs a chain of work. Step callables must return `TaskOutput`, and step evals always receive that step's output.

```python
from openai import AsyncOpenAI

from prism_evals import ApproxEqual, Contains, Experiment, JsonPathExists, ModelConfig, NonEmpty, TaskOutput, item, out, text

exp = Experiment(name="score_chain", dataset="datasets/scoring.csv", output_dir="runs")
exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))
client = AsyncOpenAI()

async def workflow(item, model, ctx):
    async def make_draft():
        response = await client.responses.create(
            model=model.model,
            **model.params,
            input=item["prompt"],
        )
        return TaskOutput(text=response.output_text)

    draft = await ctx.step(
        "draft",
        make_draft,
        evals=[
            ("draft_non_empty", NonEmpty(text())),
            ("mentions_topic", Contains(container=text(), expected=item("topic"), case_sensitive=False)),
        ],
    )

    extracted = await ctx.step(
        "extract_score",
        lambda: TaskOutput(value=extract_score(draft.text)),
        evals=[
            ("score_exists", JsonPathExists(value=out(), path="score")),
            ("score_close", ApproxEqual(actual=out("score"), expected=item("score", cast=float), abs_tol=0.01)),
        ],
    )

    final = await ctx.step(
        "final",
        lambda: TaskOutput(text=build_final_answer(draft, extracted)),
        evals=[("final_non_empty", NonEmpty(text()))],
    )

    return final

exp.workflow = workflow
```

## Structured Output

Use the OpenAI SDK directly and return `TaskOutput` with both display text and parsed data.

```python
import json

from openai import AsyncOpenAI

from prism_evals import Experiment, JsonPathExists, ModelConfig, TaskOutput, out

exp = Experiment(name="extract_people", dataset="datasets/people.csv", output_dir="runs")
exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))
client = AsyncOpenAI()

async def extract(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["text"],
        text={
            "format": {
                "type": "json_schema",
                "name": "person",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"},
                    },
                    "required": ["name", "age"],
                    "additionalProperties": False,
                },
            }
        },
    )
    return TaskOutput(text=response.output_text, value=json.loads(response.output_text))

exp.workflow = extract
exp.eval("has_name", JsonPathExists(value=out(), path="name"))
```

## Image Generation

Prism does not proxy image generation calls. Import the OpenAI SDK, call either the Responses API or Image API, save generated bytes through `ctx.media`, and return the media in `TaskOutput`.

```python
from openai import AsyncOpenAI

from prism_evals import Experiment, ModelConfig, TaskOutput

exp = Experiment(name="image_api", dataset="datasets/prompts.csv", output_dir="runs")
exp.model(ModelConfig(key="image_high", model="gpt-image-2", params={"quality": "high"}))
client = AsyncOpenAI()

async def workflow(item, model, ctx):
    response = await client.images.generate(
        model=model.model,
        prompt=item["prompt"],
        response_format="b64_json",
        **model.params,
    )
    image = ctx.media.from_base64(response.data[0].b64_json, format="png", name=item["id"])
    return TaskOutput(text="Generated image", media=[image])

exp.workflow = workflow
```

Responses API image generation follows the same pattern: call `client.responses.create(...)`, extract the image base64 from the response, save it with `ctx.media.from_base64(...)`, and return `TaskOutput(media=[...])`.

## Built-In Evaluators

Built-ins are registered directly with `exp.eval("key", evaluator)` or as step eval tuples.

```python
from prism_evals import ApproxEqual, Contains, Equal, JsonPathExists, RegexMatch, item, out, step, step_text, text

exp.eval("exact_answer", Equal(actual=text(), expected=item("expected")))
exp.eval("score_equal", Equal(actual=out("score"), expected=item("score", cast=float)))
exp.eval("price_close", ApproxEqual(actual=out("price"), expected=item("price", cast=float), abs_tol=0.01))
exp.eval("mentions_term", Contains(container=text(), expected=item("term"), case_sensitive=False))
exp.eval("has_citation", RegexMatch(value=text(), pattern=r"\[\d+\]"))
exp.eval("has_explanation", JsonPathExists(value=out(), path="explanation"))

# In a later step eval, compare against an earlier step.
("score_matches_prior_step", Equal(actual=out("score"), expected=step("extract_score.score")))
("mentions_draft", Contains(container=text(), expected=step_text("draft"), case_sensitive=False))
```

Available built-ins:

- `Equal(actual, expected)`
- `NotEqual(actual, expected)`
- `ApproxEqual(actual, expected, abs_tol=1e-6, rel_tol=1e-9)`
- `Contains(container, expected, case_sensitive=True)`
- `RegexMatch(value, pattern, flags=0)`
- `NonEmpty(value)`
- `LengthBetween(value, min_len=None, max_len=None)`
- `JsonPathExists(value, path)`
- `JsonPathEqual(value, path, expected)`

Selectors:

- `item("column", cast=None, default=...)` reads from the dataset item.
- `out("path", cast=None, default=...)` reads from `TaskOutput.value`.
- `out()` reads the full `TaskOutput.value`.
- `text(cast=None)` reads `TaskOutput.text`.
- `step("step_key.path", cast=None, default=...)` reads a prior step output value.
- `step_text("step_key", cast=None, default=...)` reads a prior step output text.

## Custom Eval Functions

Custom eval functions receive `(item, model, output, ctx)` and can return booleans, numbers, dictionaries, `EvalResult` objects, or lists of `EvalResult` objects.

```python
from prism_evals import EvalResult

def contains_expected(item, model, output, ctx):
    return item["expected"].lower() in output.text.lower()

def quality_bundle(item, model, output, ctx):
    return {
        "non_empty": bool(output.text.strip()),
        "short_enough": len(output.text) <= 200,
    }

def manual_score(item, model, output, ctx):
    return EvalResult(score=0.8, comment="Looks mostly correct")

exp.eval("contains_expected", contains_expected)
exp.eval("quality_bundle", quality_bundle)
exp.eval("manual_score", manual_score)
```

## Outputs

For `Experiment(name="qa_smoke", output_dir="runs")`, results are written to a timestamp-prefixed folder:

```text
runs/20260415-143205_qa_smoke/
  manifest.json
  results.jsonl
  results.csv
  scores.csv
  steps.csv
  artifacts/
  media/
```

- `manifest.json` stores experiment-run settings, model configs, dataset hash, experiment hash, copied artifact metadata, and environment metadata.
- `results.jsonl` stores full item-run records, including final `TaskOutput`, step records, scores, usage, latency, media metadata, and errors.
- `results.csv` is a spreadsheet-friendly summary with one row per item run, including compact media columns.
- `scores.csv` is long-form score data with `scope` and `step_key` columns.
- `steps.csv` is a spreadsheet-friendly summary with one row per step, including compact media columns.
- `artifacts/` contains files copied from `Experiment(..., artifacts=[...])`, such as prompt templates or run configs.
- `media/` contains generated outputs saved with `ctx.media`.

The final console summary includes score tables by model and by eval key, plus average per-item-run and total token usage by model for input, cached, output, reasoning, and total tokens.

Set `timestamp_output_dir=False` to keep the stable `runs/qa_smoke/` folder shape.

## Local Viewer

The local viewer opens a parent runs directory and shows every child run folder
that contains `manifest.json`:

```bash
prism view runs/
```

The viewer is read-only. It shows an all-runs table, per-run item details, media
previews, score matrices, step details, downloadable artifacts, and lane comparisons across
`run + model_key` pairs.

The first launch installs the viewer's Node dependencies if they are missing.
Node.js and npm must be available on `PATH`.

## Settings

```python
exp = Experiment(
    name="my_eval",
    dataset="datasets/input.csv",
    output_dir="runs",
    concurrency=5,
    resume=True,
    repetitions=1,
    max_retries=3,
    fail_fast=False,
    capture_raw=True,
    redact_raw_data_urls=True,
    timestamp_output_dir=True,
    artifacts=["prompts/system.md", "prompts/*.json"],
    display="progress",  # progress, quiet, debug
    metadata={"owner": "research"},
)
```

`redact_raw_data_urls=True` keeps multimodal runs compact by replacing inline
base64 media in legacy proxied raw payloads with a short deterministic marker.
Direct SDK calls are not captured automatically; store relevant details in
`TaskOutput.metadata` when you need them.

## Migration Notes

See [MIGRATION.md](MIGRATION.md) for the complete guide. Prism now requires
explicit `TaskOutput` returns.

```python
# Old
return response.output_text

# New
return TaskOutput(text=response.output_text)
```

For structured outputs:

```python
# Old
return {"answer": parsed}

# New
return TaskOutput(text=json.dumps(parsed), value=parsed)
```

Generated files belong in `media/` through `ctx.media`; `artifacts/` remains for copied experiment inputs. Custom JSONL consumers should read generated media from `output.media` or `steps[].output.media` instead of parsing image bytes from raw OpenAI responses.
