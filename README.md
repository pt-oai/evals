# Prism Evals

Prism Evals is a local Python framework for executable OpenAI API experiments.

The package distribution is `prism-evals`, the Python import is `prism_evals`,
and the primary CLI is `prism`. The short CLI alias is `pe`.

Experiments are plain Python files. Define the dataset, models, workflow, and evals in one place, then run them with the CLI:

```bash
prism run examples/01_csv_qa.py
```

## Quick Start

From the repo where you want to write and run evals:

```bash
python3.12 -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip prism-evals && prism init
```

That installs the latest published Prism Evals package into `.venv`, then seeds
Prism Evals instructions into the current repo.

To install directly from the repo before a release is published:

```bash
python3.12 -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip "prism-evals @ git+ssh://git@github.com/pt-oai/evals.git" && prism init
```

The installed console scripts are equivalent, so `prism init`, `prism-evals
init`, and `pe init` all work.

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
cat > csv_qa.py <<'PY'
from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, TaskOutput, item, text

exp = Experiment(
    name="csv_qa",
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
prism run csv_qa.py
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

## Publishing

Releases are published by GitHub Actions from version tags. Before the first
release, create a PyPI pending Trusted Publisher for this repository with:

- **PyPI project name**: `prism-evals`
- **Owner**: `pt-oai`
- **Repository name**: `evals`
- **Workflow name**: `publish-python.yml`
- **Environment name**: `pypi`

Then update `version` in `pyproject.toml`, commit the change, and push a tag:

```bash
git tag v0.9.2
git push origin v0.9.2
```

The publish workflow runs tests, builds the source distribution and wheel,
checks the package metadata, and publishes to PyPI only for `v*` tags. The
pending publisher creates the PyPI project on first publish, then becomes the
normal trusted publisher for future releases.

## Runnable Examples

The `examples/` directory is organized as a small learning path:

```bash
prism run examples/01_csv_qa.py
prism run examples/02_json_import.py
prism run examples/03_image_generation.py
prism run examples/04_config_options.py
prism run examples/05_multistep_agent.py
```

- `01_csv_qa.py`: CSV import and simple built-in evals.
- `02_json_import.py`: one JSON file per case with nested context.
- `03_image_generation.py`: generated image media saved with `ctx.media`.
- `04_config_options.py`: retries, repetitions, artifacts, metadata, stable output directories, and model variants.
- `05_multistep_agent.py`: step-level evals inside a two-step workflow.

## Terminology

- **Experiment**: the configured eval definition.
- **Experiment run**: one call to `exp.run()`, with one `run_id` and one output directory.
- **Item**: one dataset record. For CSV, this is one row. For scenario folders,
  this is one JSON/YAML file.
- **Item run**: one `item x model variant x repetition` attempt. Plain
  `exp.model(...)` entries are treated as single-model variants.
- **Workflow**: the user-defined callable assigned to `exp.workflow`. It must return `TaskOutput`.
- **Step**: a named unit inside a workflow. Step callables must return `TaskOutput`.
- **Turn**: a conversation turn recorded with `ctx.turn(...)`, `ctx.user(...)`,
  `ctx.assistant_seed(...)`, or `ctx.action_seed(...)`.
- **Tool call**: an application/tool invocation recorded with
  `ctx.record_tool_call(...)`.
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
    name="csv_qa",
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

Run the experiment with `prism run csv_qa.py`. Direct `python csv_qa.py`
execution is still supported if the file includes an explicit `exp.run()` block.

## Scenario Datasets

CSV remains supported, but multi-turn agent evals can use a folder of scenario
files:

```python
exp = Experiment(
    name="json_import",
    dataset="datasets/json_cases",
    output_dir="runs",
)
```

When `dataset` is a directory, Prism recursively reads every `*.json`,
`*.yaml`, and `*.yml` file in stable path order. Files whose names start with
`_` are ignored, so you can keep notes or shared fragments next to cases. One
file becomes one item; `id` comes from the file, falling back to the filename.

```yaml
id: hardware-followup-context
tags: [support, hardware, multiturn]
context:
  merchantId: me_bharat_bazaar

turns:
  - user: My terminal is not connecting.
  - assistant_seed: Please check the terminal power and network connection.
  - user: my screen keeps flashing
    expect:
      allowed: true
      route: support
```

The shorthand turn keys normalize to explicit turn objects. `assistant_seed`
adds an assistant message to the scenario history; it does not require the
system under test to generate that message.

CSV can also point to scenario files:

```csv
id,scenario_path,tags
hardware-followup,scenarios/hardware_followup.yaml,"support,hardware"
```

Or keep compact cases inline with JSON columns:

```csv
id,turns_json
case-1,"[{""user"":""hello""},{""assistant_seed"":""Hi. How can I help?""}]"
```

Directory datasets are hashed from the loaded scenario files, and each item
stores its own content hash so resume invalidates when a scenario changes.

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

## Model Variants

Use `exp.model(...)` for simple model comparisons. For multi-agent workflows
where roles use different models, register named variants:

```python
exp.variant(
    "baseline",
    models={
        "router": "gpt-5.4-nano",
        "support": {"model": "gpt-5.4-mini", "params": {"text": {"verbosity": "low"}}},
    },
    default_role="support",
)

exp.variant(
    "candidate",
    models={
        "router": "gpt-5.4-mini",
        "support": {"model": "gpt-5.5", "params": {"text": {"verbosity": "low"}}},
    },
    default_role="support",
)
```

The workflow still receives `(item, model, ctx)`. `model.key` is the variant
key, while `ctx.model("router")` and `ctx.model("support")` return the
role-specific `ModelConfig`. `ctx.model.model` and `model.model` continue to
return the default role's model for backwards compatibility.

## Conversation Turns And Tool Calls

Use conversation helpers when one item is a multi-turn scenario:

```python
from prism_evals import ToolCalled, TaskOutput

async def workflow(item, model, ctx):
    async with ctx.conversation(item) as convo:
        convo.user("start", item["turns"][0]["content"])
        convo.assistant_seed("prior", "Please check the terminal power and network connection.")

        async def followup():
            ctx.record_tool_call(
                "lookup_terminal",
                arguments={"serial_number": "PSP-BBR-001"},
                result={"status": "online"},
                agent="support",
            )
            return TaskOutput(text="The terminal is online. Please restart it and try one test transaction.")

        await convo.turn(
            "followup",
            followup,
            evals=[("lookup_called", ToolCalled("lookup_terminal", turn="followup"))],
        )

        return convo.task_output()
```

`ctx.user(...)`, `ctx.assistant_seed(...)`, and `ctx.action_seed(...)` record
seeded context turns. `ctx.turn(...)` records a generated turn and also writes a
normal `StepRecord` with key `turn:<turn_id>` so existing step-level storage and
comparison still work.

Tool calls can be scored with built-ins:

```python
from prism_evals import ToolArgsEqual, ToolCalled, ToolNotCalled

exp.eval("called_lookup", ToolCalled("lookup_terminal", turn="followup"))
exp.eval("quantity_is_two", ToolArgsEqual("start_order", "quantity", 2))
exp.eval("did_not_refund", ToolNotCalled("refund_transaction"))
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

Responses API image generation follows the same pattern: call `client.responses.create(...)`, extract the image base64 from the response, save it with `ctx.media.from_base64(...)`, and return `TaskOutput(media=[...])`. See `examples/03_image_generation.py` for a runnable version.

## Realtime

Use `ctx.realtime` for Realtime API evals. Prism manages the session, captures one generation record per completed Realtime response, and keeps the normal `TaskOutput` workflow contract.

```python
from prism_evals import Contains, Experiment, ModelConfig, item, text

exp = Experiment(name="realtime_text", dataset="datasets/realtime.csv", output_dir="runs")
exp.model(ModelConfig(key="realtime2_low", model="gpt-realtime-2", params={"reasoning": {"effort": "low"}}))

async def workflow(item, model, ctx):
    result = await ctx.realtime.run_text(
        item["prompt"],
        instructions="Answer briefly and follow the user's requested format exactly.",
    )
    return result.task_output()

exp.workflow = workflow
exp.eval("contains_expected", Contains(container=text(), expected=item("expected"), case_sensitive=False))
```

Realtime audio evals can stream 16-bit mono PCM WAV fixtures and store model audio as playable WAV media:

```python
async def workflow(item, model, ctx):
    result = await ctx.realtime.run_audio(
        item["audio_path"],
        session={
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None},
                "output": {"format": {"type": "audio/pcm"}, "voice": "marin"},
            }
        },
    )
    return result.task_output()
```

Realtime results include parsed tool calls in `TaskOutput.value`, so evals can
score whether each turn called a tool without walking raw events:

```python
from prism_evals import EvalResult

def called_lookup_tool(item, model, output, ctx):
    tool_calls = output.value.get("tool_calls", []) if isinstance(output.value, dict) else []
    names = [call.get("name") for call in tool_calls]
    return EvalResult(
        score="lookup_order" in names,
        metadata={"tool_call_count": len(tool_calls), "tool_call_names": names},
    )
```

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

## Pass Conditions

By default, eval scores do not change an item run's `success` status. Set
`exp.pass_condition` when specific item-level scores must determine whether the
item run passes:

```python
exp.eval("correct", Equal(actual=text(), expected=item("expected")))
exp.eval("quality", quality_judge)

exp.pass_condition = lambda scores: (
    scores["correct"] is True
    and scores["quality"] >= 0.8
)
```

The condition receives a dictionary mapping each item-level eval key to its
score. Values are boolean or numeric when evaluation succeeds, and may be `None`
when an evaluator errors. The condition may be synchronous or asynchronous. A
false result sets the item run status to `failed` with error type
`PassConditionFailed`, while preserving the workflow output and all eval results
for inspection. Without a pass condition, scores remain informational and the
existing status behavior is unchanged.

Step-level eval scores are not included in this dictionary. If `resume=True`,
item runs that fail the pass condition are retried because resume skips only
successful records. Adding or changing a pass condition does not invalidate
previously successful records in a stable run directory; use a fresh run
directory for changed criteria.

## Outputs

For `Experiment(name="csv_qa", output_dir="runs")`, results are written to a timestamp-prefixed folder:

```text
runs/20260415-143205_csv_qa/
  manifest.json
  results.jsonl
  results.csv
  scores.csv
  steps.csv
  turns.csv
  tool_calls.csv
  artifacts/
  media/
```

- `manifest.json` stores experiment-run settings, model configs, dataset hash, experiment hash, copied artifact metadata, and environment metadata.
- `results.jsonl` stores full item-run records, including final `TaskOutput`, step records, scores, usage, latency, media metadata, and errors.
- `results.csv` is a spreadsheet-friendly summary with one row per item run, including compact media columns.
- `scores.csv` is long-form score data with `scope` and `step_key` columns.
- `steps.csv` is a spreadsheet-friendly summary with one row per step, including compact media columns.
- `turns.csv` is a spreadsheet-friendly summary with one row per recorded conversation turn.
- `tool_calls.csv` is a spreadsheet-friendly summary with one row per recorded tool call.
- `artifacts/` contains files copied from `Experiment(..., artifacts=[...])`, such as prompt templates or run configs.
- `media/` contains generated outputs saved with `ctx.media`.

The final console summary includes score tables by model and by eval key, plus average per-item-run and total token usage by model for input, cached, output, reasoning, and total tokens.

Set `timestamp_output_dir=False` to keep the stable `runs/csv_qa/` folder shape.

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
