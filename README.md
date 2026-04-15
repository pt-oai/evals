# evals

`evals` is a local Python framework for executable OpenAI Responses API experiments.

The package distribution is named `pt-evals`, but the Python import is `evals`.

Experiments are plain Python files. Define the dataset, models, workflow, and evals in one place, then run the file:

```bash
python examples/qa_smoke.py
```

Install locally:

```bash
python -m pip install -e .
```

Install from GitHub:

```bash
python -m pip install "pt-evals @ git+ssh://git@github.com/pt-oai/evals.git@v0.4.3"
```

## Terminology

- **Experiment**: the configured eval definition.
- **Experiment run**: one call to `exp.run()`, with one `run_id` and one output directory.
- **Item**: one dataset record. For CSV, this is one row.
- **Item run**: one `item x model x repetition` attempt.
- **Workflow**: the user-defined callable assigned to `exp.workflow`.
- **Step**: a named unit inside a workflow, with its own output, evals, generations, latency, and usage.
- **Generation**: one captured Responses API call.
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
from evals import Contains, Experiment, LengthBetween, ModelConfig, item, text

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

async def answer(item, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=item["question"],
    )
    return response.output_text

exp.workflow = answer
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected answer appears",
)
exp.eval("brevity", LengthBetween(value=text(), max_len=200))

if __name__ == "__main__":
    exp.run()
```

## Multi-Step Workflow

Use `ctx.step(...)` when one dataset item needs a chain of work. Step evals always receive that step's output.

```python
from evals import ApproxEqual, Contains, Experiment, JsonPathExists, ModelConfig, NonEmpty, item, out, text

exp = Experiment(name="score_chain", dataset="datasets/scoring.csv", output_dir="runs")
exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))

async def workflow(item, model, ctx):
    async def make_draft():
        response = await ctx.responses.create(
            model=model.model,
            **model.params,
            input=item["prompt"],
        )
        return response.output_text

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
        lambda: extract_score(draft.text),
        evals=[
            ("score_exists", JsonPathExists(value=out(), path="score")),
            ("score_close", ApproxEqual(actual=out("score"), expected=item("score", cast=float), abs_tol=0.01)),
        ],
    )

    final = await ctx.step(
        "final",
        lambda: build_final_answer(draft, extracted),
        evals=[("final_non_empty", NonEmpty(text()))],
    )

    return final

exp.workflow = workflow
```

## Structured Output

Responses API structured output works through the wrapped `ctx.responses.create(...)` call. Return `TaskOutput` when you want to keep both text and parsed data.

```python
import json

from evals import Experiment, JsonPathExists, ModelConfig, TaskOutput, out

exp = Experiment(name="extract_people", dataset="datasets/people.csv", output_dir="runs")
exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))

async def extract(item, model, ctx):
    response = await ctx.responses.create(
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

if __name__ == "__main__":
    exp.run()
```

## Built-In Evaluators

Built-ins are registered directly with `exp.eval("key", evaluator)` or as step eval tuples.

```python
from evals import ApproxEqual, Contains, Equal, JsonPathExists, RegexMatch, item, out, step, step_text, text

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
from evals import EvalResult

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
```

- `manifest.json` stores experiment-run settings, model configs, dataset hash, experiment hash, and environment metadata.
- `results.jsonl` stores full item-run records, including raw request/response data, final output, step records, scores, usage, latency, and errors.
- `results.csv` is a spreadsheet-friendly summary with one row per item run.
- `scores.csv` is long-form score data with `scope` and `step_key` columns.
- `steps.csv` is a spreadsheet-friendly summary with one row per step.

The final console summary includes average per-item-run and total token usage by model for input, cached, output, reasoning, and total tokens.

Set `timestamp_output_dir=False` to keep the stable `runs/qa_smoke/` folder shape.

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
    timestamp_output_dir=True,
    display="progress",  # progress, quiet, debug
    metadata={"owner": "research"},
)
```
