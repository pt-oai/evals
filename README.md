# evals

`evals` is a local Python framework for executable OpenAI Responses API experiments.

The package distribution is named `pt-evals`, but the Python import is `evals`.

Experiments are plain Python files. Define the dataset, models, task, and evals in one place, then run the file:

```bash
python -m pip install -e .
```

To install from GitHub after this repo is pushed:

```bash
python -m pip install "pt-evals @ git+ssh://git@github.com/pt-oai/evals.git"
```

To pin a specific release:

```bash
python -m pip install "pt-evals @ git+ssh://git@github.com/pt-oai/evals.git@v0.1.0"
```

```bash
python examples/qa_smoke.py
```

## Basic QA Eval

Create a CSV dataset:

```csv
id,question,expected
row-1,What is 2 + 2?,4
row-2,Name the color of a clear daytime sky.,blue
```

Create an experiment file:

```python
from evals import EvalResult, Experiment, ModelConfig

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

@exp.task
async def answer(row, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=row["question"],
    )
    return response.output_text

@exp.eval("contains_expected", description="Expected answer appears")
def contains_expected(row, model, output, ctx):
    return row["expected"].lower() in output.text.lower()

@exp.eval("brevity")
def brevity(row, model, output, ctx):
    return EvalResult(score=min(1.0, 200 / max(len(output.text), 1)))

if __name__ == "__main__":
    exp.run()
```

Run it:

```bash
python examples/qa_smoke.py
```

## Structured Output

Responses API structured output works through the same wrapped `ctx.responses.create(...)` call. Return a `TaskOutput` when you want to keep both the raw text and parsed object.

```python
import json

from evals import Experiment, ModelConfig, TaskOutput

exp = Experiment(
    name="extract_people",
    dataset="datasets/people.csv",
    output_dir="runs",
)

exp.model(ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}))

@exp.task
async def extract(row, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=row["text"],
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

@exp.eval("has_name")
def has_name(row, model, output, ctx):
    return bool(output.value["name"])

@exp.eval("age_in_range")
def age_in_range(row, model, output, ctx):
    return 0 <= output.value["age"] <= 130

if __name__ == "__main__":
    exp.run()
```

## Compare Multiple Models

Register as many model configs as you want. The runner executes every dataset row against every model.

```python
from evals import Experiment, ModelConfig

exp = Experiment(
    name="reasoning_compare",
    dataset="datasets/math.csv",
    output_dir="runs",
    concurrency=4,
    repetitions=2,
)

exp.models([
    ModelConfig(key="gpt5_low", model="gpt-5", params={"reasoning": {"effort": "low"}}),
    ModelConfig(key="gpt5_high", model="gpt-5", params={"reasoning": {"effort": "high"}}),
])

@exp.task
async def solve(row, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=row["problem"],
    )
    return response.output_text

@exp.eval("exact_match")
def exact_match(row, model, output, ctx):
    return output.text.strip() == row["answer"].strip()

if __name__ == "__main__":
    exp.run()
```

## Scoring Shapes

Eval functions can return booleans, numbers, dictionaries, or `EvalResult` objects.

```python
from evals import EvalResult

@exp.eval("contains_expected")
def contains_expected(row, model, output, ctx):
    return row["expected"].lower() in output.text.lower()

@exp.eval("quality_bundle")
def quality_bundle(row, model, output, ctx):
    return {
        "non_empty": bool(output.text.strip()),
        "short_enough": len(output.text) < 500,
        "length_score": min(1.0, 200 / max(len(output.text), 1)),
    }

@exp.eval("manual_score")
def manual_score(row, model, output, ctx):
    return EvalResult(
        score=0.8,
        description="Hand-authored score with metadata",
        metadata={"rubric": "v1"},
    )
```

Each run writes local artifacts under `output_dir/name`:

- `manifest.json`
- `results.jsonl`
- `results.csv`
- `scores.csv`

## Output Files

For `Experiment(name="qa_smoke", output_dir="runs")`, results are written to:

```text
runs/qa_smoke/
  manifest.json
  results.jsonl
  results.csv
  scores.csv
```

- `results.jsonl` stores full execution records, including raw request/response data, parsed task output, scores, usage, latency, and errors.
- `results.csv` is a spreadsheet-friendly summary with one row per dataset/model/repetition execution.
- `scores.csv` is long-form score data with one row per score.
- `manifest.json` stores run settings, model configs, dataset hash, experiment hash, and environment metadata.

## Useful Options

```python
exp = Experiment(
    name="my_eval",
    dataset="datasets/input.csv",
    output_dir="runs",
    concurrency=8,
    resume=True,
    repetitions=3,
    max_retries=3,
    fail_fast=False,
    capture_raw=True,
    display="progress",  # "progress", "quiet", or "debug"
)
```

## Versioning

This project uses Semantic Versioning with Git tags:

- Patch releases fix bugs without changing the public API.
- Minor releases add features or adjust APIs while the project is pre-1.0.
- Major releases are reserved for stable post-1.0 breaking changes.

Release checklist:

```bash
python3 -m pytest
git add pyproject.toml CHANGELOG.md README.md
git commit -m "Release v0.1.0"
git tag -a v0.1.0 -m "v0.1.0"
git push
git push origin v0.1.0
```
