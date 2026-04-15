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

```bash
python examples/qa_smoke.py
```

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

Each run writes local artifacts under `output_dir/name`:

- `manifest.json`
- `results.jsonl`
- `results.csv`
- `scores.csv`
