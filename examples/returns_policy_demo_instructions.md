# Returns Policy Demo

This demo teaches evals with one policy-answering prompt and one small dataset.
The first prompt is intentionally weak so a live run should surface failures that
you can fix by editing `examples/prompts/returns_policy_weak.md`.

## Files

- `examples/prompts/returns_policy.md`: short policy to read with the user.
- `examples/datasets/returns_policy_demo.csv`: synthetic cases with a question,
  expected output label, and response explanation for the judge.
- `examples/prompts/returns_policy_weak.md`: starter prompt to improve live.
- `examples/returns_policy_demo.py`: runnable Prism experiment.

## Setup

Recommended: use a separate demo worktree so generated runs and live prompt
edits do not clutter your dev checkout.

```bash
cd /Users/pt/code/tools/evals
git worktree add ../evals-demo
cd ../evals-demo
python -m pip install -e ".[dev]"
```

After practicing, reset the prompt and generated runs:

```bash
git checkout -- examples/prompts/returns_policy_weak.md
rm -rf examples/runs
```

To delete the demo worktree when you are done:

```bash
cd /Users/pt/code/tools/evals
git worktree remove ../evals-demo
```

Set your API key:

```bash
export OPENAI_API_KEY="..."
```

If `prism run` says `run` is not a valid command, the shell is using an older
installed console script. Re-run the editable install above, or use the checkout
directly:

```bash
PYTHONPATH=src python -m prism_evals run examples/returns_policy_demo.py
```

## Run

```bash
prism run examples/returns_policy_demo.py
```

By default, the demo uses `gpt-5.4-mini` with low reasoning so the weak prompt is
more likely to miss policy details. To run a different model:

```bash
PRISM_RETURNS_MODEL_KEYS=gpt55_low prism run examples/returns_policy_demo.py
```

## Live Iteration

After the first run, inspect failures for:

- `decision_exact_match`: exact label match.
- `llm_policy_judge`: LLM-as-judge score using the response explanation.

A useful live prompt fix is to add:

```text
Before deciding, check delivery timing, item condition, proof of purchase,
final-sale status, damage status, and whether photos are required. If any
required detail is missing, choose NEEDS_INFO instead of guessing. Never promise
a refund, exchange, free shipping, or store credit unless the policy explicitly
allows it. Mention the concrete next step.
```
