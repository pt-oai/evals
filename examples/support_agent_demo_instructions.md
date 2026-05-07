# Support Agent Demo Instructions

This demo shows a two-step support agent eval: a guardrail prompt routes the customer request, then a workload prompt writes the final reply. First, run the guardrail by itself, edit one sentence live, and rerun to show `route_accuracy` improve; then run the full agent to show `end_to_end_success` improves because the second step receives better routing.

## Setup

Recommended: use a separate demo worktree so generated runs and live prompt edits do not clutter your dev checkout.

```bash
cd /Users/pt/code/tools/evals
git worktree add ../evals-demo
cd ../evals-demo
python -m pip install -e ".[dev]"
```

After practicing, reset the prompt and generated runs:

```bash
git checkout -- examples/prompts/support_guardrail.md
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

If `prism run` says `run` is not a valid command, the shell is using an older installed console script. Re-run the editable install above, or use the checkout directly:

```bash
PYTHONPATH=src python -m prism_evals run examples/support_guardrail_demo.py
```

## 1. Iterate On The Guardrail Prompt Only

```bash
prism run examples/support_guardrail_demo.py
```

Open `examples/prompts/support_guardrail.md` and replace this weak sentence:

```text
If a customer sounds angry, threatens a dispute, says "scam", mentions a stolen card, or asks for coercive language, route to blocked.
```

with this better sentence:

```text
Only use blocked when the requested action itself is unsafe, illegal, abusive, or coercive; route angry but serviceable requests to the right support team.
```

Then rerun the same command:

```bash
prism run examples/support_guardrail_demo.py
```

In the viewer, compare the two `support_guardrail_demo` runs by timestamp. This is the fast loop for improving the first prompt without paying for the workload step. Each run copies the current `support_guardrail.md` into `artifacts/`, so you can show exactly what changed.

## 2. Run The Full Agent Chain

```bash
prism run examples/support_agent_demo.py
```

If you want the before/after full-chain story, run this once before the prompt edit and once after the prompt edit. In the viewer, compare the two `support_agent_demo` runs by timestamp. Show `guardrail_triage / route_accuracy` first, then show `end_to_end_success` to connect the prompt improvement to the full agent score.

## Open The Viewer

```bash
prism view examples/runs
```

The model lanes also show which model/config gives the best quality, latency, and token tradeoff.

## Quick Smoke Run

To test one model before running the full matrix:

```bash
PRISM_SUPPORT_MODEL_KEYS=gpt55_low prism run examples/support_guardrail_demo.py
PRISM_SUPPORT_MODEL_KEYS=gpt55_low prism run examples/support_agent_demo.py
```

Available model keys:

- `gpt55_low`
- `gpt55_medium`
- `gpt54_mini_low`

## Troubleshooting

- Missing API key: set `OPENAI_API_KEY` in the shell where you run `prism`.
- Unavailable model slug: edit the model list in the demo file you are running, or run with `PRISM_SUPPORT_MODEL_KEYS` to use a model that is available to your account.
- Scores look unexpected: open the generated run directory under `examples/runs/` and inspect `results.csv`, `scores.csv`, and `steps.csv`.
- Prompt provenance: each run copies the active guardrail prompt and workload prompt into `artifacts/`.
