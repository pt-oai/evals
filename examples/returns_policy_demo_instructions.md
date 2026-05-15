# Returns Policy Demo

This demo teaches evals with one policy-answering prompt and one small dataset.
The first prompt is intentionally weak so a live run should surface failures that
you can fix by editing `examples/prompts/returns_policy_weak.md`.

## Files

- `examples/prompts/returns_policy.md`: short policy to read with the user.
- `examples/datasets/returns_policy_demo.csv`: synthetic cases with expected
  decisions, required terms, and forbidden terms.
- `examples/prompts/returns_policy_weak.md`: starter prompt to improve live.
- `examples/returns_policy_demo.py`: runnable Prism experiment.

## Run

```bash
prism run examples/returns_policy_demo.py
```

If the installed CLI is stale, run from this checkout:

```bash
PYTHONPATH=src python -m prism_evals run examples/returns_policy_demo.py
```

By default, the demo uses `gpt-5.4-mini` with low reasoning so the weak prompt is
more likely to miss policy details. To run a different model:

```bash
PRISM_RETURNS_MODEL_KEYS=gpt55_low prism run examples/returns_policy_demo.py
```

## Live Iteration

After the first run, inspect failures for:

- `decision_exact_match`: exact label match.
- `required_terms_present`: required phrase includes.
- `forbidden_terms_absent`: forbidden promise checks.
- `required_term_coverage`: numeric coverage score.
- `llm_policy_judge`: LLM-as-judge quality score.

A useful live prompt fix is to add:

```text
Before deciding, check delivery timing, item condition, proof of purchase,
final-sale status, damage status, and whether photos are required. If any
required detail is missing, choose NEEDS_INFO instead of guessing. Never promise
a refund, exchange, free shipping, or store credit unless the policy explicitly
allows it. Mention the concrete next step.
```
