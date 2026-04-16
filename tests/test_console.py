from __future__ import annotations

from prism_evals.console import aggregate_usage_by_model, collect_score_values, format_score_cell
from prism_evals.models import EvalResult, ItemRunRecord, StepRecord, TokenUsage


def record(
    model_key: str,
    usage: TokenUsage | None = None,
    *,
    evals: list[EvalResult] | None = None,
    steps: list[StepRecord] | None = None,
) -> ItemRunRecord:
    return ItemRunRecord(
        item_run_id=f"{model_key}-run",
        run_id="run",
        experiment_name="demo",
        item_index=0,
        item_id="item",
        item={},
        model_key=model_key,
        model="gpt-test",
        repetition=0,
        status="success",
        started_at="2026-04-15T00:00:00Z",
        ended_at="2026-04-15T00:00:01Z",
        duration_s=1.0,
        usage=usage or TokenUsage(),
        evals=evals or [],
        steps=steps or [],
    )


def test_aggregate_usage_by_model_tracks_all_token_types():
    stats = aggregate_usage_by_model(
        [
            record(
                "m1",
                TokenUsage(
                    input_tokens=10,
                    cached_tokens=2,
                    output_tokens=6,
                    reasoning_tokens=3,
                    total_tokens=16,
                ),
            ),
            record(
                "m1",
                TokenUsage(
                    input_tokens=20,
                    cached_tokens=4,
                    output_tokens=8,
                    reasoning_tokens=5,
                    total_tokens=28,
                ),
            ),
        ]
    )

    assert stats["m1"].count == 2
    assert stats["m1"].totals == [30, 6, 14, 8, 44]
    assert [value / stats["m1"].count for value in stats["m1"].totals] == [15, 3, 7, 4, 22]


def test_collect_score_values_includes_item_run_and_step_scores_by_model():
    scores = collect_score_values(
        [
            record("m1", evals=[EvalResult(key="exact", score=True)]),
            record("m2", evals=[EvalResult(key="exact", score=False)]),
            record(
                "m1",
                steps=[
                    StepRecord(
                        key="draft",
                        status="success",
                        started_at="2026-04-15T00:00:00Z",
                        ended_at="2026-04-15T00:00:01Z",
                        duration_s=1.0,
                        evals=[EvalResult(key="non_empty", score=True)],
                    )
                ],
            ),
        ]
    )

    assert scores[("item_run", "", "m1", "exact")] == [1.0]
    assert scores[("item_run", "", "m2", "exact")] == [0.0]
    assert scores[("step", "draft", "m1", "non_empty")] == [1.0]
    assert format_score_cell([1.0, 0.0]) == "0.500 (2)"
    assert format_score_cell([]) == "-"
