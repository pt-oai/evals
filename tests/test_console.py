from __future__ import annotations

from evals.console import aggregate_usage_by_model
from evals.models import ItemRunRecord, TokenUsage


def record(model_key: str, usage: TokenUsage) -> ItemRunRecord:
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
        usage=usage,
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
