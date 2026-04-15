from __future__ import annotations

import csv

import pytest

from evals import EvalResult, Experiment, ModelConfig
from evals.runner import load_dataset


def write_dataset(tmp_path, rows):
    dataset = tmp_path / "data.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "expected"])
        writer.writeheader()
        writer.writerows(rows)
    return dataset


def make_experiment(tmp_path, fake_client, rows=None, **kwargs):
    rows = rows or [
        {"id": "a", "question": "alpha", "expected": "alpha"},
        {"id": "b", "question": "beta", "expected": "beta"},
    ]
    exp = Experiment(
        name="demo",
        dataset=write_dataset(tmp_path, rows),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
        **kwargs,
    )
    exp.models(
        [
            ModelConfig(key="m1", model="gpt-test", params={"temperature": 0}),
            ModelConfig(key="m2", model="gpt-test", params={"temperature": 1}),
        ]
    )

    async def task(row, model, ctx):
        response = await ctx.responses.create(
            model=model.model,
            **model.params,
            input=row["question"],
        )
        return response.output_text

    def contains_expected(row, model, output, ctx):
        return row["expected"] in output.text

    async def brevity(row, model, output, ctx):
        return EvalResult(score=min(1.0, 100 / max(len(output.text), 1)))

    exp.task = task
    exp.eval("contains_expected", contains_expected, description="Expected appears")
    exp.eval("brevity", brevity)

    return exp


def test_load_dataset_uses_id_or_index(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question\n,first\nrow-2,second\n", encoding="utf-8")
    rows = load_dataset(dataset)
    assert rows[0].row_id == "0"
    assert rows[1].row_id == "row-2"


def test_runner_executes_row_model_matrix_and_writes_artifacts(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client)
    records = exp.run()

    assert len(records) == 4
    assert len(fake_client.calls) == 4
    assert all(record.status == "success" for record in records)
    assert all(record.usage.input_tokens == 3 for record in records)
    assert all(record.usage.cached_tokens == 1 for record in records)
    assert all(record.usage.output_tokens == 5 for record in records)
    assert all(record.usage.reasoning_tokens == 2 for record in records)

    run_dir = tmp_path / "runs" / "demo"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "scores.csv").exists()


def test_resume_skips_successful_records(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client, resume=True)
    first = exp.run()
    assert len(first) == 4
    assert len(fake_client.calls) == 4

    second = exp.run()
    assert second == []
    assert len(fake_client.calls) == 4


def test_failed_task_records_error_and_continues(tmp_path, fake_client):
    exp = make_experiment(
        tmp_path,
        fake_client,
        rows=[
            {"id": "a", "question": "alpha", "expected": "alpha"},
            {"id": "b", "question": "fail please", "expected": "anything"},
        ],
    )
    records = exp.run()
    failures = [record for record in records if record.status == "failed"]
    successes = [record for record in records if record.status == "success"]

    assert len(failures) == 2
    assert len(successes) == 2
    assert all(record.error is not None for record in failures)


def test_eval_dict_return_is_supported(tmp_path, fake_client):
    exp = Experiment(
        name="dict_eval",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(row, model, ctx):
        return "alpha"

    def bundle(row, model, output, ctx):
        return {"correct": True, "score": 0.5}

    exp.task = task
    exp.eval("bundle", bundle)

    records = exp.run()
    assert {result.key for result in records[0].evals} == {"correct", "score"}


def test_eval_errors_are_recorded(tmp_path, fake_client):
    exp = Experiment(
        name="eval_error",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(row, model, ctx):
        return "alpha"

    def broken(row, model, output, ctx):
        raise RuntimeError("nope")

    exp.task = task
    exp.eval("broken", broken)

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].key == "broken"
    assert records[0].evals[0].error is not None


def test_sync_task_callable_is_supported(tmp_path, fake_client):
    exp = Experiment(
        name="sync_task",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    exp.task = lambda row, model, ctx: "alpha"
    exp.eval("correct", lambda row, model, output, ctx: output.text == row["expected"])

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].score is True


def test_task_object_is_supported(tmp_path, fake_client):
    class Task:
        async def __call__(self, row, model, ctx):
            return "alpha"

    exp = Experiment(
        name="task_object",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    exp.task = Task()
    exp.eval("correct", lambda row, model, output, ctx: output.text == row["expected"])

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].score is True
