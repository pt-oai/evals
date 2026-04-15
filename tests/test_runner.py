from __future__ import annotations

import csv

import pytest

from evals import EvalResult, Experiment, ModelConfig
from evals.runner import item_run_id, load_dataset


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

    async def workflow(item, model, ctx):
        response = await ctx.responses.create(
            model=model.model,
            **model.params,
            input=item["question"],
        )
        return response.output_text

    def contains_expected(item, model, output, ctx):
        return item["expected"] in output.text

    async def brevity(item, model, output, ctx):
        return EvalResult(score=min(1.0, 100 / max(len(output.text), 1)))

    exp.workflow = workflow
    exp.eval("contains_expected", contains_expected, description="Expected appears")
    exp.eval("brevity", brevity)

    return exp


def test_load_dataset_uses_id_or_index(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question\n,first\nitem-2,second\n", encoding="utf-8")
    items = load_dataset(dataset)
    assert items[0].item_id == "0"
    assert items[1].item_id == "item-2"


def test_runner_executes_item_model_matrix_and_writes_artifacts(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client)
    records = exp.run()

    assert len(records) == 4
    assert len(fake_client.calls) == 4
    assert all(record.status == "success" for record in records)
    assert all(record.usage.input_tokens == 3 for record in records)
    assert all(record.usage.cached_tokens == 1 for record in records)
    assert all(record.usage.output_tokens == 5 for record in records)
    assert all(record.usage.reasoning_tokens == 2 for record in records)

    run_dir = exp.run_dir()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "results.csv").exists()
    assert (run_dir / "scores.csv").exists()
    assert (run_dir / "steps.csv").exists()


def test_item_run_id_is_deterministic(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client)
    item = load_dataset(exp.dataset)[0]
    model = exp.registered_models[0]
    assert item_run_id(exp, item, model, 0) == item_run_id(exp, item, model, 0)
    assert item_run_id(exp, item, model, 0) != item_run_id(exp, item, model, 1)


def test_resume_skips_successful_records(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client, resume=True)
    first = exp.run()
    assert len(first) == 4
    assert len(fake_client.calls) == 4

    second = exp.run()
    assert second == []
    assert len(fake_client.calls) == 4


def test_failed_workflow_records_error_and_continues(tmp_path, fake_client):
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

    async def workflow(item, model, ctx):
        return "alpha"

    def bundle(item, model, output, ctx):
        return {"correct": True, "score": 0.5}

    exp.workflow = workflow
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

    async def workflow(item, model, ctx):
        return "alpha"

    def broken(item, model, output, ctx):
        raise RuntimeError("nope")

    exp.workflow = workflow
    exp.eval("broken", broken)

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].key == "broken"
    assert records[0].evals[0].error is not None


def test_sync_workflow_callable_is_supported(tmp_path, fake_client):
    exp = Experiment(
        name="sync_workflow",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    exp.workflow = lambda item, model, ctx: "alpha"
    exp.eval("correct", lambda item, model, output, ctx: output.text == item["expected"])

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].score is True


def test_workflow_object_is_supported(tmp_path, fake_client):
    class Workflow:
        async def __call__(self, item, model, ctx):
            return "alpha"

    exp = Experiment(
        name="workflow_object",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    exp.workflow = Workflow()
    exp.eval("correct", lambda item, model, output, ctx: output.text == item["expected"])

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].score is True


def test_item_run_records_multiple_steps_and_step_evals(tmp_path, fake_client):
    exp = Experiment(
        name="steps",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "answer: alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def workflow(item, model, ctx):
        async def make_draft():
            response = await ctx.responses.create(model=model.model, input=item["question"])
            return response.output_text

        draft = await ctx.step(
            "draft",
            make_draft,
            evals=[("draft_has_text", lambda item, model, output, ctx: bool(output.text))],
        )
        final = await ctx.step(
            "final",
            lambda: {"text": draft.text, "value": {"answer": draft.text}},
            evals=[("final_matches", lambda item, model, output, ctx: output.text == item["expected"])],
        )
        return final

    exp.workflow = workflow
    records = exp.run()
    record = records[0]

    assert record.status == "success"
    assert record.item_id == "a"
    assert record.output.text == "answer: alpha"
    assert [step.key for step in record.steps] == ["draft", "final"]
    assert record.steps[0].evals[0].key == "draft_has_text"
    assert record.steps[0].evals[0].score is True
    assert record.steps[0].usage.input_tokens == 3
    assert len(record.steps[0].generations) == 1
    assert record.steps[1].evals[0].score is True
    assert record.usage.input_tokens == 3

    run_dir = exp.run_dir()
    with (run_dir / "results.csv").open("r", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    assert result_rows[0]["step:draft.score:draft_has_text"] == "True"
    assert result_rows[0]["step:final.score:final_matches"] == "True"

    with (run_dir / "scores.csv").open("r", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    assert {(row["scope"], row["step_key"], row["score_key"]) for row in score_rows} == {
        ("step", "draft", "draft_has_text"),
        ("step", "final", "final_matches"),
    }

    with (run_dir / "steps.csv").open("r", encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    assert [row["step_key"] for row in step_rows] == ["draft", "final"]
    assert step_rows[0]["input_tokens"] == "3"


def test_duplicate_step_key_fails_item_run(tmp_path, fake_client):
    exp = Experiment(
        name="dup_step",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def workflow(item, model, ctx):
        await ctx.step("same", "first")
        await ctx.step("same", "second")

    exp.workflow = workflow
    records = exp.run()
    assert records[0].status == "failed"
    assert "duplicate step key" in records[0].error.message


def test_step_callable_failure_records_failed_step(tmp_path, fake_client):
    exp = Experiment(
        name="step_failure",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def workflow(item, model, ctx):
        await ctx.step("broken", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    exp.workflow = workflow
    records = exp.run()
    assert records[0].status == "failed"
    assert records[0].steps[0].key == "broken"
    assert records[0].steps[0].status == "failed"
    assert records[0].steps[0].error.message == "boom"


def test_step_eval_failure_is_recorded_without_failing_item_run(tmp_path, fake_client):
    exp = Experiment(
        name="step_eval_failure",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    def broken(item, model, output, ctx):
        raise RuntimeError("eval boom")

    async def workflow(item, model, ctx):
        return await ctx.step("checked", "alpha", evals=[("broken", broken)])

    exp.workflow = workflow
    records = exp.run()
    assert records[0].status == "success"
    assert records[0].steps[0].status == "success"
    assert records[0].steps[0].evals[0].error.message == "eval boom"


def test_step_eval_failure_fails_item_run_when_fail_fast(tmp_path, fake_client):
    exp = Experiment(
        name="step_eval_fail_fast",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
        fail_fast=True,
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    def broken(item, model, output, ctx):
        raise RuntimeError("eval boom")

    async def workflow(item, model, ctx):
        return await ctx.step("checked", "alpha", evals=[("broken", broken)])

    exp.workflow = workflow
    with pytest.raises(RuntimeError, match="step eval failed"):
        exp.run()

    run_dir = exp.run_dir()
    assert (run_dir / "results.jsonl").exists()
