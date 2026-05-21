from __future__ import annotations

import base64
import csv
import json

import pytest

from prism_evals import EvalResult, Experiment, ModelConfig, TaskOutput, ToolArgsEqual, ToolCalled
from prism_evals.runner import item_run_id, load_dataset


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
        return TaskOutput(text=response.output_text)

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
    assert (run_dir / "turns.csv").exists()
    assert (run_dir / "tool_calls.csv").exists()


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
        return TaskOutput(text="alpha")

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
        return TaskOutput(text="alpha")

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
    exp.workflow = lambda item, model, ctx: TaskOutput(text="alpha")
    exp.eval("correct", lambda item, model, output, ctx: output.text == item["expected"])

    records = exp.run()
    assert records[0].status == "success"
    assert records[0].evals[0].score is True


def test_workflow_object_is_supported(tmp_path, fake_client):
    class Workflow:
        async def __call__(self, item, model, ctx):
            return TaskOutput(text="alpha")

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


def test_workflow_must_return_task_output(tmp_path, fake_client):
    exp = Experiment(
        name="strict_workflow",
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

    records = exp.run()

    assert records[0].status == "failed"
    assert "must return prism_evals.TaskOutput" in records[0].error.message
    assert "TaskOutput(text=...)" in records[0].error.message


def test_step_must_return_task_output(tmp_path, fake_client):
    exp = Experiment(
        name="strict_step",
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
        return await ctx.step("draft", "alpha")

    exp.workflow = workflow
    records = exp.run()

    assert records[0].status == "failed"
    assert records[0].steps[0].status == "failed"
    assert "step 'draft' must return prism_evals.TaskOutput" in records[0].steps[0].error.message


def test_task_output_media_is_persisted_to_jsonl_and_csv(tmp_path, fake_client):
    exp = Experiment(
        name="media",
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
        image = ctx.media.from_base64(
            base64.b64encode(b"image bytes").decode("ascii"),
            format="png",
            name="sample",
            metadata={"alt": "Sample"},
        )
        return TaskOutput(text="generated", media=[image])

    exp.workflow = workflow
    records = exp.run()
    record = records[0]

    assert record.status == "success"
    assert record.output.media[0].path.startswith("media/")
    assert record.output.media[0].metadata == {"alt": "Sample"}
    assert (exp.run_dir() / record.output.media[0].path).read_bytes() == b"image bytes"

    jsonl_record = json.loads((exp.run_dir() / "results.jsonl").read_text(encoding="utf-8").strip())
    assert jsonl_record["output"]["media"][0]["path"] == record.output.media[0].path

    with (exp.run_dir() / "results.csv").open("r", encoding="utf-8") as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["media_count"] == "1"
    assert json.loads(row["media_paths_json"]) == [record.output.media[0].path]
    assert row["primary_media_path"] == record.output.media[0].path


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
            return TaskOutput(text=response.output_text)

        draft = await ctx.step(
            "draft",
            make_draft,
            evals=[("draft_has_text", lambda item, model, output, ctx: bool(output.text))],
        )
        final = await ctx.step(
            "final",
            lambda: TaskOutput(text=draft.text, value={"answer": draft.text}),
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
        await ctx.step("same", TaskOutput(text="first"))
        await ctx.step("same", TaskOutput(text="second"))

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
        return await ctx.step("checked", TaskOutput(text="alpha"), evals=[("broken", broken)])

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
        return await ctx.step("checked", TaskOutput(text="alpha"), evals=[("broken", broken)])

    exp.workflow = workflow
    with pytest.raises(RuntimeError, match="step eval failed"):
        exp.run()

    run_dir = exp.run_dir()
    assert (run_dir / "results.jsonl").exists()


def test_variant_models_are_available_by_role(tmp_path, fake_client):
    exp = Experiment(
        name="variants",
        dataset=write_dataset(
            tmp_path,
            [{"id": "a", "question": "alpha", "expected": "alpha"}],
        ),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.variant(
        "candidate",
        models={
            "router": "gpt-router",
            "support": {"model": "gpt-support", "params": {"temperature": 0}},
        },
        default_role="support",
    )

    async def workflow(item, model, ctx):
        router = ctx.model("router")
        support = ctx.model("support")
        assert model.key == "candidate"
        return TaskOutput(
            text="ok",
            value={
                "router": router.model,
                "support": support.model,
                "default": ctx.model.model,
                "support_temperature": support.params["temperature"],
            },
        )

    exp.workflow = workflow
    records = exp.run()

    assert records[0].status == "success"
    assert records[0].model_key == "candidate"
    assert records[0].variant_key == "candidate"
    assert records[0].output.value == {
        "router": "gpt-router",
        "support": "gpt-support",
        "default": "gpt-support",
        "support_temperature": 0,
    }
    manifest = json.loads((exp.run_dir() / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant_configs"][0]["key"] == "candidate"


def test_turns_and_tool_calls_are_recorded_and_scored(tmp_path, fake_client):
    exp = Experiment(
        name="scenario",
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
        async with ctx.conversation({"id": "conv-a"}) as convo:
            convo.user("user_start", "I need paper rolls")
            convo.assistant_seed("assistant_prior", "I can help with that.")

            async def run_ordering_turn():
                ctx.record_tool_call(
                    "start_or_update_paper_roll_order",
                    arguments={"quantity": 2},
                    result={"status": "needs_confirmation"},
                    agent="ordering",
                )
                return TaskOutput(text="Please confirm the order.", value={"route": "ordering"})

            await convo.turn(
                "order_turn",
                run_ordering_turn,
                evals=[
                    ("called_order_tool", ToolCalled("start_or_update_paper_roll_order", turn="order_turn")),
                    (
                        "quantity_matches",
                        ToolArgsEqual("start_or_update_paper_roll_order", "quantity", 2, turn="order_turn"),
                    ),
                ],
            )
            return convo.task_output()

    exp.workflow = workflow
    exp.eval("order_tool_called", ToolCalled("start_or_update_paper_roll_order", turn="order_turn"))

    records = exp.run()
    record = records[0]

    assert record.status == "success"
    assert [turn.id for turn in record.turns] == ["user_start", "assistant_prior", "order_turn"]
    assert record.turns[1].role == "assistant"
    assert record.turns[1].mode == "seed"
    assert record.tool_calls[0].name == "start_or_update_paper_roll_order"
    assert record.tool_calls[0].turn_id == "order_turn"
    assert record.evals[0].score is True
    assert record.steps[0].key == "turn:order_turn"
    assert record.steps[0].tool_calls[0].name == "start_or_update_paper_roll_order"
    assert record.steps[0].evals[0].score is True
    assert record.steps[0].evals[1].score is True

    with (exp.run_dir() / "turns.csv").open("r", encoding="utf-8") as handle:
        turn_rows = list(csv.DictReader(handle))
    assert [row["turn_id"] for row in turn_rows] == ["user_start", "assistant_prior", "order_turn"]

    with (exp.run_dir() / "tool_calls.csv").open("r", encoding="utf-8") as handle:
        tool_rows = list(csv.DictReader(handle))
    assert tool_rows[0]["name"] == "start_or_update_paper_roll_order"
    assert json.loads(tool_rows[0]["arguments_json"]) == {"quantity": 2}
