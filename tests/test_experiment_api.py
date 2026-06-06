from __future__ import annotations

import re

import pytest

from prism_evals import Experiment, ModelConfig, TaskOutput


def write_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question,expected\n1,hello,hello\n", encoding="utf-8")
    return dataset


def test_registers_workflow_model_and_eval(tmp_path, fake_client):
    exp = Experiment(
        name="demo",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def workflow(item, model, ctx):
        return TaskOutput(text="ok")

    def always(item, model, output, ctx):
        return True

    exp.workflow = workflow
    exp.eval("always", always)

    assert exp.workflow is workflow
    assert exp.registered_models[0].key == "m1"
    assert exp.registered_evals[0].key == "always"


def test_rejects_duplicate_model_keys(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    with pytest.raises(ValueError, match="duplicate model key"):
        exp.model(ModelConfig(key="m1", model="gpt-test"))


def test_run_dir_is_timestamp_prefixed_by_default(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path / "runs")
    assert exp.run_dir().parent == tmp_path / "runs"
    assert re.fullmatch(r"\d{8}-\d{6}_demo", exp.run_dir().name)
    assert exp.run_dir() == exp.run_dir()


def test_run_dir_timestamp_prefix_can_be_disabled(tmp_path):
    exp = Experiment(
        name="demo",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        timestamp_output_dir=False,
    )
    assert exp.run_dir() == tmp_path / "runs" / "demo"


def test_workflow_must_be_callable(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)

    with pytest.raises(TypeError, match="callable"):
        exp.workflow = "not callable"


def test_pass_condition_must_be_callable(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)

    with pytest.raises(TypeError, match="pass_condition must be callable"):
        exp.pass_condition = "not callable"


def test_task_output_serializes_text_value_media():
    output = TaskOutput(
        text="hello",
        value={"answer": "world"},
        media=[
            {
                "path": "media/example.png",
                "mime_type": "image/png",
                "format": "png",
                "sha256": "abc",
                "bytes": 3,
            }
        ],
    )

    payload = output.model_dump(mode="json")

    assert payload["text"] == "hello"
    assert payload["value"] == {"answer": "world"}
    assert payload["media"][0]["path"] == "media/example.png"


def test_eval_requires_callable(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)

    with pytest.raises(TypeError, match="callable"):
        exp.eval("bad", "not callable")


def test_validate_requires_workflow_and_model(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one model"):
        exp.validate()

    exp.model(ModelConfig(key="m1", model="gpt-test"))
    with pytest.raises(ValueError, match="workflow callable"):
        exp.validate()
