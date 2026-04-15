from __future__ import annotations

import pytest

from evals import Experiment, ModelConfig


def write_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question,expected\n1,hello,hello\n", encoding="utf-8")
    return dataset


def test_registers_task_model_and_eval(tmp_path, fake_client):
    exp = Experiment(
        name="demo",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    @exp.task
    async def task(row, model, ctx):
        return "ok"

    @exp.eval("always")
    def always(row, model, output, ctx):
        return True

    assert exp.task_fn is task
    assert exp.registered_models[0].key == "m1"
    assert exp.registered_evals[0].key == "always"


def test_rejects_duplicate_model_keys(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    with pytest.raises(ValueError, match="duplicate model key"):
        exp.model(ModelConfig(key="m1", model="gpt-test"))


def test_task_must_be_async(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)

    with pytest.raises(TypeError, match="async"):

        @exp.task
        def task(row, model, ctx):
            return "ok"


def test_validate_requires_task_and_model(tmp_path):
    exp = Experiment(name="demo", dataset=write_dataset(tmp_path), output_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one model"):
        exp.validate()

    exp.model(ModelConfig(key="m1", model="gpt-test"))
    with pytest.raises(ValueError, match="task function"):
        exp.validate()

