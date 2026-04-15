from __future__ import annotations

import csv
import json
import re

import pytest

from evals import (
    ApproxEqual,
    Contains,
    Equal,
    EvalResult,
    Experiment,
    JsonPathEqual,
    JsonPathExists,
    LengthBetween,
    ModelConfig,
    NonEmpty,
    NotEqual,
    RegexMatch,
    TaskOutput,
    out,
    row,
    text,
)


def call(evaluator, *, dataset_row=None, output=None):
    dataset_row = dataset_row or {"score": "0.5", "expected": "hello", "term": "WORLD"}
    output = output or TaskOutput(
        text="Hello world [1]",
        value={
            "score": 0.5,
            "nested": {"items": [{"name": "first"}]},
            "labels": ["alpha", "beta"],
        },
    )
    return evaluator(dataset_row, model=None, output=output, ctx=None)


def write_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question,expected,score,term\n1,hello,hello,0.5,world\n", encoding="utf-8")
    return dataset


def test_equal_and_not_equal():
    assert call(Equal(actual=out("score"), expected=row("score", cast=float))).score is True
    assert call(Equal(actual=out("score"), expected=row("score"))).score is False
    assert call(NotEqual(actual=out("score"), expected=row("score"))).score is True


def test_approx_equal_and_non_numeric_failure():
    result = call(ApproxEqual(actual=out("score"), expected=0.51, abs_tol=0.02))
    assert result.score is True
    failed = call(ApproxEqual(actual=row("expected"), expected=0.5))
    assert failed.score is False
    assert "numeric" in failed.comment


def test_contains_for_strings_lists_and_dicts():
    assert call(Contains(container=text(), expected=row("term"), case_sensitive=False)).score is True
    assert call(Contains(container=out("labels"), expected="alpha")).score is True
    assert call(Contains(container={"score": 1}, expected="score")).score is True
    assert call(Contains(container=text(), expected=row("term"), case_sensitive=True)).score is False


def test_regex_non_empty_and_length_between():
    assert call(RegexMatch(value=text(), pattern=r"\[\d+\]")).score is True
    assert call(RegexMatch(value=text(), pattern=r"hello", flags=re.I)).score is True
    assert call(NonEmpty(value=text())).score is True
    assert call(NonEmpty(value="   ")).score is False
    assert call(LengthBetween(value=text(), min_len=5, max_len=20)).score is True
    assert call(LengthBetween(value=text(), max_len=3)).score is False
    assert call(LengthBetween(value=5, min_len=1)).score is False


def test_length_between_validates_bounds():
    with pytest.raises(ValueError, match="requires"):
        LengthBetween(value=text())
    with pytest.raises(ValueError, match="min_len"):
        LengthBetween(value=text(), min_len=5, max_len=2)


def test_json_path_helpers_support_dicts_lists_and_strings():
    assert call(JsonPathExists(value=out(), path="nested.items.0.name")).score is True
    assert call(JsonPathEqual(value=out(), path="nested.items.0.name", expected="first")).score is True
    json_output = TaskOutput(text="", value=json.dumps({"person": {"name": "Ada"}}))
    assert call(JsonPathExists(value=out(), path="person.name"), output=json_output).score is True
    assert call(JsonPathEqual(value=out(), path="person.name", expected="Ada"), output=json_output).score is True
    assert call(JsonPathExists(value=out(), path="person.age"), output=json_output).score is False
    assert call(JsonPathExists(value=out(), path="person.name"), output=TaskOutput(value="{")).score is False


def test_selectors_handle_nested_paths_casts_defaults_and_failures():
    assert call(Equal(actual=out("nested.items.0.name"), expected="first")).score is True
    assert call(Equal(actual=row("score", cast=float), expected=0.5)).score is True
    assert call(Equal(actual=row("missing", default="fallback"), expected="fallback")).score is True
    missing = call(Equal(actual=out("missing.path"), expected="x"))
    assert missing.score is False
    assert "missing" in missing.comment
    cast_failed = call(Equal(actual=row("expected", cast=float), expected=1.0))
    assert cast_failed.score is False
    assert "cast" in cast_failed.comment


def test_experiment_eval_registers_direct_custom_functions_and_builtins(tmp_path, fake_client):
    exp = Experiment(
        name="registration",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(dataset_row, model, ctx):
        return TaskOutput(text="hello", value={"score": 0.5})

    def direct_custom(dataset_row, model, output, ctx):
        return True

    exp.task = task
    exp.eval("direct_custom", direct_custom)
    exp.eval("builtin_equal", Equal(actual=out("score"), expected=row("score", cast=float)))
    exp.eval("inline_custom", lambda dataset_row, model, output, ctx: True)

    records = exp.run()
    keys = {result.key for result in records[0].evals}
    assert keys == {"direct_custom", "builtin_equal", "inline_custom"}


def test_duplicate_eval_keys_are_rejected_across_registration_styles(tmp_path):
    exp = Experiment(name="dup", dataset=write_dataset(tmp_path), output_dir=tmp_path / "runs")
    exp.eval("same", Equal(actual=1, expected=1))
    with pytest.raises(ValueError, match="duplicate eval key"):
        exp.eval("same", Equal(actual=1, expected=1))


def test_registered_key_overrides_single_eval_result_key(tmp_path, fake_client):
    exp = Experiment(
        name="override",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(dataset_row, model, ctx):
        return "ok"

    exp.task = task
    exp.eval("registered", lambda dataset_row, model, output, ctx: EvalResult(key="other", score=True))

    records = exp.run()
    assert records[0].evals[0].key == "registered"


def test_dict_and_list_eval_returns_keep_multi_score_behavior(tmp_path, fake_client):
    exp = Experiment(
        name="multi",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(dataset_row, model, ctx):
        return "ok"

    exp.task = task
    exp.eval("dict_bundle", lambda dataset_row, model, output, ctx: {"a": True, "b": 0.5})
    exp.eval(
        "list_bundle",
        lambda dataset_row, model, output, ctx: [
            EvalResult(score=True),
            EvalResult(key="explicit", score=False),
        ],
    )

    records = exp.run()
    keys = {result.key for result in records[0].evals}
    assert keys == {"a", "b", "list_bundle", "explicit"}


def test_builtin_scores_are_persisted_to_csv_files(tmp_path, fake_client):
    exp = Experiment(
        name="persist",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))

    async def task(dataset_row, model, ctx):
        return TaskOutput(text="hello world", value={"score": 0.5})

    exp.task = task
    exp.eval("score_equal", Equal(actual=out("score"), expected=row("score", cast=float)))
    records = exp.run()
    assert records[0].evals[0].score is True

    run_dir = tmp_path / "runs" / "persist"
    with (run_dir / "results.csv").open("r", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    assert result_rows[0]["score:score_equal"] == "True"

    with (run_dir / "scores.csv").open("r", encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    assert score_rows[0]["score_key"] == "score_equal"
    assert score_rows[0]["score"] == "True"
