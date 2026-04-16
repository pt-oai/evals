from __future__ import annotations

import json

import pytest

from prism_evals import Experiment, ModelConfig
from prism_evals.storage import Storage


def write_dataset(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question,expected\n1,hello,hello\n", encoding="utf-8")
    return dataset


def make_experiment(tmp_path, fake_client, artifacts):
    exp = Experiment(
        name="artifact_eval",
        dataset=write_dataset(tmp_path),
        output_dir=tmp_path / "runs",
        artifacts=artifacts,
        base_dir=tmp_path,
        openai_client=fake_client,
        display="quiet",
    )
    exp.model(ModelConfig(key="m1", model="gpt-test"))
    exp.workflow = lambda item, model, ctx: "ok"
    return exp


def test_literal_artifact_is_copied_and_recorded_in_manifest(tmp_path, fake_client):
    prompt = tmp_path / "prompts" / "system.md"
    prompt.parent.mkdir()
    prompt.write_text("Answer tersely.", encoding="utf-8")

    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts/system.md"])
    exp.run()

    copied = exp.run_dir() / "artifacts" / "prompts" / "system.md"
    assert copied.read_text(encoding="utf-8") == "Answer tersely."
    assert (exp.run_dir() / "results.jsonl").exists()
    assert Storage(exp.run_dir()).artifact_paths()["artifacts"] == exp.run_dir() / "artifacts"

    manifest = json.loads((exp.run_dir() / "manifest.json").read_text(encoding="utf-8"))
    artifact = manifest["metadata"]["copied_artifacts"][0]
    assert artifact["spec"] == "prompts/system.md"
    assert artifact["source_path"] == str(prompt)
    assert artifact["destination_path"] == str(copied)
    assert artifact["destination_relative_path"] == "artifacts/prompts/system.md"
    assert artifact["sha256"]


def test_glob_artifacts_are_copied_with_relative_paths(tmp_path, fake_client):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a.md").write_text("A", encoding="utf-8")
    (prompts / "b.md").write_text("B", encoding="utf-8")

    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts/*.md"])
    exp.run()

    assert (exp.run_dir() / "artifacts" / "prompts" / "a.md").read_text(encoding="utf-8") == "A"
    assert (exp.run_dir() / "artifacts" / "prompts" / "b.md").read_text(encoding="utf-8") == "B"


def test_absolute_artifact_copies_by_basename(tmp_path, fake_client):
    prompt = tmp_path / "external.md"
    prompt.write_text("External", encoding="utf-8")

    exp = make_experiment(tmp_path, fake_client, artifacts=[prompt])
    exp.run()

    assert (exp.run_dir() / "artifacts" / "external.md").read_text(encoding="utf-8") == "External"


def test_missing_literal_artifact_fails_early(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts/missing.md"])

    with pytest.raises(ValueError, match="artifact file not found"):
        exp.run()


def test_unmatched_glob_artifact_fails_early(tmp_path, fake_client):
    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts/*.md"])

    with pytest.raises(ValueError, match="artifact glob matched no files"):
        exp.run()


def test_directory_artifact_fails_early(tmp_path, fake_client):
    (tmp_path / "prompts").mkdir()
    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts"])

    with pytest.raises(ValueError, match="artifact spec matched a directory"):
        exp.run()


def test_duplicate_artifact_destinations_fail_early(tmp_path, fake_client):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system.md").write_text("System", encoding="utf-8")
    exp = make_experiment(tmp_path, fake_client, artifacts=["prompts/system.md", "prompts/*.md"])

    with pytest.raises(ValueError, match="duplicate artifact destination"):
        exp.run()
