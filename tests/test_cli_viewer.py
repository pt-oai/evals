from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from prism_evals.cli import (
    discover_experiments,
    ensure_viewer_dependencies,
    latest_viewer_tag,
    load_experiment_module,
    newest_version_tag,
    main,
    validate_runs_parent,
    version_key,
    version_tag,
    viewer_dependencies_installed,
    viewer_dir,
    viewer_version,
)
from prism_evals.experiment import Experiment


def test_validate_runs_parent_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="runs directory does not exist"):
        validate_runs_parent(tmp_path / "missing")


def test_validate_runs_parent_rejects_single_run_directory(tmp_path):
    run_dir = tmp_path / "20260415-120000_demo"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="pass the parent directory"):
        validate_runs_parent(run_dir)


def test_validate_runs_parent_requires_child_runs(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run folders"):
        validate_runs_parent(tmp_path)


def test_validate_runs_parent_accepts_parent_directory(tmp_path):
    run_dir = tmp_path / "20260415-120000_demo"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert validate_runs_parent(tmp_path) == tmp_path.resolve()


def test_cli_view_reports_validation_failures(tmp_path, capsys):
    result = main(["view", str(tmp_path)])

    assert result == 1
    assert "prism view failed" in capsys.readouterr().err


def test_project_scripts_expose_prism_aliases():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "prism": "prism_evals.cli:main",
        "prism-evals": "prism_evals.cli:main",
        "pe": "prism_evals.cli:main",
    }


def test_cli_run_discovers_and_runs_experiment_file(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question\n1,hello\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    experiment_file = tmp_path / "eval.py"
    experiment_file.write_text(
        f"""
from prism_evals import Experiment, ModelConfig, TaskOutput

exp = Experiment(
    name="cli_demo",
    dataset={str(dataset)!r},
    output_dir={str(runs_dir)!r},
    openai_client=object(),
    display="quiet",
    timestamp_output_dir=False,
)
exp.model(ModelConfig(key="m1", model="gpt-test"))

def workflow(item, model, ctx):
    return TaskOutput(text=item["question"])

exp.workflow = workflow
""",
        encoding="utf-8",
    )

    result = main(["run", str(experiment_file)])

    assert result == 0
    assert (runs_dir / "cli_demo" / "manifest.json").exists()
    assert (runs_dir / "cli_demo" / "results.csv").exists()


def test_cli_run_runs_multiple_experiments_in_declaration_order(tmp_path, monkeypatch):
    experiment_file = tmp_path / "evals.py"
    experiment_file.write_text(
        """
from prism_evals import Experiment

first = Experiment(name="first", dataset="missing.csv")
second = Experiment(name="second", dataset="missing.csv")
""",
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(Experiment, "run", lambda self: calls.append(self.name))

    result = main(["run", str(experiment_file)])

    assert result == 0
    assert calls == ["first", "second"]


def test_cli_run_deduplicates_experiment_aliases_and_skips_main_block(tmp_path, monkeypatch):
    experiment_file = tmp_path / "eval.py"
    experiment_file.write_text(
        """
from prism_evals import Experiment

exp = Experiment(name="once", dataset="missing.csv")
alias = exp

if __name__ == "__main__":
    raise RuntimeError("main block fired")
""",
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(Experiment, "run", lambda self: calls.append(self.name))

    result = main(["run", str(experiment_file)])

    assert result == 0
    assert calls == ["once"]


def test_cli_run_temporarily_adds_experiment_parent_to_sys_path(tmp_path, monkeypatch):
    (tmp_path / "helper.py").write_text('NAME = "from_helper"\n', encoding="utf-8")
    experiment_file = tmp_path / "eval.py"
    experiment_file.write_text(
        """
from helper import NAME
from prism_evals import Experiment

exp = Experiment(name=NAME, dataset="missing.csv")
""",
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(Experiment, "run", lambda self: calls.append(self.name))

    result = main(["run", str(experiment_file)])

    assert result == 0
    assert calls == ["from_helper"]


def test_cli_run_reports_missing_file(capsys):
    result = main(["run", "/tmp/prism-evals-missing-file.py"])

    assert result == 1
    captured = capsys.readouterr()
    assert "prism run failed" in captured.err
    assert "experiment file does not exist" in captured.err


def test_cli_run_reports_no_discovered_experiments(tmp_path, capsys):
    experiment_file = tmp_path / "empty.py"
    experiment_file.write_text("VALUE = 1\n", encoding="utf-8")

    result = main(["run", str(experiment_file)])

    assert result == 1
    captured = capsys.readouterr()
    assert "prism run failed" in captured.err
    assert "no module-level Experiment instances found" in captured.err


def test_cli_run_reports_invalid_experiment_configuration(tmp_path, capsys):
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,question\n1,hello\n", encoding="utf-8")
    experiment_file = tmp_path / "invalid.py"
    experiment_file.write_text(
        f"""
from prism_evals import Experiment

exp = Experiment(
    name="invalid",
    dataset={str(dataset)!r},
    output_dir={str(tmp_path / "runs")!r},
    display="quiet",
)
""",
        encoding="utf-8",
    )

    result = main(["run", str(experiment_file)])

    assert result == 1
    captured = capsys.readouterr()
    assert "prism run failed" in captured.err
    assert "at least one model" in captured.err


def test_discover_experiments_preserves_module_order_and_identity(tmp_path):
    experiment_file = tmp_path / "eval.py"
    experiment_file.write_text(
        """
from prism_evals import Experiment

first = Experiment(name="first", dataset="missing.csv")
second = Experiment(name="second", dataset="missing.csv")
again = first
""",
        encoding="utf-8",
    )

    module = load_experiment_module(experiment_file)

    assert [exp.name for exp in discover_experiments(module)] == ["first", "second"]


def test_viewer_dir_uses_env_override(tmp_path, monkeypatch):
    override = tmp_path / "viewer"
    monkeypatch.setenv("PRISM_VIEWER_DIR", str(override))

    assert viewer_dir() == override.resolve()


def test_viewer_version_prefers_viewer_package_json(tmp_path):
    app_dir = tmp_path / "viewer"
    app_dir.mkdir()
    (app_dir / "package.json").write_text('{"version":"9.9.9"}', encoding="utf-8")

    assert viewer_version(app_dir) == "9.9.9"


def test_ensure_viewer_dependencies_skips_existing_install(tmp_path, monkeypatch):
    app_dir = tmp_path / "viewer"
    bin_dir = app_dir / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "next").write_text("", encoding="utf-8")
    (app_dir / "node_modules" / "next").mkdir()
    (app_dir / "package.json").write_text('{"dependencies":{"next":"^15.0.0"}}', encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("npm install should not run")

    monkeypatch.setattr("prism_evals.cli.subprocess.run", fail_run)

    ensure_viewer_dependencies(app_dir)


def test_viewer_dependencies_installed_detects_missing_dependency(tmp_path):
    app_dir = tmp_path / "viewer"
    (app_dir / "node_modules" / ".bin").mkdir(parents=True)
    (app_dir / "node_modules" / ".bin" / "next").write_text("", encoding="utf-8")
    (app_dir / "node_modules" / "next").mkdir()
    (app_dir / "package.json").write_text(
        '{"dependencies":{"next":"^15.0.0","recharts":"^3.0.0"}}',
        encoding="utf-8",
    )

    assert not viewer_dependencies_installed(app_dir)


def test_ensure_viewer_dependencies_runs_npm_install(tmp_path, monkeypatch):
    app_dir = tmp_path / "viewer"
    app_dir.mkdir()
    calls = []

    monkeypatch.setattr("prism_evals.cli.shutil.which", lambda name: "/usr/bin/npm" if name == "npm" else None)

    def fake_run(args, *, cwd, check):
        calls.append((args, cwd, check))

    monkeypatch.setattr("prism_evals.cli.subprocess.run", fake_run)

    ensure_viewer_dependencies(app_dir)

    assert calls == [(["npm", "install"], app_dir, True)]


def test_version_tag_adds_v_prefix():
    assert version_tag("0.5.8") == "v0.5.8"
    assert version_tag("v0.5.8") == "v0.5.8"


def test_newest_version_tag_sorts_semver_tags():
    assert newest_version_tag(["v0.5.8", "v0.5.10", "not-a-version"]) == "v0.5.10"
    assert version_key("refs/tags/v1.2.3") == (1, 2, 3)


def test_latest_viewer_tag_uses_github_cli_for_release_tags(tmp_path, monkeypatch):
    app_dir = tmp_path / "viewer"

    def fake_git(args, *, timeout):
        if args == ["git", "-C", str(app_dir), "remote", "get-url", "origin"]:
            return ""
        if args == ["gh", "api", "repos/pt-oai/evals/tags", "--paginate", "--jq", ".[].name"]:
            return "v0.6.4\nv0.6.3\n"
        if args[:2] == ["git", "ls-remote"]:
            raise AssertionError("git remotes should not be used when gh returns release tags")
        return ""

    monkeypatch.delenv("PRISM_VIEWER_LATEST_TAG", raising=False)
    monkeypatch.delenv("PRISM_RELEASE_REPOSITORY", raising=False)
    monkeypatch.delenv("PRISM_RELEASE_GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr("prism_evals.cli.run_git_command", fake_git)

    assert latest_viewer_tag(app_dir, "v0.6.3") == "v0.6.4"


def test_latest_viewer_tag_uses_release_repo_when_viewer_is_inside_another_repo(tmp_path, monkeypatch):
    app_dir = tmp_path / "customer-repo" / ".venv" / "lib" / "python3.12" / "site-packages" / "prism_evals" / "viewer"
    app_dir.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_git(args, *, timeout):
        calls.append(list(args))
        if args == ["git", "-C", str(app_dir), "remote", "get-url", "origin"]:
            return "git@github.com:example/customer-evals.git\n"
        if args[:4] == ["git", "-C", str(app_dir), "tag"]:
            raise AssertionError("host repo tags should not be used for release checks")
        if args == ["git", "ls-remote", "--tags", "--refs", "git@github.com:pt-oai/evals.git", "v[0-9]*"]:
            return "abc123\trefs/tags/v0.6.4\n"
        return ""

    monkeypatch.delenv("PRISM_VIEWER_LATEST_TAG", raising=False)
    monkeypatch.delenv("PRISM_RELEASE_REPOSITORY", raising=False)
    monkeypatch.setattr("prism_evals.cli.run_git_command", fake_git)

    assert latest_viewer_tag(app_dir, "v0.6.3") == "v0.6.4"
    assert ["git", "ls-remote", "--tags", "--refs", "git@github.com:pt-oai/evals.git", "v[0-9]*"] in calls


def test_latest_viewer_tag_accepts_release_repository_override(tmp_path, monkeypatch):
    app_dir = tmp_path / "viewer"

    def fake_git(args, *, timeout):
        if args == ["git", "-C", str(app_dir), "remote", "get-url", "origin"]:
            return ""
        if args == ["git", "ls-remote", "--tags", "--refs", "ssh://example/release.git", "v[0-9]*"]:
            return "abc123\trefs/tags/v0.7.0\n"
        return ""

    monkeypatch.delenv("PRISM_VIEWER_LATEST_TAG", raising=False)
    monkeypatch.setenv("PRISM_RELEASE_REPOSITORY", "ssh://example/release.git")
    monkeypatch.setattr("prism_evals.cli.run_git_command", fake_git)

    assert latest_viewer_tag(app_dir, "v0.6.3") == "v0.7.0"
