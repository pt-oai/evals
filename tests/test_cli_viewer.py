from __future__ import annotations

import pytest

from evals.cli import main, validate_runs_parent


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
    assert "pt-evals view failed" in capsys.readouterr().err
