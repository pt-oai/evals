from __future__ import annotations

import pytest

from evals.cli import (
    ensure_viewer_dependencies,
    main,
    validate_runs_parent,
    viewer_dependencies_installed,
    viewer_dir,
)


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


def test_viewer_dir_uses_env_override(tmp_path, monkeypatch):
    override = tmp_path / "viewer"
    monkeypatch.setenv("PT_EVALS_VIEWER_DIR", str(override))

    assert viewer_dir() == override.resolve()


def test_ensure_viewer_dependencies_skips_existing_install(tmp_path, monkeypatch):
    app_dir = tmp_path / "viewer"
    bin_dir = app_dir / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "next").write_text("", encoding="utf-8")
    (app_dir / "node_modules" / "next").mkdir()
    (app_dir / "package.json").write_text('{"dependencies":{"next":"^15.0.0"}}', encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("npm install should not run")

    monkeypatch.setattr("evals.cli.subprocess.run", fail_run)

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

    monkeypatch.setattr("evals.cli.shutil.which", lambda name: "/usr/bin/npm" if name == "npm" else None)

    def fake_run(args, *, cwd, check):
        calls.append((args, cwd, check))

    monkeypatch.setattr("evals.cli.subprocess.run", fake_run)

    ensure_viewer_dependencies(app_dir)

    assert calls == [(["npm", "install"], app_dir, True)]
