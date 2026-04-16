from __future__ import annotations

import pytest

from prism_evals.cli import main
from prism_evals.scaffold import BEGIN_MARKER, END_MARKER, install_agents_md, load_agents_template


def test_load_agents_template_contains_result_guidance():
    template = load_agents_template()

    assert BEGIN_MARKER in template
    assert END_MARKER in template
    assert "This repo uses `prism-evals`" in template
    assert "# Prism Evals" in template
    assert "results.jsonl" in template
    assert "Where To Make Changes" in template


def test_install_agents_md_creates_file(tmp_path):
    path, action = install_agents_md(tmp_path)

    assert action == "created"
    assert path == tmp_path / "AGENTS.md"
    assert "This repo uses `prism-evals`" in path.read_text(encoding="utf-8")


def test_install_agents_md_appends_to_existing_file(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing\n\nKeep this.", encoding="utf-8")

    path, action = install_agents_md(tmp_path)

    assert path == agents
    assert action == "appended"
    content = agents.read_text(encoding="utf-8")
    assert content.startswith("# Existing\n\nKeep this.")
    assert BEGIN_MARKER in content


def test_install_agents_md_is_noop_when_section_exists(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text(load_agents_template(), encoding="utf-8")

    path, action = install_agents_md(tmp_path)

    assert path == agents
    assert action == "unchanged"
    assert agents.read_text(encoding="utf-8").count(BEGIN_MARKER) == 1


def test_install_agents_md_can_refuse_append(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        install_agents_md(tmp_path, append=False)

    assert agents.read_text(encoding="utf-8") == "# Existing\n"


def test_install_agents_md_force_overwrites(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing\n", encoding="utf-8")

    path, action = install_agents_md(tmp_path, force=True)

    assert path == agents
    assert action == "overwritten"
    assert agents.read_text(encoding="utf-8").startswith(BEGIN_MARKER)


def test_cli_init_creates_agents_file(tmp_path, capsys):
    result = main(["init", "--repo-root", str(tmp_path)])

    assert result == 0
    assert (tmp_path / "AGENTS.md").exists()
    assert "Created Prism Evals instructions" in capsys.readouterr().out


def test_cli_init_reports_failures(tmp_path, capsys):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Existing\n", encoding="utf-8")

    result = main(["init", "--repo-root", str(tmp_path), "--no-append"])

    assert result == 1
    assert "prism init failed" in capsys.readouterr().err
