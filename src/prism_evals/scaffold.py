from __future__ import annotations

from importlib.resources import files
from pathlib import Path


BEGIN_MARKER = "<!-- prism-evals instructions begin -->"
END_MARKER = "<!-- prism-evals instructions end -->"


def load_agents_template() -> str:
    template = files("prism_evals").joinpath("templates", "AGENTS.md")
    return template.read_text(encoding="utf-8")


def install_agents_md(
    repo_root: str | Path = ".",
    *,
    force: bool = False,
    append: bool = True,
) -> tuple[Path, str]:
    root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise NotADirectoryError(f"repo root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"repo root is not a directory: {root}")

    destination = root / "AGENTS.md"
    template = load_agents_template()

    if force:
        destination.write_text(template, encoding="utf-8")
        return destination, "overwritten"

    if not destination.exists():
        destination.write_text(template, encoding="utf-8")
        return destination, "created"

    existing = destination.read_text(encoding="utf-8")
    if BEGIN_MARKER in existing and END_MARKER in existing:
        return destination, "unchanged"

    if not append:
        raise FileExistsError(
            f"{destination} already exists; rerun with append enabled or --force"
        )

    destination.write_text(existing.rstrip() + "\n\n" + template, encoding="utf-8")
    return destination, "appended"
