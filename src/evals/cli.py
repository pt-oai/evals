from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from evals.scaffold import install_agents_md


def validate_runs_parent(path: str | Path) -> Path:
    runs_dir = Path(path).expanduser().resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory does not exist: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"runs path is not a directory: {runs_dir}")
    if (runs_dir / "manifest.json").exists():
        raise ValueError(
            f"{runs_dir} is a run directory; pass the parent directory that contains all runs"
        )
    if not any(child.is_dir() and (child / "manifest.json").exists() for child in runs_dir.iterdir()):
        raise FileNotFoundError(f"no run folders with manifest.json found in {runs_dir}")
    return runs_dir


def viewer_dir() -> Path:
    override = os.environ.get("PT_EVALS_VIEWER_DIR")
    if override:
        return Path(override).expanduser().resolve()
    source_checkout = Path(__file__).resolve().parents[2] / "viewer"
    bundled_viewer = Path(__file__).resolve().parent / "viewer"
    for candidate in (source_checkout, bundled_viewer):
        if (candidate / "package.json").exists():
            return candidate
    return source_checkout


def ensure_viewer_dependencies(app_dir: Path) -> None:
    if viewer_dependencies_installed(app_dir):
        return
    if shutil.which("npm") is None:
        raise FileNotFoundError("npm is required to start the viewer")
    print(f"Installing viewer dependencies in {app_dir}", flush=True)
    subprocess.run(["npm", "install"], cwd=app_dir, check=True)


def viewer_dependencies_installed(app_dir: Path) -> bool:
    if not (app_dir / "node_modules" / ".bin" / "next").exists():
        return False
    package_json = app_dir / "package.json"
    if not package_json.exists():
        return False
    package_data = json.loads(package_json.read_text(encoding="utf-8"))
    dependencies = package_data.get("dependencies", {})
    if not isinstance(dependencies, dict):
        return False
    for name in dependencies:
        if not (app_dir / "node_modules" / Path(*name.split("/"))).exists():
            return False
    return True


def run_viewer(
    runs_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 3000,
    open_browser: bool = True,
) -> int:
    resolved_runs_dir = validate_runs_parent(runs_dir)
    app_dir = viewer_dir()
    if not app_dir.exists():
        raise FileNotFoundError(f"viewer app not found: {app_dir}")
    if not (app_dir / "package.json").exists():
        raise FileNotFoundError(f"viewer package not found: {app_dir / 'package.json'}")
    ensure_viewer_dependencies(app_dir)

    env = os.environ.copy()
    env["PT_EVALS_RUNS_DIR"] = str(resolved_runs_dir)
    env["NEXT_TELEMETRY_DISABLED"] = "1"
    env["WATCHPACK_POLLING"] = env.get("WATCHPACK_POLLING", "true")
    url = f"http://{host}:{port}"

    print(f"Opening eval runs from {resolved_runs_dir}", flush=True)
    print(f"Viewer: {url}", flush=True)
    process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--hostname", host, "--port", str(port)],
        cwd=app_dir,
        env=env,
    )
    if open_browser:
        time.sleep(1)
        webbrowser.open(url)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pt-evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Add pt-evals instructions to a repo-root AGENTS.md file.",
    )
    init_parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to update. Defaults to the current directory.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite AGENTS.md instead of creating or appending.",
    )
    init_parser.add_argument(
        "--no-append",
        action="store_true",
        help="Fail if AGENTS.md already exists without pt-evals instructions.",
    )

    view_parser = subparsers.add_parser(
        "view",
        help="Open a local viewer for a parent directory of eval runs.",
    )
    view_parser.add_argument(
        "runs_dir",
        help="Parent directory containing run folders.",
    )
    view_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the local viewer. Defaults to 127.0.0.1.",
    )
    view_parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port for the local viewer. Defaults to 3000.",
    )
    view_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Start the viewer without opening a browser.",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        try:
            path, action = install_agents_md(
                args.repo_root,
                force=args.force,
                append=not args.no_append,
            )
        except Exception as exc:
            print(f"pt-evals init failed: {exc}", file=sys.stderr)
            return 1

        if action == "unchanged":
            print(f"pt-evals instructions already present in {path}")
        else:
            print(f"{action.capitalize()} pt-evals instructions in {path}")
        return 0

    if args.command == "view":
        try:
            return run_viewer(
                args.runs_dir,
                host=args.host,
                port=args.port,
                open_browser=not args.no_open,
            )
        except Exception as exc:
            print(f"pt-evals view failed: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2
