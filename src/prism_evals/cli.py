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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from prism_evals.scaffold import install_agents_md

DEFAULT_RELEASE_REPOSITORIES = (
    "git@github.com:pt-oai/evals.git",
    "https://github.com/pt-oai/evals.git",
)
DEFAULT_RELEASE_GITHUB_REPOSITORY = "pt-oai/evals"
RELEASE_GITHUB_REPOSITORY_ENV = "PRISM_RELEASE_GITHUB_REPOSITORY"
RELEASE_REPOSITORY_ENV = "PRISM_RELEASE_REPOSITORY"


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
    override = os.environ.get("PRISM_VIEWER_DIR")
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


def viewer_version(app_dir: Path) -> str:
    package_json = app_dir / "package.json"
    if package_json.exists():
        package_data = json.loads(package_json.read_text(encoding="utf-8"))
        package_version = package_data.get("version")
        if isinstance(package_version, str) and package_version:
            return package_version

    try:
        return version("prism-evals")
    except PackageNotFoundError:
        return "0.0.0"


def version_tag(package_version: str) -> str:
    return package_version if package_version.startswith("v") else f"v{package_version}"


def latest_viewer_tag(app_dir: Path, current_tag: str) -> str:
    override = os.environ.get("PRISM_VIEWER_LATEST_TAG")
    if override:
        return override

    tags = [current_tag]
    if is_release_checkout(app_dir):
        tags.extend(local_git_tags(app_dir))
    tags.extend(remote_git_tags())
    return newest_version_tag(tags) or current_tag


def newest_version_tag(tags: Sequence[str]) -> str | None:
    valid_tags = [tag for tag in tags if version_key(tag)]
    if not valid_tags:
        return None
    return max(valid_tags, key=version_key)


def version_key(tag: str) -> tuple[int, ...]:
    raw = tag.removeprefix("refs/tags/").removeprefix("v")
    parts: list[int] = []
    for part in raw.split("."):
        if not part.isdigit():
            return ()
        parts.append(int(part))
    return tuple(parts)


def local_git_tags(app_dir: Path) -> list[str]:
    result = run_git_command(
        ["git", "-C", str(app_dir), "tag", "--list", "v[0-9]*", "--sort=-v:refname"],
        timeout=2,
    )
    return result.splitlines()


def remote_git_tags() -> list[str]:
    if not os.environ.get(RELEASE_REPOSITORY_ENV):
        tags = github_release_tags()
        if tags:
            return tags
    for remote in release_repositories():
        tags = tags_from_remote(remote)
        if tags:
            return tags
    return []


def github_release_tags() -> list[str]:
    repository = os.environ.get(RELEASE_GITHUB_REPOSITORY_ENV, DEFAULT_RELEASE_GITHUB_REPOSITORY)
    result = run_git_command(["gh", "api", f"repos/{repository}/tags", "--paginate", "--jq", ".[].name"], timeout=10)
    return result.splitlines()


def tags_from_remote(remote: str) -> list[str]:
    result = run_git_command(["git", "ls-remote", "--tags", "--refs", remote, "v[0-9]*"], timeout=10)
    tags: list[str] = []
    for line in result.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            tags.append(parts[1].removeprefix("refs/tags/"))
    return tags


def release_repositories() -> list[str]:
    override = os.environ.get(RELEASE_REPOSITORY_ENV)
    if override:
        return [override]
    return list(DEFAULT_RELEASE_REPOSITORIES)


def is_release_checkout(app_dir: Path) -> bool:
    remote = git_remote(app_dir)
    return bool(remote and is_release_repository(remote))


def is_release_repository(remote: str) -> bool:
    normalized = remote.strip().rstrip("/").removesuffix(".git")
    return normalized.endswith("github.com:pt-oai/evals") or normalized.endswith("github.com/pt-oai/evals")


def git_remote(app_dir: Path) -> str | None:
    result = run_git_command(["git", "-C", str(app_dir), "remote", "get-url", "origin"], timeout=2)
    return result.strip() or None


def run_git_command(args: Sequence[str], *, timeout: int) -> str:
    try:
        result = subprocess.run(args, capture_output=True, check=False, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


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
    package_version = viewer_version(app_dir)
    current_tag = version_tag(package_version)
    latest_tag = latest_viewer_tag(app_dir, current_tag)
    env["PRISM_RUNS_DIR"] = str(resolved_runs_dir)
    env["PRISM_VIEWER_VERSION"] = package_version
    env["PRISM_VIEWER_TAG"] = current_tag
    env["PRISM_VIEWER_LATEST_TAG"] = latest_tag
    env["NEXT_TELEMETRY_DISABLED"] = "1"
    env["WATCHPACK_POLLING"] = env.get("WATCHPACK_POLLING", "true")
    url = f"http://{host}:{port}"

    print(f"Opening Prism Evals runs from {resolved_runs_dir}", flush=True)
    print(f"Viewer: {url}", flush=True)
    print(f"Version: {current_tag} (latest: {latest_tag})", flush=True)
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
    parser = argparse.ArgumentParser(prog="prism")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="Add Prism Evals instructions to a repo-root AGENTS.md file.",
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
        help="Fail if AGENTS.md already exists without Prism Evals instructions.",
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
            print(f"prism init failed: {exc}", file=sys.stderr)
            return 1

        if action == "unchanged":
            print(f"Prism Evals instructions already present in {path}")
        else:
            print(f"{action.capitalize()} Prism Evals instructions in {path}")
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
            print(f"prism view failed: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2
