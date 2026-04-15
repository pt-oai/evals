from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from evals.scaffold import install_agents_md


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

    parser.error(f"unknown command: {args.command}")
    return 2
