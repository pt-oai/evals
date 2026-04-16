from __future__ import annotations

import glob
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prism_evals._utils import file_sha256


@dataclass(frozen=True)
class ArtifactCopy:
    spec: str
    source: Path
    destination: Path
    destination_relative_path: Path

    def metadata(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "source_path": str(self.source),
            "destination_path": str(self.destination),
            "destination_relative_path": self.destination_relative_path.as_posix(),
            "sha256": file_sha256(self.source),
        }


def copy_artifacts(
    specs: list[str | Path],
    *,
    base_dir: Path,
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    copies = plan_artifact_copies(specs, base_dir=base_dir, artifacts_dir=artifacts_dir)
    if not copies:
        return []

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for copy in copies:
        copy.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy.source, copy.destination)
    return [copy.metadata() for copy in copies]


def plan_artifact_copies(
    specs: list[str | Path],
    *,
    base_dir: Path,
    artifacts_dir: Path,
) -> list[ArtifactCopy]:
    copies: list[ArtifactCopy] = []
    destinations: dict[str, str] = {}
    base_dir = base_dir.resolve()

    for spec in specs:
        spec_text = str(spec)
        if "**" in Path(spec_text).parts:
            raise ValueError(f"recursive artifact globs are not supported: {spec_text}")

        sources = expand_artifact_sources(spec_text, base_dir=base_dir)
        for source in sources:
            if source.is_dir():
                raise ValueError(f"artifact spec matched a directory: {spec_text} -> {source}")
            if not source.is_file():
                raise ValueError(f"artifact source is not a file: {source}")

            destination_relative_path = artifact_destination_relative_path(spec_text, source, base_dir)
            destination_key = destination_relative_path.as_posix()
            if destination_key in destinations:
                raise ValueError(
                    "duplicate artifact destination: "
                    f"artifacts/{destination_key} from {destinations[destination_key]!r} and {spec_text!r}"
                )
            destinations[destination_key] = spec_text
            copies.append(
                ArtifactCopy(
                    spec=spec_text,
                    source=source,
                    destination=artifacts_dir / destination_relative_path,
                    destination_relative_path=Path("artifacts") / destination_relative_path,
                )
            )
    return copies


def expand_artifact_sources(spec: str, *, base_dir: Path) -> list[Path]:
    path = Path(spec)
    has_magic = glob.has_magic(spec)
    if has_magic:
        pattern = spec if path.is_absolute() else str(base_dir / spec)
        matches = sorted(Path(match).resolve() for match in glob.glob(pattern))
        if not matches:
            raise ValueError(f"artifact glob matched no files: {spec}")
        return matches

    source = path if path.is_absolute() else base_dir / path
    source = source.resolve()
    if not source.exists():
        raise ValueError(f"artifact file not found: {spec}")
    return [source]


def artifact_destination_relative_path(spec: str, source: Path, base_dir: Path) -> Path:
    if Path(spec).is_absolute():
        return Path(source.name)
    try:
        relative_path = source.resolve().relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(
            f"relative artifact resolves outside the experiment directory: {spec}"
        ) from exc
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"artifact destination cannot escape artifacts directory: {spec}")
    return relative_path
