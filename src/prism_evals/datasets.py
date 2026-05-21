from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCENARIO_SUFFIXES = {".json", ".yaml", ".yml"}
JSONL_SUFFIX = ".jsonl"


@dataclass(frozen=True)
class DatasetItem:
    index: int
    item_id: str
    data: dict[str, Any]
    source_path: str | None = None
    content_hash: str | None = None


def load_dataset(path: Path) -> list[DatasetItem]:
    if path.is_dir():
        return load_dataset_folder(path)
    if path.suffix.lower() == ".csv":
        return load_csv_dataset(path)
    if path.suffix.lower() == JSONL_SUFFIX:
        return load_jsonl_dataset(path)
    if path.suffix.lower() in SCENARIO_SUFFIXES:
        data = load_structured_file(path)
        return [dataset_item(0, data, source_path=str(path), content_hash=file_sha256(path))]
    raise ValueError(f"unsupported dataset path: {path}")


def dataset_sha256(path: Path) -> str:
    if path.is_dir():
        digest = hashlib.sha256()
        for item in load_dataset_folder(path):
            digest.update((item.source_path or item.item_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update((item.content_hash or "").encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
    if path.suffix.lower() == ".csv":
        digest = hashlib.sha256()
        digest.update(file_sha256(path).encode("utf-8"))
        for item in load_csv_dataset(path):
            digest.update(item.item_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update((item.content_hash or "").encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
    return file_sha256(path)


def load_dataset_folder(path: Path) -> list[DatasetItem]:
    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not candidate.name.startswith("_")
        and candidate.suffix.lower() in SCENARIO_SUFFIXES
    ]
    candidates.sort(key=lambda item: item.relative_to(path).as_posix())
    items = [
        dataset_item(
            index,
            load_structured_file(candidate),
            source_path=candidate.relative_to(path).as_posix(),
            content_hash=file_sha256(candidate),
        )
        for index, candidate in enumerate(candidates)
    ]
    ensure_items(items, path)
    return items


def load_csv_dataset(path: Path) -> list[DatasetItem]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset has no header row: {path}")
        items: list[DatasetItem] = []
        for index, raw in enumerate(reader):
            row = {str(key): "" if value is None else str(value) for key, value in raw.items()}
            data = expand_csv_row(row, base_dir=path.parent)
            item_id = str(row.get("id") or data.get("id") or index)
            content_hash = stable_content_hash(data)
            items.append(
                DatasetItem(
                    index=index,
                    item_id=item_id,
                    data=data,
                    source_path=str(path),
                    content_hash=content_hash,
                )
            )
    ensure_items(items, path)
    return items


def load_jsonl_dataset(path: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"JSONL row {line_index + 1} must be an object: {path}")
            items.append(
                dataset_item(
                    len(items),
                    data,
                    source_path=f"{path}:{line_index + 1}",
                    content_hash=stable_content_hash(data),
                )
            )
    ensure_items(items, path)
    return items


def expand_csv_row(row: dict[str, str], *, base_dir: Path) -> dict[str, Any]:
    scenario_path = row.get("scenario_path", "").strip()
    if scenario_path:
        scenario_file = (base_dir / scenario_path).resolve()
        data = load_structured_file(scenario_file)
        data.setdefault("id", row.get("id") or data.get("id") or scenario_file.stem)
        data.setdefault("_csv", row)
        if row.get("tags") and "tags" not in data:
            data["tags"] = split_tags(row["tags"])
        return data

    data: dict[str, Any] = dict(row)
    turns_json = row.get("turns_json", "").strip()
    if turns_json:
        turns = json.loads(turns_json)
        if not isinstance(turns, list):
            raise ValueError("turns_json must contain a list")
        data["turns"] = [normalize_turn(turn, index) for index, turn in enumerate(turns)]
    expectations_json = row.get("expectations_json", "").strip()
    if expectations_json:
        data["expectations"] = json.loads(expectations_json)
    if row.get("tags"):
        data["tags"] = split_tags(row["tags"])
    return data


def load_structured_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported scenario file: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"scenario file must contain an object: {path}")
    return normalize_scenario_data(data)


def normalize_scenario_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    turns = normalized.get("turns")
    if turns is not None:
        if not isinstance(turns, list):
            raise ValueError("scenario turns must be a list")
        normalized["turns"] = [normalize_turn(turn, index) for index, turn in enumerate(turns)]
    return normalized


def normalize_turn(turn: Any, index: int) -> dict[str, Any]:
    if isinstance(turn, str):
        return {"id": f"turn_{index + 1:02d}", "role": "user", "content": turn}
    if not isinstance(turn, dict):
        raise ValueError(f"turn {index + 1} must be an object or string")

    normalized = dict(turn)
    for shorthand, role in (
        ("user", "user"),
        ("assistant_seed", "assistant"),
        ("assistant_expect", "assistant"),
        ("action", "action"),
    ):
        if shorthand in normalized:
            value = normalized.pop(shorthand)
            normalized.setdefault("role", role)
            if shorthand == "assistant_seed":
                normalized.setdefault("mode", "seed")
            elif shorthand == "assistant_expect":
                normalized.setdefault("mode", "expect")
            if shorthand == "action":
                normalized.setdefault("action", value)
            else:
                normalized.setdefault("content", value)

    normalized.setdefault("id", f"turn_{index + 1:02d}")
    normalized.setdefault("role", "user")
    return normalized


def dataset_item(
    index: int,
    data: dict[str, Any],
    *,
    source_path: str | None,
    content_hash: str | None,
) -> DatasetItem:
    fallback_id = str(index)
    if source_path:
        fallback_id = Path(source_path).with_suffix("").as_posix()
    item_id = str(data.get("id") or fallback_id)
    return DatasetItem(index=index, item_id=item_id, data=data, source_path=source_path, content_hash=content_hash)


def ensure_items(items: list[DatasetItem], path: Path) -> None:
    if not items:
        raise ValueError(f"dataset has no rows: {path}")
    seen: dict[str, DatasetItem] = {}
    for item in items:
        previous = seen.get(item.item_id)
        if previous is not None:
            raise ValueError(f"duplicate dataset item id {item.item_id!r}: {path}")
        seen[item.item_id] = item


def split_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def stable_content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
