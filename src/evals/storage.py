from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evals._utils import to_jsonable
from evals.models import ItemRunRecord, RunManifest, StepRecord


class Storage:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.results_jsonl_path = run_dir / "results.jsonl"
        self.results_csv_path = run_dir / "results.csv"
        self.scores_csv_path = run_dir / "scores.csv"
        self.steps_csv_path = run_dir / "steps.csv"

    def prepare(self, manifest: RunManifest) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.write_manifest(manifest)

    def write_manifest(self, manifest: RunManifest) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def append_record(self, record: ItemRunRecord) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.results_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def load_latest_records(self) -> dict[str, ItemRunRecord]:
        records: dict[str, ItemRunRecord] = {}
        if not self.results_jsonl_path.exists():
            return records
        with self.results_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ItemRunRecord.model_validate_json(line)
                except Exception:
                    continue
                records[record.item_run_id] = record
        return records

    def write_csvs(self, records: list[ItemRunRecord]) -> None:
        self._write_results_csv(records)
        self._write_scores_csv(records)
        self._write_steps_csv(records)

    def artifact_paths(self) -> dict[str, Path]:
        return {
            "manifest": self.manifest_path,
            "jsonl": self.results_jsonl_path,
            "results_csv": self.results_csv_path,
            "scores_csv": self.scores_csv_path,
            "steps_csv": self.steps_csv_path,
        }

    def _write_results_csv(self, records: list[ItemRunRecord]) -> None:
        eval_keys = sorted(
            {
                result.key
                for record in records
                for result in record.evals
                if result.key is not None
            }
        )
        step_eval_keys = sorted(
            {
                (step.key, result.key)
                for record in records
                for step in record.steps
                for result in step.evals
                if result.key is not None
            }
        )
        item_keys = sorted({key for record in records for key in record.item})
        fieldnames = [
            "item_run_id",
            "run_id",
            "experiment_name",
            "item_index",
            "item_id",
            "model_key",
            "model",
            "repetition",
            "status",
            "response_id",
            "duration_s",
            "latency_s",
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "output_text",
            "error_type",
            "error_message",
        ]
        fieldnames.extend(f"item:{key}" for key in item_keys)
        fieldnames.extend(f"score:{key}" for key in eval_keys)
        fieldnames.extend(f"score_error:{key}" for key in eval_keys)
        fieldnames.extend(f"step:{step_key}.score:{score_key}" for step_key, score_key in step_eval_keys)
        fieldnames.extend(f"step:{step_key}.score_error:{score_key}" for step_key, score_key in step_eval_keys)

        with self.results_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                flat = flatten_item_run(record)
                for key, value in record.item.items():
                    flat[f"item:{key}"] = value
                for result in record.evals:
                    if result.key is None:
                        continue
                    flat[f"score:{result.key}"] = result.score
                    if result.error:
                        flat[f"score_error:{result.key}"] = result.error.message
                for step in record.steps:
                    for result in step.evals:
                        if result.key is None:
                            continue
                        flat[f"step:{step.key}.score:{result.key}"] = result.score
                        if result.error:
                            flat[f"step:{step.key}.score_error:{result.key}"] = result.error.message
                writer.writerow({key: flat.get(key, "") for key in fieldnames})

    def _write_scores_csv(self, records: list[ItemRunRecord]) -> None:
        fieldnames = [
            "item_run_id",
            "run_id",
            "experiment_name",
            "item_id",
            "item_index",
            "model_key",
            "model",
            "repetition",
            "scope",
            "step_key",
            "score_key",
            "score",
            "data_type",
            "description",
            "comment",
            "error_type",
            "error_message",
            "metadata_json",
        ]
        with self.scores_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                for result in record.evals:
                    writer.writerow(score_row(record, result, scope="item_run", step_key=""))
                for step in record.steps:
                    for result in step.evals:
                        writer.writerow(score_row(record, result, scope="step", step_key=step.key))

    def _write_steps_csv(self, records: list[ItemRunRecord]) -> None:
        fieldnames = [
            "item_run_id",
            "run_id",
            "experiment_name",
            "item_id",
            "item_index",
            "model_key",
            "model",
            "repetition",
            "step_key",
            "status",
            "duration_s",
            "latency_s",
            "response_id",
            "input_tokens",
            "cached_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "output_text",
            "error_type",
            "error_message",
            "scores_json",
        ]
        with self.steps_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                for step in record.steps:
                    writer.writerow(step_row(record, step))


def score_row(record: ItemRunRecord, result: Any, *, scope: str, step_key: str) -> dict[str, Any]:
    return {
        "item_run_id": record.item_run_id,
        "run_id": record.run_id,
        "experiment_name": record.experiment_name,
        "item_id": record.item_id,
        "item_index": record.item_index,
        "model_key": record.model_key,
        "model": record.model,
        "repetition": record.repetition,
        "scope": scope,
        "step_key": step_key,
        "score_key": result.key,
        "score": result.score,
        "data_type": result.data_type,
        "description": result.description,
        "comment": result.comment,
        "error_type": result.error.type if result.error else "",
        "error_message": result.error.message if result.error else "",
        "metadata_json": json.dumps(to_jsonable(result.metadata), sort_keys=True),
    }


def step_row(record: ItemRunRecord, step: StepRecord) -> dict[str, Any]:
    latency_s = sum(generation.latency_s for generation in step.generations)
    scores = {
        result.key: {
            "score": result.score,
            "error": result.error.message if result.error else None,
        }
        for result in step.evals
        if result.key is not None
    }
    return {
        "item_run_id": record.item_run_id,
        "run_id": record.run_id,
        "experiment_name": record.experiment_name,
        "item_id": record.item_id,
        "item_index": record.item_index,
        "model_key": record.model_key,
        "model": record.model,
        "repetition": record.repetition,
        "step_key": step.key,
        "status": step.status,
        "duration_s": f"{step.duration_s:.6f}",
        "latency_s": f"{latency_s:.6f}",
        "response_id": step.response_id,
        "input_tokens": step.usage.input_tokens,
        "cached_tokens": step.usage.cached_tokens,
        "output_tokens": step.usage.output_tokens,
        "reasoning_tokens": step.usage.reasoning_tokens,
        "total_tokens": step.usage.total_tokens,
        "output_text": step.output.text if step.output else "",
        "error_type": step.error.type if step.error else "",
        "error_message": step.error.message if step.error else "",
        "scores_json": json.dumps(to_jsonable(scores), sort_keys=True),
    }


def flatten_item_run(record: ItemRunRecord) -> dict[str, Any]:
    latency_s = sum(generation.latency_s for generation in record.generations)
    return {
        "item_run_id": record.item_run_id,
        "run_id": record.run_id,
        "experiment_name": record.experiment_name,
        "item_index": record.item_index,
        "item_id": record.item_id,
        "model_key": record.model_key,
        "model": record.model,
        "repetition": record.repetition,
        "status": record.status,
        "response_id": record.response_id,
        "duration_s": f"{record.duration_s:.6f}",
        "latency_s": f"{latency_s:.6f}",
        "input_tokens": record.usage.input_tokens,
        "cached_tokens": record.usage.cached_tokens,
        "output_tokens": record.usage.output_tokens,
        "reasoning_tokens": record.usage.reasoning_tokens,
        "total_tokens": record.usage.total_tokens,
        "output_text": record.output.text if record.output else "",
        "error_type": record.error.type if record.error else "",
        "error_message": record.error.message if record.error else "",
    }
