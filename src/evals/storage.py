from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evals._utils import to_jsonable
from evals.models import ExecutionRecord, RunManifest


class Storage:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.results_jsonl_path = run_dir / "results.jsonl"
        self.results_csv_path = run_dir / "results.csv"
        self.scores_csv_path = run_dir / "scores.csv"

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

    def append_record(self, record: ExecutionRecord) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.results_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def load_latest_records(self) -> dict[str, ExecutionRecord]:
        records: dict[str, ExecutionRecord] = {}
        if not self.results_jsonl_path.exists():
            return records
        with self.results_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = ExecutionRecord.model_validate_json(line)
                except Exception:
                    continue
                records[record.execution_id] = record
        return records

    def write_csvs(self, records: list[ExecutionRecord]) -> None:
        self._write_results_csv(records)
        self._write_scores_csv(records)

    def artifact_paths(self) -> dict[str, Path]:
        return {
            "manifest": self.manifest_path,
            "jsonl": self.results_jsonl_path,
            "results_csv": self.results_csv_path,
            "scores_csv": self.scores_csv_path,
        }

    def _write_results_csv(self, records: list[ExecutionRecord]) -> None:
        eval_keys = sorted(
            {
                result.key
                for record in records
                for result in record.evals
                if result.key is not None
            }
        )
        row_keys = sorted({key for record in records for key in record.row})
        fieldnames = [
            "execution_id",
            "run_id",
            "experiment_name",
            "row_index",
            "row_id",
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
        fieldnames.extend(f"row:{key}" for key in row_keys)
        fieldnames.extend(f"score:{key}" for key in eval_keys)
        fieldnames.extend(f"score_error:{key}" for key in eval_keys)

        with self.results_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = flatten_execution(record)
                for key, value in record.row.items():
                    row[f"row:{key}"] = value
                for result in record.evals:
                    if result.key is None:
                        continue
                    row[f"score:{result.key}"] = result.score
                    if result.error:
                        row[f"score_error:{result.key}"] = result.error.message
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _write_scores_csv(self, records: list[ExecutionRecord]) -> None:
        fieldnames = [
            "execution_id",
            "run_id",
            "experiment_name",
            "row_id",
            "row_index",
            "model_key",
            "model",
            "repetition",
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
                    writer.writerow(
                        {
                            "execution_id": record.execution_id,
                            "run_id": record.run_id,
                            "experiment_name": record.experiment_name,
                            "row_id": record.row_id,
                            "row_index": record.row_index,
                            "model_key": record.model_key,
                            "model": record.model,
                            "repetition": record.repetition,
                            "score_key": result.key,
                            "score": result.score,
                            "data_type": result.data_type,
                            "description": result.description,
                            "comment": result.comment,
                            "error_type": result.error.type if result.error else "",
                            "error_message": result.error.message if result.error else "",
                            "metadata_json": json.dumps(to_jsonable(result.metadata), sort_keys=True),
                        }
                    )


def flatten_execution(record: ExecutionRecord) -> dict[str, Any]:
    latency_s = sum(generation.latency_s for generation in record.generations)
    return {
        "execution_id": record.execution_id,
        "run_id": record.run_id,
        "experiment_name": record.experiment_name,
        "row_index": record.row_index,
        "row_id": record.row_id,
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

