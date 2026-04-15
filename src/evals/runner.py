from __future__ import annotations

import asyncio
import csv
import inspect
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals._utils import file_sha256, git_commit, stable_hash, utc_now_iso
from evals.console import ConsoleReporter
from evals.models import (
    ErrorRecord,
    EvalResult,
    ExecutionRecord,
    ModelConfig,
    RunManifest,
    TaskOutput,
    TokenUsage,
)
from evals.openai import ExperimentContext, exception_to_error, make_default_client
from evals.storage import Storage

if TYPE_CHECKING:
    from evals.experiment import EvalDefinition, Experiment


@dataclass(frozen=True)
class DatasetRow:
    index: int
    row_id: str
    data: dict[str, str]


class Runner:
    def __init__(self, experiment: "Experiment") -> None:
        self.experiment = experiment
        self.storage = Storage(experiment.run_dir())
        self.reporter = ConsoleReporter(experiment.display)

    async def run(self) -> list[ExecutionRecord]:
        exp = self.experiment
        exp.validate()
        rows = load_dataset(exp.dataset)
        run_id = stable_hash(
            {
                "name": exp.name,
                "dataset": file_sha256(exp.dataset),
                "experiment": file_sha256(exp.source_file) if exp.source_file else None,
            }
        )[:16]
        manifest = self._manifest(run_id)
        self.storage.prepare(manifest)
        existing = self.storage.load_latest_records()
        successful_ids = {
            execution_id
            for execution_id, record in existing.items()
            if record.status == "success"
        }
        work = [
            (row, model, repetition)
            for repetition in range(exp.repetitions)
            for model in exp.registered_models
            for row in rows
            if not (exp.resume and execution_id(exp, row, model, repetition) in successful_ids)
        ]
        total = len(rows) * len(exp.registered_models) * exp.repetitions
        skipped = total - len(work)
        remaining_by_model = Counter(model.key for _, model, _ in work)
        client = exp.openai_client or make_default_client()
        new_records: list[ExecutionRecord] = []
        latest_records = dict(existing)
        append_lock = asyncio.Lock()

        self.reporter.start(
            experiment_name=exp.name,
            row_count=len(rows),
            model_count=len(exp.registered_models),
            repetitions=exp.repetitions,
            total_executions=total,
            remaining_executions=len(work),
            skipped=skipped,
            run_dir=exp.run_dir(),
            models=exp.registered_models,
            remaining_by_model=dict(remaining_by_model),
        )

        semaphore = asyncio.Semaphore(exp.concurrency)

        async def run_one(row: DatasetRow, model: ModelConfig, repetition: int) -> ExecutionRecord:
            async with semaphore:
                record = await self._run_execution(
                    run_id=run_id,
                    row=row,
                    model=model,
                    repetition=repetition,
                    client=client,
                )
                async with append_lock:
                    self.storage.append_record(record)
                    latest_records[record.execution_id] = record
                    new_records.append(record)
                self.reporter.record(record)
                if exp.fail_fast and record.status == "failed":
                    raise RuntimeError(record.error.message if record.error else "execution failed")
                return record

        try:
            await asyncio.gather(*(run_one(*item) for item in work))
        finally:
            records = sorted(
                latest_records.values(),
                key=lambda record: (record.model_key, record.repetition, record.row_index),
            )
            manifest.ended_at = utc_now_iso()
            self.storage.write_manifest(manifest)
            self.storage.write_csvs(records)
            self.reporter.finish(records, self.storage.artifact_paths())

        return new_records

    async def _run_execution(
        self,
        *,
        run_id: str,
        row: DatasetRow,
        model: ModelConfig,
        repetition: int,
        client: Any,
    ) -> ExecutionRecord:
        exp = self.experiment
        task = exp.task
        if task is None:
            raise ValueError("missing task function")

        exec_id = execution_id(exp, row, model, repetition)
        started_at = utc_now_iso()
        started = time.perf_counter()
        ctx = ExperimentContext(
            client=client,
            row=row.data,
            model=model,
            execution_id=exec_id,
            capture_raw=exp.capture_raw,
            max_retries=exp.max_retries,
        )
        output: TaskOutput | None = None
        evals: list[EvalResult] = []
        error: ErrorRecord | None = None
        status = "success"

        try:
            task_result = task(row.data, model, ctx)
            if inspect.isawaitable(task_result):
                task_result = await task_result
            output = TaskOutput.normalize(task_result)
            evals = await self._run_evals(row, model, output, ctx)
        except Exception as exc:
            status = "failed"
            error = exception_to_error(exc)

        ended_at = utc_now_iso()
        duration_s = time.perf_counter() - started
        usage = TokenUsage()
        for generation in ctx.generations:
            usage += generation.usage
        response_id = None
        for generation in reversed(ctx.generations):
            if generation.response_id:
                response_id = generation.response_id
                break
        raw_input = ctx.generations[0].raw_request if ctx.generations else None
        raw_output = ctx.generations[-1].raw_response if ctx.generations else None

        return ExecutionRecord(
            execution_id=exec_id,
            run_id=run_id,
            experiment_name=exp.name,
            row_index=row.index,
            row_id=row.row_id,
            row=row.data,
            model_key=model.key,
            model=model.model,
            model_params=model.params,
            repetition=repetition,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
            output=output,
            evals=evals,
            usage=usage,
            response_id=response_id,
            generations=ctx.generations,
            raw_input=raw_input,
            raw_output=raw_output,
            error=error,
        )

    async def _run_evals(
        self,
        row: DatasetRow,
        model: ModelConfig,
        output: TaskOutput,
        ctx: ExperimentContext,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        for definition in self.experiment.registered_evals:
            try:
                value = definition.func(row.data, model, output, ctx)
                if inspect.isawaitable(value):
                    value = await value
                results.extend(normalize_eval_return(value, definition))
            except Exception as exc:
                error_result = EvalResult(
                    key=definition.key,
                    description=definition.description,
                    error=exception_to_error(exc),
                )
                results.append(error_result)
                if self.experiment.fail_fast:
                    raise
        return results

    def _manifest(self, run_id: str) -> RunManifest:
        exp = self.experiment
        return RunManifest(
            run_id=run_id,
            experiment_name=exp.name,
            started_at=utc_now_iso(),
            dataset_path=str(exp.dataset),
            dataset_sha256=file_sha256(exp.dataset),
            experiment_file=str(exp.source_file) if exp.source_file else None,
            experiment_sha256=file_sha256(exp.source_file) if exp.source_file else None,
            output_dir=str(exp.run_dir()),
            settings={
                "concurrency": exp.concurrency,
                "resume": exp.resume,
                "repetitions": exp.repetitions,
                "max_retries": exp.max_retries,
                "fail_fast": exp.fail_fast,
                "capture_raw": exp.capture_raw,
                "display": exp.display,
            },
            model_configs=exp.registered_models,
            metadata=exp.metadata,
            git_commit=git_commit(Path.cwd()),
            python_version=platform.python_version(),
        )


def load_dataset(path: Path) -> list[DatasetRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"dataset has no header row: {path}")
        rows = []
        for index, raw in enumerate(reader):
            data = {str(key): "" if value is None else str(value) for key, value in raw.items()}
            row_id = data.get("id") or str(index)
            rows.append(DatasetRow(index=index, row_id=str(row_id), data=data))
    if not rows:
        raise ValueError(f"dataset has no rows: {path}")
    return rows


def execution_id(
    experiment: "Experiment",
    row: DatasetRow,
    model: ModelConfig,
    repetition: int,
) -> str:
    return stable_hash(
        {
            "experiment_name": experiment.name,
            "dataset_sha256": file_sha256(experiment.dataset),
            "row_id": row.row_id,
            "row_index": row.index,
            "model_key": model.key,
            "repetition": repetition,
        }
    )[:24]


def normalize_eval_return(value: Any, definition: "EvalDefinition") -> list[EvalResult]:
    if value is None:
        return []
    if isinstance(value, EvalResult):
        return [apply_registered_key(value, definition.key, definition.description)]
    if isinstance(value, (bool, int, float)):
        return [
            EvalResult(score=value).with_defaults(definition.key, definition.description)
        ]
    if isinstance(value, dict):
        results = []
        for key, score in value.items():
            if not isinstance(score, (bool, int, float)):
                raise TypeError(f"eval dict value for {key!r} must be bool, int, or float")
            results.append(EvalResult(key=str(key), score=score))
        return results
    if isinstance(value, list):
        results = []
        for item in value:
            if not isinstance(item, EvalResult):
                raise TypeError("eval lists may only contain EvalResult instances")
            results.append(item.with_defaults(definition.key, definition.description))
        return results
    raise TypeError(
        "eval must return None, bool, int, float, dict, EvalResult, or list[EvalResult]"
    )


def apply_registered_key(
    result: EvalResult,
    key: str,
    description: str | None = None,
) -> EvalResult:
    data = result.model_dump()
    data["key"] = key
    data["description"] = data["description"] or description
    if data["data_type"] is None and data["score"] is not None:
        from evals.models import infer_score_type

        data["data_type"] = infer_score_type(data["score"])
    return EvalResult(**data)
