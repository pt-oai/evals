from __future__ import annotations

import asyncio
import inspect
import platform
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prism_evals._utils import file_sha256, git_commit, stable_hash, utc_now_iso
from prism_evals.console import ConsoleReporter
from prism_evals.datasets import DatasetItem, dataset_sha256, load_dataset
from prism_evals.errors import exception_to_error
from prism_evals.evaluation import has_eval_errors, run_eval_definitions
from prism_evals.models import (
    ErrorRecord,
    EvalResult,
    ItemRunRecord,
    ModelConfig,
    ModelVariant,
    RunManifest,
    TaskOutput,
    TokenUsage,
    require_task_output,
)
from prism_evals.openai import ExperimentContext
from prism_evals.storage import Storage

if TYPE_CHECKING:
    from prism_evals.experiment import Experiment


class Runner:
    def __init__(self, experiment: "Experiment") -> None:
        self.experiment = experiment
        self.storage = Storage(experiment.run_dir())
        self.reporter = ConsoleReporter(experiment.display)

    async def run(self) -> list[ItemRunRecord]:
        exp = self.experiment
        exp.validate()
        items = load_dataset(exp.dataset)
        dataset_hash = dataset_sha256(exp.dataset)
        run_id = stable_hash(
            {
                "name": exp.name,
                "dataset": dataset_hash,
                "experiment": file_sha256(exp.source_file) if exp.source_file else None,
            }
        )[:16]
        manifest = self._manifest(run_id)
        self.storage.prepare(manifest)
        copied_artifacts = self.storage.copy_artifacts(list(exp.artifacts), base_dir=exp.base_dir)
        if copied_artifacts:
            manifest.metadata = dict(manifest.metadata)
            manifest.metadata["copied_artifacts"] = copied_artifacts
            self.storage.write_manifest(manifest)
        existing = self.storage.load_latest_records()
        successful_ids = {
            item_run_id
            for item_run_id, record in existing.items()
            if record.status == "success"
        }
        work = [
            (item, model, repetition)
            for repetition in range(exp.repetitions)
            for model in exp.registered_variants
            for item in items
            if not (exp.resume and item_run_id(exp, item, model, repetition) in successful_ids)
        ]
        total = len(items) * len(exp.registered_variants) * exp.repetitions
        skipped = total - len(work)
        remaining_by_model = Counter(model.key for _, model, _ in work)
        client = exp.openai_client
        new_records: list[ItemRunRecord] = []
        latest_records = dict(existing)
        append_lock = asyncio.Lock()

        self.reporter.start(
            experiment_name=exp.name,
            item_count=len(items),
            model_count=len(exp.registered_variants),
            repetitions=exp.repetitions,
            total_item_runs=total,
            remaining_item_runs=len(work),
            skipped=skipped,
            run_dir=exp.run_dir(),
            models=exp.registered_variants,
            remaining_by_model=dict(remaining_by_model),
        )

        semaphore = asyncio.Semaphore(exp.concurrency)

        async def run_one(item: DatasetItem, model: ModelConfig, repetition: int) -> ItemRunRecord:
            async with semaphore:
                record = await self._run_item_run(
                    run_id=run_id,
                    item=item,
                    model=model,
                    repetition=repetition,
                    client=client,
                )
                async with append_lock:
                    self.storage.append_record(record)
                    latest_records[record.item_run_id] = record
                    new_records.append(record)
                self.reporter.record(record)
                if exp.fail_fast and record.status == "failed":
                    raise RuntimeError(record.error.message if record.error else "item run failed")
                return record

        try:
            await asyncio.gather(*(run_one(*item) for item in work))
        finally:
            records = sorted(
                latest_records.values(),
                key=lambda record: (record.model_key, record.repetition, record.item_index),
            )
            manifest.ended_at = utc_now_iso()
            self.storage.write_manifest(manifest)
            self.storage.write_csvs(records)
            self.reporter.finish(records, self.storage.artifact_paths())

        return new_records

    async def _run_item_run(
        self,
        *,
        run_id: str,
        item: DatasetItem,
        model: ModelVariant,
        repetition: int,
        client: Any,
    ) -> ItemRunRecord:
        exp = self.experiment
        workflow = exp.workflow
        if workflow is None:
            raise ValueError("missing workflow function")

        item_run_id_value = item_run_id(exp, item, model, repetition)
        started_at = utc_now_iso()
        started = time.perf_counter()
        ctx = ExperimentContext(
            client=client,
            item=item.data,
            model=model,
            item_run_id=item_run_id_value,
            run_dir=exp.run_dir(),
            capture_raw=exp.capture_raw,
            max_retries=exp.max_retries,
            redact_raw_data_urls=exp.redact_raw_data_urls,
            fail_fast=exp.fail_fast,
        )
        output: TaskOutput | None = None
        evals: list[EvalResult] = []
        error = None
        status = "success"

        try:
            workflow_result = workflow(item.data, model, ctx)
            if inspect.isawaitable(workflow_result):
                workflow_result = await workflow_result
            output = require_task_output(workflow_result, context="workflow")
            evals = await self._run_evals(item, model, output, ctx)
            if exp.fail_fast and has_eval_errors(evals):
                raise RuntimeError("item-run eval failed")
            if not await self._passes_condition(evals):
                status = "failed"
                error = ErrorRecord(
                    type="PassConditionFailed",
                    message="pass condition was not met",
                )
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

        return ItemRunRecord(
            item_run_id=item_run_id_value,
            run_id=run_id,
            experiment_name=exp.name,
            item_index=item.index,
            item_id=item.item_id,
            item=item.data,
            model_key=model.key,
            model=model.model,
            model_params=model.params,
            variant_key=model.key,
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
            steps=ctx.steps,
            turns=ctx.turns,
            tool_calls=ctx.tool_calls,
            raw_input=raw_input,
            raw_output=raw_output,
            error=error,
        )

    async def _run_evals(
        self,
        item: DatasetItem,
        model: ModelConfig,
        output: TaskOutput,
        ctx: ExperimentContext,
    ) -> list[EvalResult]:
        return await run_eval_definitions(
            self.experiment.registered_evals,
            item=item.data,
            model=model,
            output=output,
            ctx=ctx,
        )

    async def _passes_condition(self, evals: list[EvalResult]) -> bool:
        condition = self.experiment.pass_condition
        if condition is None:
            return True
        scores = {result.key: result.score for result in evals if result.key}
        passed = condition(scores)
        if inspect.isawaitable(passed):
            passed = await passed
        return bool(passed)

    def _manifest(self, run_id: str) -> RunManifest:
        exp = self.experiment
        dataset_hash = dataset_sha256(exp.dataset)
        return RunManifest(
            run_id=run_id,
            experiment_name=exp.name,
            started_at=utc_now_iso(),
            dataset_path=str(exp.dataset),
            dataset_sha256=dataset_hash,
            experiment_file=str(exp.source_file) if exp.source_file else None,
            experiment_sha256=file_sha256(exp.source_file) if exp.source_file else None,
            output_dir=str(exp.run_dir()),
            settings={
                "concurrency": exp.concurrency,
                "resume": exp.resume,
                "repetitions": exp.repetitions,
                "max_retries": exp.max_retries,
                "fail_fast": exp.fail_fast,
                "pass_condition": exp.pass_condition is not None,
                "capture_raw": exp.capture_raw,
                "redact_raw_data_urls": exp.redact_raw_data_urls,
                "timestamp_output_dir": exp.timestamp_output_dir,
                "artifacts": [str(artifact) for artifact in exp.artifacts],
                "display": exp.display,
            },
            model_configs=manifest_model_configs(exp.registered_variants),
            variant_configs=exp.registered_variants,
            metadata=exp.metadata,
            git_commit=git_commit(Path.cwd()),
            python_version=platform.python_version(),
        )


def manifest_model_configs(variants: list[ModelVariant]) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for variant in variants:
        default = variant.default_model
        metadata = dict(default.metadata)
        metadata["variant_key"] = variant.key
        metadata["variant_models"] = {
            role: config.model_dump(mode="json")
            for role, config in variant.models.items()
        }
        configs.append(
            ModelConfig(
                key=variant.key,
                model=default.model,
                params=default.params,
                metadata=metadata,
            )
        )
    return configs


def item_run_id(
    experiment: "Experiment",
    item: DatasetItem,
    model: ModelVariant,
    repetition: int,
) -> str:
    return stable_hash(
        {
            "experiment_name": experiment.name,
            "dataset_sha256": dataset_sha256(experiment.dataset),
            "item_sha256": item.content_hash,
            "item_id": item.item_id,
            "item_index": item.index,
            "model_key": model.key,
            "repetition": repetition,
        }
    )[:24]
