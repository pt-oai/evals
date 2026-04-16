from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from prism_evals.models import EvalDefinition, ItemRunRecord, ModelConfig
from prism_evals.runner import Runner

WorkflowFn = Callable[..., Any]
EvalFn = Callable[..., Any]


class Experiment:
    def __init__(
        self,
        *,
        name: str,
        dataset: str | Path,
        output_dir: str | Path = "runs",
        concurrency: int = 5,
        resume: bool = True,
        repetitions: int = 1,
        max_retries: int = 3,
        fail_fast: bool = False,
        capture_raw: bool = True,
        redact_raw_data_urls: bool = True,
        timestamp_output_dir: bool = True,
        artifacts: Iterable[str | Path] | None = None,
        display: str = "progress",
        metadata: dict[str, Any] | None = None,
        openai_client: Any | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        caller = Path(inspect.stack()[1].filename).resolve()
        self.name = name
        self._source_file = caller if caller.exists() else None
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else caller.parent
        self.dataset = self._resolve_path(dataset)
        self.output_dir = self._resolve_path(output_dir)
        self.concurrency = concurrency
        self.resume = resume
        self.repetitions = repetitions
        self.max_retries = max_retries
        self.fail_fast = fail_fast
        self.capture_raw = capture_raw
        self.redact_raw_data_urls = redact_raw_data_urls
        self.timestamp_output_dir = timestamp_output_dir
        self.artifacts = tuple(artifacts or ())
        self.display = display
        self.metadata = metadata or {}
        self.openai_client = openai_client
        self._models: list[ModelConfig] = []
        self._workflow: WorkflowFn | None = None
        self._evals: list[EvalDefinition] = []
        self._output_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S") if timestamp_output_dir else None
        self._run_dir: Path | None = None

    @property
    def source_file(self) -> Path | None:
        return self._source_file

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def registered_models(self) -> list[ModelConfig]:
        return list(self._models)

    @property
    def registered_evals(self) -> list[EvalDefinition]:
        return list(self._evals)

    @property
    def workflow(self) -> WorkflowFn | None:
        return self._workflow

    @workflow.setter
    def workflow(self, func: WorkflowFn | None) -> None:
        if func is not None and not callable(func):
            raise TypeError("workflow must be callable")
        self._workflow = func

    def model(self, config: ModelConfig) -> ModelConfig:
        if any(existing.key == config.key for existing in self._models):
            raise ValueError(f"duplicate model key: {config.key}")
        self._models.append(config)
        return config

    def models(self, configs: Iterable[ModelConfig]) -> list[ModelConfig]:
        registered = []
        for config in configs:
            registered.append(self.model(config))
        return registered

    def eval(
        self,
        key: str,
        evaluator: EvalFn,
        *,
        description: str | None = None,
    ) -> EvalFn:
        if any(existing.key == key for existing in self._evals):
            raise ValueError(f"duplicate eval key: {key}")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._evals.append(EvalDefinition(key=key, func=evaluator, description=description))
        return evaluator

    def run(self) -> list[ItemRunRecord]:
        return asyncio.run(self.run_async())

    async def run_async(self) -> list[ItemRunRecord]:
        load_dotenv()
        runner = Runner(self)
        return await runner.run()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("experiment name must not be empty")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.display not in {"progress", "quiet", "debug"}:
            raise ValueError("display must be one of: progress, quiet, debug")
        if not self.dataset.exists():
            raise FileNotFoundError(f"dataset not found: {self.dataset}")
        if not self._models:
            raise ValueError("at least one model must be registered")
        if self._workflow is None:
            raise ValueError("a workflow callable must be assigned to exp.workflow")

    def run_dir(self) -> Path:
        if self._run_dir is None:
            dirname = f"{self._output_timestamp}_{self.name}" if self._output_timestamp else self.name
            self._run_dir = self.output_dir / dirname
        return self._run_dir

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self._base_dir / path).resolve()
