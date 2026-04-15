from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from evals.models import ItemRunRecord, ModelConfig


@dataclass
class UsageAggregate:
    count: int = 0
    totals: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])


class ConsoleReporter:
    def __init__(self, display: str = "progress") -> None:
        self.display = display
        self.console = Console()
        self.progress: Progress | None = None
        self.overall_task: TaskID | None = None
        self.model_tasks: dict[str, TaskID] = {}
        self.completed = 0
        self.failed = 0

    def start(
        self,
        *,
        experiment_name: str,
        item_count: int,
        model_count: int,
        repetitions: int,
        total_item_runs: int,
        remaining_item_runs: int,
        skipped: int,
        run_dir: Path,
        models: list[ModelConfig],
        remaining_by_model: dict[str, int],
    ) -> None:
        if self.display == "quiet":
            return
        summary = (
            f"[bold]{experiment_name}[/bold]\n"
            f"Items: {item_count}\n"
            f"Models: {model_count}\n"
            f"Repetitions: {repetitions}\n"
            f"Item runs: {total_item_runs}"
        )
        if skipped:
            summary += f"\nResumed: {skipped} already complete"
        summary += f"\nOutput: {run_dir}"
        self.console.print(Panel(summary, title="evals", border_style="cyan"))

        if self.display in {"progress", "debug"}:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=False,
            )
            self.progress.start()
            self.overall_task = self.progress.add_task("item runs", total=remaining_item_runs)
            for model in models:
                self.model_tasks[model.key] = self.progress.add_task(
                    f"model:{model.key}", total=remaining_by_model.get(model.key, 0)
                )

    def record(self, record: ItemRunRecord) -> None:
        if record.status == "failed":
            self.failed += 1
        else:
            self.completed += 1

        if self.progress is not None:
            if self.overall_task is not None:
                self.progress.advance(self.overall_task)
                self.progress.update(
                    self.overall_task,
                    description=f"overall ok={self.completed} failed={self.failed}",
                )
            model_task = self.model_tasks.get(record.model_key)
            if model_task is not None:
                self.progress.advance(model_task)

        if self.display == "debug":
            style = "red" if record.status == "failed" else "green"
            self.console.print(
                f"[{style}]{record.status}[/{style}] "
                f"item={record.item_id} model={record.model_key} repetition={record.repetition}"
            )

        if record.status == "failed" and self.display != "quiet":
            message = record.error.message if record.error else "item run failed"
            failed_step = first_failed_step(record)
            step_part = f" step={failed_step.key}" if failed_step else ""
            self.console.print(
                f"[red]Failed[/red] item={record.item_id} "
                f"model={record.model_key} repetition={record.repetition}{step_part}: {message}"
            )

    def finish(self, records: list[ItemRunRecord], artifacts: dict[str, Path]) -> None:
        if self.progress is not None:
            self.progress.stop()
        if self.display == "quiet":
            return
        self.console.print()
        self._print_score_table(records)
        self._print_usage_table(records)
        self._print_latency_table(records)
        self._print_failure_table(records)
        self._print_artifacts(artifacts)

    def _print_score_table(self, records: list[ItemRunRecord]) -> None:
        score_rows: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        for record in records:
            for result in record.evals:
                if result.key and result.score is not None and isinstance(result.score, (bool, int, float)):
                    score_rows[("item_run", "", record.model_key, result.key)].append(float(result.score))
            for step in record.steps:
                for result in step.evals:
                    if result.key and result.score is not None and isinstance(result.score, (bool, int, float)):
                        score_rows[("step", step.key, record.model_key, result.key)].append(float(result.score))
        table = Table(title="Scores By Model")
        table.add_column("Scope")
        table.add_column("Step")
        table.add_column("Model")
        table.add_column("Score")
        table.add_column("Mean", justify="right")
        table.add_column("Count", justify="right")
        if not score_rows:
            table.add_row("-", "-", "-", "-", "-", "0")
        else:
            for (scope, step_key, model_key, score_key), values in sorted(score_rows.items()):
                mean = sum(values) / len(values)
                table.add_row(scope, step_key or "-", model_key, score_key, f"{mean:.3f}", str(len(values)))
        self.console.print(table)

    def _print_usage_table(self, records: list[ItemRunRecord]) -> None:
        usage = aggregate_usage_by_model(records)
        table = Table(title="Token Usage By Model")
        table.add_column("Model")
        table.add_column("Item Runs", justify="right")
        table.add_column("Metric")
        table.add_column("Input", justify="right")
        table.add_column("Cached", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Reasoning", justify="right")
        table.add_column("Total", justify="right")
        if not usage:
            table.add_row("-", "0", "avg/item run", "0.0", "0.0", "0.0", "0.0", "0.0")
            table.add_row("-", "0", "total", "0", "0", "0", "0", "0")
        else:
            for model_key, stats in sorted(usage.items()):
                table.add_row(
                    model_key,
                    str(stats.count),
                    "avg/item run",
                    *(f"{value / stats.count:.1f}" for value in stats.totals),
                )
                table.add_row(model_key, str(stats.count), "total", *(str(value) for value in stats.totals))
        self.console.print(table)

    def _print_latency_table(self, records: list[ItemRunRecord]) -> None:
        latencies: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record.status == "success":
                latencies[record.model_key].append(record.duration_s)
        table = Table(title="Latency")
        table.add_column("Model")
        table.add_column("Avg", justify="right")
        table.add_column("P50", justify="right")
        table.add_column("P95", justify="right")
        table.add_column("Max", justify="right")
        if not latencies:
            table.add_row("-", "-", "-", "-", "-")
        else:
            for model_key, values in sorted(latencies.items()):
                values = sorted(values)
                avg = sum(values) / len(values)
                table.add_row(
                    model_key,
                    f"{avg:.3f}s",
                    f"{percentile(values, 50):.3f}s",
                    f"{percentile(values, 95):.3f}s",
                    f"{max(values):.3f}s",
                )
        self.console.print(table)

    def _print_failure_table(self, records: list[ItemRunRecord]) -> None:
        failures = [record for record in records if record.status == "failed"]
        table = Table(title="Failures")
        table.add_column("Item")
        table.add_column("Model")
        table.add_column("Repetition")
        table.add_column("Step")
        table.add_column("Error")
        if not failures:
            table.add_row("-", "-", "-", "-", "None")
        else:
            for record in failures[:20]:
                failed_step = first_failed_step(record)
                table.add_row(
                    record.item_id,
                    record.model_key,
                    str(record.repetition),
                    failed_step.key if failed_step else "-",
                    record.error.message if record.error else "item run failed",
                )
        self.console.print(table)

    def _print_artifacts(self, artifacts: dict[str, Path]) -> None:
        table = Table(title="Artifacts")
        table.add_column("Name")
        table.add_column("Path")
        for name, path in artifacts.items():
            table.add_row(name, str(path))
        self.console.print(table)


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * (percent / 100)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def first_failed_step(record: ItemRunRecord):
    for step in record.steps:
        if step.status == "failed":
            return step
    return None


def aggregate_usage_by_model(records: list[ItemRunRecord]) -> dict[str, UsageAggregate]:
    usage: dict[str, UsageAggregate] = defaultdict(UsageAggregate)
    for record in records:
        stats = usage[record.model_key]
        stats.count += 1
        totals = stats.totals
        totals[0] += record.usage.input_tokens
        totals[1] += record.usage.cached_tokens
        totals[2] += record.usage.output_tokens
        totals[3] += record.usage.reasoning_tokens
        totals[4] += record.usage.total_tokens
    return dict(usage)
