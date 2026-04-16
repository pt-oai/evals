from __future__ import annotations

import inspect
from typing import Any

from prism_evals.errors import exception_to_error
from prism_evals.models import EvalDefinition, EvalResult, ModelConfig, TaskOutput, infer_score_type


async def run_eval_definitions(
    definitions: list[EvalDefinition],
    *,
    item: dict[str, str],
    model: ModelConfig,
    output: TaskOutput,
    ctx: Any,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for definition in definitions:
        try:
            value = definition.func(item, model, output, ctx)
            if inspect.isawaitable(value):
                value = await value
            results.extend(normalize_eval_return(value, definition))
        except Exception as exc:
            results.append(
                EvalResult(
                    key=definition.key,
                    description=definition.description,
                    error=exception_to_error(exc),
                )
            )
    return results


def has_eval_errors(results: list[EvalResult]) -> bool:
    return any(result.error is not None for result in results)


def eval_definitions_from_specs(specs: list[tuple[Any, ...]] | None) -> list[EvalDefinition]:
    if not specs:
        return []
    definitions: list[EvalDefinition] = []
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, tuple) or len(spec) not in {2, 3}:
            raise TypeError("step evals must be (key, evaluator) or (key, evaluator, description) tuples")
        key = str(spec[0])
        evaluator = spec[1]
        description = spec[2] if len(spec) == 3 else None
        if not key.strip():
            raise ValueError("step eval key must not be empty")
        if key in seen:
            raise ValueError(f"duplicate step eval key: {key}")
        if not callable(evaluator):
            raise TypeError("step evaluator must be callable")
        seen.add(key)
        definitions.append(EvalDefinition(key=key, func=evaluator, description=description))
    return definitions


def normalize_eval_return(value: Any, definition: EvalDefinition) -> list[EvalResult]:
    if value is None:
        return []
    if isinstance(value, EvalResult):
        return [apply_registered_key(value, definition.key, definition.description)]
    if isinstance(value, (bool, int, float)):
        return [EvalResult(score=value).with_defaults(definition.key, definition.description)]
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
    raise TypeError("eval must return None, bool, int, float, dict, EvalResult, or list[EvalResult]")


def apply_registered_key(
    result: EvalResult,
    key: str,
    description: str | None = None,
) -> EvalResult:
    data = result.model_dump()
    data["key"] = key
    data["description"] = data["description"] or description
    if data["data_type"] is None and data["score"] is not None:
        data["data_type"] = infer_score_type(data["score"])
    return EvalResult(**data)
