from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evals.models import EvalResult

_UNSET = object()


class SelectorError(Exception):
    pass


@dataclass(frozen=True)
class Resolved:
    ok: bool
    value: Any = None
    error: str | None = None


@dataclass(frozen=True)
class Selector:
    source: str
    path: str | None = None
    cast: Callable[[Any], Any] | None = None
    default: Any = _UNSET

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> Any:
        resolved = self.resolve(dataset_item, model, output, ctx)
        if not resolved.ok:
            raise SelectorError(resolved.error or "selector failed")
        return resolved.value

    def resolve(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> Resolved:
        try:
            value, remaining_path = self._root(dataset_item, output, ctx)
            if remaining_path:
                value = resolve_path(value, remaining_path)
            elif self.path and self.source not in {"step", "step_text"}:
                value = resolve_path(value, self.path)
            if self.cast is not None:
                value = self.cast(value)
            return Resolved(ok=True, value=value)
        except Exception as exc:
            if self.default is not _UNSET:
                return Resolved(ok=True, value=self.default)
            return Resolved(ok=False, error=f"{self.label()} failed: {exc}")

    def label(self) -> str:
        cast_name = getattr(self.cast, "__name__", None)
        bits = [self.source]
        if self.path is not None:
            bits.append(self.path)
        if cast_name:
            bits.append(f"cast={cast_name}")
        return ".".join(bits)

    def _root(self, dataset_item: dict[str, str], output: Any, ctx: Any) -> tuple[Any, str | None]:
        if self.source == "item":
            return dataset_item, self.path
        if self.source == "out":
            return output.value, self.path
        if self.source == "text":
            return output.text, None
        if self.source == "step":
            step_key, remaining_path = split_step_path(self.path)
            return step_output_value(ctx, step_key), remaining_path
        if self.source == "step_text":
            step_key, remaining_path = split_step_path(self.path)
            value = step_output_text(ctx, step_key)
            return value, remaining_path
        raise SelectorError(f"unknown selector source: {self.source}")


def item(path: str, *, cast: Callable[[Any], Any] | None = None, default: Any = _UNSET) -> Selector:
    return Selector(source="item", path=path, cast=cast, default=default)


def out(
    path: str | None = None,
    *,
    cast: Callable[[Any], Any] | None = None,
    default: Any = _UNSET,
) -> Selector:
    return Selector(source="out", path=path, cast=cast, default=default)


def text(*, cast: Callable[[Any], Any] | None = None) -> Selector:
    return Selector(source="text", cast=cast)


def step(
    path: str,
    *,
    cast: Callable[[Any], Any] | None = None,
    default: Any = _UNSET,
) -> Selector:
    return Selector(source="step", path=path, cast=cast, default=default)


def step_text(
    path: str,
    *,
    cast: Callable[[Any], Any] | None = None,
    default: Any = _UNSET,
) -> Selector:
    return Selector(source="step_text", path=path, cast=cast, default=default)


class BuiltInEvaluator:
    operator = "builtin"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        raise NotImplementedError

    def _result(
        self,
        score: bool,
        *,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalResult:
        data = {"operator": self.operator}
        data.update(metadata or {})
        return EvalResult(score=score, data_type="BOOLEAN", comment=comment, metadata=data)


@dataclass(frozen=True)
class Equal(BuiltInEvaluator):
    actual: Any
    expected: Any
    operator = "equal"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        actual = resolve_operand(self.actual, dataset_item, model, output, ctx)
        expected = resolve_operand(self.expected, dataset_item, model, output, ctx)
        if not actual.ok or not expected.ok:
            return selector_failure(self.operator, actual, expected)
        score = actual.value == expected.value
        return self._result(
            score,
            comment=None if score else f"expected {short_repr(expected.value)}, got {short_repr(actual.value)}",
        )


@dataclass(frozen=True)
class NotEqual(BuiltInEvaluator):
    actual: Any
    expected: Any
    operator = "not_equal"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        actual = resolve_operand(self.actual, dataset_item, model, output, ctx)
        expected = resolve_operand(self.expected, dataset_item, model, output, ctx)
        if not actual.ok or not expected.ok:
            return selector_failure(self.operator, actual, expected)
        score = actual.value != expected.value
        return self._result(
            score,
            comment=None if score else f"did not expect {short_repr(actual.value)}",
        )


@dataclass(frozen=True)
class ApproxEqual(BuiltInEvaluator):
    actual: Any
    expected: Any
    abs_tol: float = 1e-6
    rel_tol: float = 1e-9
    operator = "approx_equal"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        actual = resolve_operand(self.actual, dataset_item, model, output, ctx)
        expected = resolve_operand(self.expected, dataset_item, model, output, ctx)
        if not actual.ok or not expected.ok:
            return selector_failure(self.operator, actual, expected)
        try:
            actual_float = float(actual.value)
            expected_float = float(expected.value)
        except Exception as exc:
            return self._result(False, comment=f"could not compare numeric values: {exc}")
        score = math.isclose(
            actual_float,
            expected_float,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )
        return self._result(
            score,
            comment=None
            if score
            else (
                f"expected {short_repr(expected_float)}, got {short_repr(actual_float)} "
                f"(abs_tol={self.abs_tol}, rel_tol={self.rel_tol})"
            ),
            metadata={"abs_tol": self.abs_tol, "rel_tol": self.rel_tol},
        )


@dataclass(frozen=True)
class Contains(BuiltInEvaluator):
    container: Any
    expected: Any
    case_sensitive: bool = True
    operator = "contains"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        container = resolve_operand(self.container, dataset_item, model, output, ctx)
        expected = resolve_operand(self.expected, dataset_item, model, output, ctx)
        if not container.ok or not expected.ok:
            return selector_failure(self.operator, container, expected)
        haystack = container.value
        needle = expected.value
        try:
            if isinstance(haystack, str) and isinstance(needle, str):
                left = haystack if self.case_sensitive else haystack.lower()
                right = needle if self.case_sensitive else needle.lower()
                score = right in left
            else:
                score = needle in haystack
        except Exception as exc:
            return self._result(False, comment=f"could not check containment: {exc}")
        return self._result(
            score,
            comment=None if score else f"{short_repr(needle)} not found",
            metadata={"case_sensitive": self.case_sensitive},
        )


@dataclass(frozen=True)
class RegexMatch(BuiltInEvaluator):
    value: Any
    pattern: str
    flags: int = 0
    operator = "regex_match"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        value = resolve_operand(self.value, dataset_item, model, output, ctx)
        if not value.ok:
            return selector_failure(self.operator, value)
        try:
            score = re.search(self.pattern, str(value.value), flags=self.flags) is not None
        except Exception as exc:
            return self._result(False, comment=f"invalid regex: {exc}", metadata={"pattern": self.pattern})
        return self._result(
            score,
            comment=None if score else f"pattern {short_repr(self.pattern)} did not match",
            metadata={"pattern": self.pattern, "flags": self.flags},
        )


@dataclass(frozen=True)
class NonEmpty(BuiltInEvaluator):
    value: Any
    operator = "non_empty"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        value = resolve_operand(self.value, dataset_item, model, output, ctx)
        if not value.ok:
            return selector_failure(self.operator, value)
        score = is_non_empty(value.value)
        return self._result(
            score,
            comment=None if score else f"value is empty: {short_repr(value.value)}",
        )


@dataclass(frozen=True)
class LengthBetween(BuiltInEvaluator):
    value: Any
    min_len: int | None = None
    max_len: int | None = None
    operator = "length_between"

    def __post_init__(self) -> None:
        if self.min_len is None and self.max_len is None:
            raise ValueError("LengthBetween requires min_len, max_len, or both")
        if self.min_len is not None and self.max_len is not None and self.min_len > self.max_len:
            raise ValueError("min_len cannot be greater than max_len")

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        value = resolve_operand(self.value, dataset_item, model, output, ctx)
        if not value.ok:
            return selector_failure(self.operator, value)
        try:
            length = len(value.value)
        except Exception as exc:
            return self._result(False, comment=f"could not measure length: {exc}")
        score = True
        if self.min_len is not None:
            score = score and length >= self.min_len
        if self.max_len is not None:
            score = score and length <= self.max_len
        return self._result(
            score,
            comment=None if score else f"length {length} outside bounds",
            metadata={"min_len": self.min_len, "max_len": self.max_len, "length": length},
        )


@dataclass(frozen=True)
class JsonPathExists(BuiltInEvaluator):
    value: Any
    path: str
    operator = "json_path_exists"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        value = resolve_json_value(self.value, dataset_item, model, output, ctx)
        if not value.ok:
            return selector_failure(self.operator, value)
        try:
            resolve_path(value.value, self.path)
        except Exception as exc:
            return self._result(False, comment=f"path {self.path!r} missing: {exc}", metadata={"path": self.path})
        return self._result(True, metadata={"path": self.path})


@dataclass(frozen=True)
class JsonPathEqual(BuiltInEvaluator):
    value: Any
    path: str
    expected: Any
    operator = "json_path_equal"

    def __call__(self, dataset_item: dict[str, str], model: Any, output: Any, ctx: Any) -> EvalResult:
        value = resolve_json_value(self.value, dataset_item, model, output, ctx)
        expected = resolve_operand(self.expected, dataset_item, model, output, ctx)
        if not value.ok or not expected.ok:
            return selector_failure(self.operator, value, expected)
        try:
            actual = resolve_path(value.value, self.path)
        except Exception as exc:
            return self._result(False, comment=f"path {self.path!r} missing: {exc}", metadata={"path": self.path})
        score = actual == expected.value
        return self._result(
            score,
            comment=None if score else f"expected {short_repr(expected.value)}, got {short_repr(actual)}",
            metadata={"path": self.path},
        )


def resolve_operand(
    operand: Any,
    dataset_item: dict[str, str],
    model: Any,
    output: Any,
    ctx: Any,
) -> Resolved:
    if isinstance(operand, Selector):
        return operand.resolve(dataset_item, model, output, ctx)
    return Resolved(ok=True, value=operand)


def resolve_json_value(
    operand: Any,
    dataset_item: dict[str, str],
    model: Any,
    output: Any,
    ctx: Any,
) -> Resolved:
    resolved = resolve_operand(operand, dataset_item, model, output, ctx)
    if not resolved.ok:
        return resolved
    value = resolved.value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception as exc:
            return Resolved(ok=False, error=f"could not parse JSON: {exc}")
    return Resolved(ok=True, value=value)


def selector_failure(operator: str, *resolved_values: Resolved) -> EvalResult:
    errors = [item.error for item in resolved_values if not item.ok and item.error]
    return EvalResult(
        score=False,
        data_type="BOOLEAN",
        comment="; ".join(errors) or "selector failed",
        metadata={"operator": operator},
    )


def split_step_path(path: str | None) -> tuple[str, str | None]:
    if not path:
        raise KeyError("step key is required")
    step_key, separator, remaining_path = path.partition(".")
    if not step_key:
        raise KeyError("step key is required")
    return step_key, remaining_path if separator else None


def step_output_value(ctx: Any, step_key: str) -> Any:
    output = get_step_output(ctx, step_key)
    return output.value


def step_output_text(ctx: Any, step_key: str) -> str:
    output = get_step_output(ctx, step_key)
    return output.text


def get_step_output(ctx: Any, step_key: str) -> Any:
    if ctx is None or not hasattr(ctx, "step_outputs"):
        raise KeyError("step outputs are unavailable")
    outputs = getattr(ctx, "step_outputs")
    if step_key not in outputs:
        raise KeyError(step_key)
    return outputs[step_key]


def resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if part == "":
            raise KeyError("empty path segment")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(part)
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(part) from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise KeyError(part) from exc
        else:
            raise KeyError(part)
    return current


def is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0
    except TypeError:
        return bool(value)


def short_repr(value: Any, *, max_length: int = 120) -> str:
    text = repr(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
