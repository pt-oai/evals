from __future__ import annotations

import inspect
import time
from typing import Any

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from evals._utils import to_jsonable, utc_now_iso
from evals.errors import exception_to_error
from evals.evaluation import eval_definitions_from_specs, has_eval_errors, run_eval_definitions
from evals.models import GenerationRecord, ModelConfig, StepRecord, TaskOutput, TokenUsage


class ResponsesProxy:
    def __init__(
        self,
        client: Any,
        generations: list[GenerationRecord],
        *,
        capture_raw: bool,
        max_retries: int,
    ) -> None:
        self._client = client
        self._generations = generations
        self._capture_raw = capture_raw
        self._max_retries = max(1, max_retries)

    async def create(self, **kwargs: Any) -> Any:
        raw_request = to_jsonable(kwargs) if self._capture_raw else None
        started = time.perf_counter()
        try:
            response = await self._call_with_retries(**kwargs)
        except Exception as exc:
            latency = time.perf_counter() - started
            self._generations.append(
                GenerationRecord(
                    latency_s=latency,
                    raw_request=raw_request,
                    error=exception_to_error(exc),
                )
            )
            raise

        latency = time.perf_counter() - started
        raw_response = to_jsonable(response) if self._capture_raw else None
        self._generations.append(
            GenerationRecord(
                response_id=getattr(response, "id", None),
                latency_s=latency,
                usage=extract_usage(response),
                raw_request=raw_request,
                raw_response=raw_response,
                output_text=str(getattr(response, "output_text", "") or ""),
            )
        )
        return response

    async def _call_with_retries(self, **kwargs: Any) -> Any:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(is_retryable_openai_error),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            reraise=True,
        ):
            with attempt:
                return await self._client.responses.create(**kwargs)
        raise RuntimeError("unreachable retry state")


class ExperimentContext:
    def __init__(
        self,
        *,
        client: Any,
        item: dict[str, str],
        model: ModelConfig,
        item_run_id: str,
        capture_raw: bool,
        max_retries: int,
        fail_fast: bool = False,
    ) -> None:
        self.client = client
        self.item = item
        self.model = model
        self.item_run_id = item_run_id
        self.fail_fast = fail_fast
        self.generations: list[GenerationRecord] = []
        self.steps: list[StepRecord] = []
        self.step_outputs: dict[str, TaskOutput] = {}
        self._step_keys: set[str] = set()
        self.responses = ResponsesProxy(
            client,
            self.generations,
            capture_raw=capture_raw,
            max_retries=max_retries,
        )

    @property
    def usage(self) -> TokenUsage:
        usage = TokenUsage()
        for generation in self.generations:
            usage += generation.usage
        return usage

    async def step(
        self,
        key: str,
        callable_or_value: Any,
        *,
        evals: list[tuple[Any, ...]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskOutput:
        if not key.strip():
            raise ValueError("step key must not be empty")
        if key in self._step_keys:
            raise ValueError(f"duplicate step key: {key}")
        self._step_keys.add(key)

        eval_definitions = eval_definitions_from_specs(evals)
        generation_start = len(self.generations)
        started_at = utc_now_iso()
        started = time.perf_counter()
        output: TaskOutput | None = None
        step_evals = []
        error = None
        status = "success"

        try:
            value = callable_or_value() if callable(callable_or_value) else callable_or_value
            if inspect.isawaitable(value):
                value = await value
            output = TaskOutput.normalize(value)
            self.step_outputs[key] = output
            step_evals = await run_eval_definitions(
                eval_definitions,
                item=self.item,
                model=self.model,
                output=output,
                ctx=self,
            )
            if self.fail_fast and has_eval_errors(step_evals):
                raise RuntimeError(f"step eval failed: {key}")
            return output
        except Exception as exc:
            status = "failed"
            error = exception_to_error(exc)
            raise
        finally:
            ended_at = utc_now_iso()
            step_generations = self.generations[generation_start:]
            usage = TokenUsage()
            response_id = None
            for generation in step_generations:
                usage += generation.usage
            for generation in reversed(step_generations):
                if generation.response_id:
                    response_id = generation.response_id
                    break
            self.steps.append(
                StepRecord(
                    key=key,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_s=time.perf_counter() - started,
                    output=output,
                    evals=step_evals,
                    usage=usage,
                    response_id=response_id,
                    generations=step_generations,
                    error=error,
                    metadata=metadata or {},
                )
            )


def make_default_client() -> AsyncOpenAI:
    return AsyncOpenAI()


def extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return TokenUsage()

    input_details = _get(usage, "input_tokens_details", {}) or {}
    output_details = _get(usage, "output_tokens_details", {}) or {}
    return TokenUsage(
        input_tokens=int(_get(usage, "input_tokens", 0) or 0),
        cached_tokens=int(_get(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(_get(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(_get(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=int(_get(usage, "total_tokens", 0) or 0),
    )


def is_retryable_openai_error(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
