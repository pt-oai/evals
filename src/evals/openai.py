from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from evals._utils import to_jsonable
from evals.models import ErrorRecord, GenerationRecord, ModelConfig, TokenUsage


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
        row: dict[str, str],
        model: ModelConfig,
        execution_id: str,
        capture_raw: bool,
        max_retries: int,
    ) -> None:
        self.client = client
        self.row = row
        self.model = model
        self.execution_id = execution_id
        self.generations: list[GenerationRecord] = []
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


def exception_to_error(exc: BaseException) -> ErrorRecord:
    details: dict[str, Any] = {}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        details["status_code"] = status_code
    request_id = getattr(exc, "request_id", None)
    if request_id is not None:
        details["request_id"] = request_id
    code = getattr(exc, "code", None)
    if code is not None:
        details["code"] = code
    return ErrorRecord(type=exc.__class__.__name__, message=str(exc), details=details)


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

