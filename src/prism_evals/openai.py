from __future__ import annotations

import base64
import hashlib
import inspect
import re
import shutil
import time
from pathlib import Path
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from prism_evals._utils import raw_payload, utc_now_iso
from prism_evals.errors import exception_to_error
from prism_evals.evaluation import eval_definitions_from_specs, has_eval_errors, run_eval_definitions
from prism_evals.models import (
    GenerationRecord,
    MediaArtifact,
    ModelConfig,
    StepRecord,
    TaskOutput,
    TokenUsage,
    require_task_output,
)


class ResponsesProxy:
    def __init__(
        self,
        client: Any,
        generations: list[GenerationRecord],
        *,
        capture_raw: bool,
        max_retries: int,
        redact_raw_data_urls: bool = True,
    ) -> None:
        self._client = client
        self._generations = generations
        self._capture_raw = capture_raw
        self._redact_raw_data_urls = redact_raw_data_urls
        self._max_retries = max(1, max_retries)

    async def create(self, **kwargs: Any) -> Any:
        raw_request = (
            raw_payload(kwargs, redact_raw_data_urls=self._redact_raw_data_urls)
            if self._capture_raw
            else None
        )
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
        raw_response = (
            raw_payload(response, redact_raw_data_urls=self._redact_raw_data_urls)
            if self._capture_raw
            else None
        )
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
        run_dir: Path,
        capture_raw: bool,
        max_retries: int,
        redact_raw_data_urls: bool = True,
        fail_fast: bool = False,
    ) -> None:
        self.client = client
        self.item = item
        self.model = model
        self.item_run_id = item_run_id
        self.fail_fast = fail_fast
        self.media = MediaStore(run_dir=run_dir, item_run_id=item_run_id)
        self.generations: list[GenerationRecord] = []
        self.steps: list[StepRecord] = []
        self.step_outputs: dict[str, TaskOutput] = {}
        self._step_keys: set[str] = set()
        self.responses = ResponsesProxy(
            client,
            self.generations,
            capture_raw=capture_raw,
            max_retries=max_retries,
            redact_raw_data_urls=redact_raw_data_urls,
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
            output = require_task_output(value, context=f"step {key!r}")
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


class MediaStore:
    def __init__(self, *, run_dir: Path, item_run_id: str) -> None:
        self.run_dir = run_dir
        self.media_dir = run_dir / "media"
        self.item_run_id = item_run_id
        self._counter = 0

    def from_base64(
        self,
        data: str,
        *,
        format: str = "png",
        name: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaArtifact:
        payload = data.strip()
        if payload.startswith("data:"):
            header, separator, body = payload.partition(",")
            if not separator:
                raise ValueError("base64 media data URL is missing a comma separator")
            payload = body
            media_type = header[5:].split(";", 1)[0]
            if media_type:
                mime_type = mime_type or media_type
                if format == "png":
                    format = media_type.rsplit("/", 1)[-1] or format
        content = base64.b64decode("".join(payload.split()), validate=True)
        return self.from_bytes(
            content,
            format=format,
            name=name,
            mime_type=mime_type,
            metadata=metadata,
        )

    def from_bytes(
        self,
        data: bytes,
        *,
        format: str = "png",
        name: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaArtifact:
        media_format = normalize_media_format(format)
        media_mime_type = mime_type or mime_type_for_format(media_format)
        digest = hashlib.sha256(data).hexdigest()
        target = self._next_path(media_format, name=name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        relative_path = target.relative_to(self.run_dir).as_posix()
        return MediaArtifact(
            path=relative_path,
            mime_type=media_mime_type,
            format=media_format,
            sha256=digest,
            bytes=len(data),
            metadata=dict(metadata or {}),
        )

    def from_path(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        name: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MediaArtifact:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"media source file not found: {source}")
        media_format = normalize_media_format(format or source.suffix.lstrip(".") or "png")
        target = self._next_path(media_format, name=name or source.stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        relative_path = target.relative_to(self.run_dir).as_posix()
        return MediaArtifact(
            path=relative_path,
            mime_type=mime_type or mime_type_for_format(media_format),
            format=media_format,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            metadata=dict(metadata or {}),
        )

    def _next_path(self, media_format: str, *, name: str | None) -> Path:
        self._counter += 1
        safe_name = sanitize_media_name(name)
        stem = f"{self.item_run_id}-{self._counter:03d}"
        if safe_name:
            stem = f"{stem}-{safe_name}"
        return self.media_dir / f"{stem}.{media_format}"


def normalize_media_format(value: str) -> str:
    media_format = value.lower().strip().lstrip(".")
    if media_format == "jpg":
        return "jpeg"
    if not media_format:
        raise ValueError("media format must not be empty")
    return media_format


def mime_type_for_format(media_format: str) -> str:
    if media_format in {"png", "jpeg", "webp", "gif"}:
        return f"image/{media_format}"
    return "application/octet-stream"


def sanitize_media_name(name: str | None) -> str:
    if not name:
        return ""
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip(".-")[:80]


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
