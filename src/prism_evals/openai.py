from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import re
import shutil
import time
import wave
from dataclasses import dataclass, field
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
    ModelVariant,
    StepRecord,
    TaskOutput,
    TokenUsage,
    ToolCallRecord,
    TurnRecord,
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


@dataclass(frozen=True)
class RealtimeRunResult:
    response_id: str | None
    status: str | None
    text: str = ""
    transcript: str = ""
    audio: bytes = b""
    media: list[MediaArtifact] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    events: list[Any] = field(default_factory=list)
    response: Any | None = None

    def task_output(self) -> TaskOutput:
        output_text = self.text or self.transcript
        return TaskOutput(
            text=output_text,
            value={
                "response_id": self.response_id,
                "status": self.status,
                "text": self.text,
                "transcript": self.transcript,
                "audio_bytes": len(self.audio),
                "event_count": len(self.events),
                "tool_call_count": len(self.tool_calls),
                "tool_calls": self.tool_calls,
            },
            media=self.media,
            metadata={
                "realtime_response_id": self.response_id,
                "tool_call_count": len(self.tool_calls),
            },
        )


class RealtimeProxy:
    def __init__(
        self,
        client: Any,
        generations: list[GenerationRecord],
        media: "MediaStore",
        *,
        model: ModelConfig,
        capture_raw: bool,
        max_retries: int,
        redact_raw_data_urls: bool = True,
    ) -> None:
        self._client = client
        self._generations = generations
        self._media = media
        self._model = model
        self._capture_raw = capture_raw
        self._redact_raw_data_urls = redact_raw_data_urls
        self._max_retries = max(1, max_retries)

    def connect(self, *, model: str | None = None, **kwargs: Any) -> Any:
        return self._client.realtime.connect(model=model or self._model.model, **kwargs)

    async def run_text(
        self,
        text: str,
        *,
        model: str | None = None,
        instructions: str | None = None,
        session: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        timeout_s: float = 60.0,
    ) -> RealtimeRunResult:
        async def add_text_input(connection: Any) -> None:
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }
            )

        return await self._run(
            model=model,
            instructions=instructions,
            session=session,
            response=response,
            output_modalities=["text"],
            setup_input=add_text_input,
            timeout_s=timeout_s,
            raw_request={"input_text": text},
        )

    async def run_audio(
        self,
        audio: bytes | str | Path,
        *,
        model: str | None = None,
        instructions: str | None = None,
        session: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        input_rate: int = 24000,
        output_rate: int = 24000,
        chunk_bytes: int = 15 * 1024 * 1024,
        output_name: str | None = None,
        timeout_s: float = 60.0,
    ) -> RealtimeRunResult:
        pcm_audio, input_rate = pcm16_audio(audio, default_rate=input_rate)

        async def add_audio_input(connection: Any) -> None:
            for start in range(0, len(pcm_audio), chunk_bytes):
                chunk = pcm_audio[start : start + chunk_bytes]
                await connection.input_audio_buffer.append(audio=base64.b64encode(chunk).decode("ascii"))
            await connection.input_audio_buffer.commit()

        result = await self._run(
            model=model,
            instructions=instructions,
            session=session,
            response=response,
            output_modalities=["audio", "text"],
            setup_input=add_audio_input,
            timeout_s=timeout_s,
            raw_request={"input_audio_bytes": len(pcm_audio), "input_rate": input_rate},
        )
        if not result.audio:
            return result
        wav_audio = wav_from_pcm16(result.audio, sample_rate=output_rate)
        artifact = self._media.from_bytes(
            wav_audio,
            format="wav",
            name=output_name or "realtime-output",
            mime_type="audio/wav",
            metadata={"kind": "realtime_audio", "sample_rate": output_rate},
        )
        return RealtimeRunResult(
            response_id=result.response_id,
            status=result.status,
            text=result.text,
            transcript=result.transcript,
            audio=result.audio,
            media=[artifact],
            tool_calls=result.tool_calls,
            usage=result.usage,
            events=result.events,
            response=result.response,
        )

    async def _run(
        self,
        *,
        model: str | None,
        instructions: str | None,
        session: dict[str, Any] | None,
        response: dict[str, Any] | None,
        output_modalities: list[str],
        setup_input: Any,
        timeout_s: float,
        raw_request: dict[str, Any],
    ) -> RealtimeRunResult:
        started = time.perf_counter()
        model_name = model or self._model.model
        session_payload = {
            "type": "realtime",
            "model": model_name,
            "output_modalities": output_modalities,
            **self._model.params,
            **dict(session or {}),
        }
        if instructions is not None:
            session_payload["instructions"] = instructions
        response_payload = {"output_modalities": output_modalities, **dict(response or {})}
        sent_events = [
            {"type": "session.update", "session": session_payload},
            raw_request,
            {"type": "response.create", "response": response_payload},
        ]
        events: list[Any] = []
        text_parts: list[str] = []
        transcript_parts: list[str] = []
        audio_parts: list[bytes] = []
        response_done = None
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(is_retryable_openai_error),
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential_jitter(initial=0.5, max=8.0),
                reraise=True,
            ):
                with attempt:
                    async with self.connect(model=model_name) as connection:
                        await connection.session.update(session=session_payload)
                        await setup_input(connection)
                        await connection.response.create(response=response_payload)
                        async with asyncio.timeout(timeout_s):
                            async for event in connection:
                                events.append(event)
                                event_type = _get(event, "type", "")
                                if event_type == "response.output_text.delta":
                                    text_parts.append(str(_get(event, "delta", "")))
                                elif event_type == "response.output_audio_transcript.delta":
                                    transcript_parts.append(str(_get(event, "delta", "")))
                                elif event_type == "response.output_audio.delta":
                                    audio_parts.append(base64.b64decode(str(_get(event, "delta", ""))))
                                elif event_type == "error":
                                    message = _get(_get(event, "error", {}), "message", "Realtime API error")
                                    raise RuntimeError(str(message))
                                elif event_type == "response.done":
                                    response_done = _get(event, "response")
                                    break
                        if response_done is None:
                            raise RuntimeError("Realtime response stream ended before response.done")
                    break
        except Exception as exc:
            self._generations.append(
                GenerationRecord(
                    latency_s=time.perf_counter() - started,
                    raw_request=raw_payload(sent_events, redact_raw_data_urls=self._redact_raw_data_urls)
                    if self._capture_raw
                    else None,
                    raw_response=redact_realtime_events(events) if self._capture_raw else None,
                    error=exception_to_error(exc),
                    metadata={"api": "realtime", "event_count": len(events)},
                )
            )
            raise

        response_id = _get(response_done, "id")
        status = _get(response_done, "status")
        usage = extract_usage(response_done)
        output_text = "".join(text_parts)
        transcript = "".join(transcript_parts)
        audio = b"".join(audio_parts)
        tool_calls = parse_realtime_tool_calls(events, response_done)
        self._generations.append(
            GenerationRecord(
                response_id=response_id,
                latency_s=time.perf_counter() - started,
                usage=usage,
                raw_request=raw_payload(sent_events, redact_raw_data_urls=self._redact_raw_data_urls)
                if self._capture_raw
                else None,
                raw_response=redact_realtime_events(events) if self._capture_raw else None,
                output_text=output_text or transcript,
                metadata={
                    "api": "realtime",
                    "status": status,
                    "event_count": len(events),
                    "audio_bytes": len(audio),
                    "transcript_chars": len(transcript),
                    "tool_call_count": len(tool_calls),
                    "tool_call_names": [
                        call["name"] for call in tool_calls if isinstance(call.get("name"), str)
                    ],
                },
            )
        )
        return RealtimeRunResult(
            response_id=response_id,
            status=status,
            text=output_text,
            transcript=transcript,
            audio=audio,
            tool_calls=tool_calls,
            usage=usage,
            events=events,
            response=response_done,
        )


class ExperimentContext:
    def __init__(
        self,
        *,
        client: Any,
        item: dict[str, Any],
        model: ModelConfig | ModelVariant,
        item_run_id: str,
        run_dir: Path,
        capture_raw: bool,
        max_retries: int,
        redact_raw_data_urls: bool = True,
        fail_fast: bool = False,
    ) -> None:
        self.client = client
        self.item = item
        self.model = ModelResolver(model)
        self.item_run_id = item_run_id
        self.fail_fast = fail_fast
        self.media = MediaStore(run_dir=run_dir, item_run_id=item_run_id)
        self.generations: list[GenerationRecord] = []
        self.steps: list[StepRecord] = []
        self.turns: list[TurnRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.step_outputs: dict[str, TaskOutput] = {}
        self.turn_outputs: dict[str, TaskOutput] = {}
        self._step_keys: set[str] = set()
        self._turn_keys: set[str] = set()
        self._active_turn_stack: list[str] = []
        self.responses = ResponsesProxy(
            client,
            self.generations,
            capture_raw=capture_raw,
            max_retries=max_retries,
            redact_raw_data_urls=redact_raw_data_urls,
        )
        self.realtime = RealtimeProxy(
            client,
            self.generations,
            self.media,
            model=self.model.default,
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
        tool_call_start = len(self.tool_calls)
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
            step_tool_calls = self.tool_calls[tool_call_start:]
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
                    tool_calls=step_tool_calls,
                    error=error,
                    metadata=metadata or {},
                )
            )

    async def turn(
        self,
        key: str,
        callable_or_value: Any,
        *,
        role: str = "assistant",
        mode: str | None = None,
        input: Any | None = None,
        evals: list[tuple[Any, ...]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskOutput:
        if not key.strip():
            raise ValueError("turn key must not be empty")
        if key in self._turn_keys:
            raise ValueError(f"duplicate turn key: {key}")
        self._turn_keys.add(key)

        eval_definitions = eval_definitions_from_specs(evals)
        tool_call_start = len(self.tool_calls)
        started_at = utc_now_iso()
        started = time.perf_counter()
        output: TaskOutput | None = None
        turn_evals = []
        error = None
        status = "success"

        self._active_turn_stack.append(key)
        try:
            output = await self.step(
                f"turn:{key}",
                callable_or_value,
                evals=evals,
                metadata={"turn_id": key, "role": role, **dict(metadata or {})},
            )
            self.turn_outputs[key] = output
            # Step evals have already run; re-use them on the turn record for easy filtering.
            if self.steps and self.steps[-1].key == f"turn:{key}":
                turn_evals = self.steps[-1].evals
            elif eval_definitions:
                turn_evals = await run_eval_definitions(
                    eval_definitions,
                    item=self.item,
                    model=self.model.default,
                    output=output,
                    ctx=self,
                )
            return output
        except Exception as exc:
            status = "failed"
            error = exception_to_error(exc)
            raise
        finally:
            if self._active_turn_stack and self._active_turn_stack[-1] == key:
                self._active_turn_stack.pop()
            ended_at = utc_now_iso()
            self.turns.append(
                TurnRecord(
                    id=key,
                    role=role,
                    mode=mode,
                    status=status,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_s=time.perf_counter() - started,
                    input=input,
                    output=output,
                    evals=turn_evals,
                    tool_calls=self.tool_calls[tool_call_start:],
                    error=error,
                    metadata=metadata or {},
                )
            )

    def record_turn(
        self,
        key: str,
        *,
        role: str,
        content: str | None = None,
        mode: str | None = None,
        input: Any | None = None,
        output: TaskOutput | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskOutput:
        if not key.strip():
            raise ValueError("turn key must not be empty")
        if key in self._turn_keys:
            raise ValueError(f"duplicate turn key: {key}")
        self._turn_keys.add(key)
        started_at = utc_now_iso()
        final_output = output or TaskOutput(
            text=content or "",
            value={
                "role": role,
                "mode": mode,
                "content": content or "",
            },
        )
        self.turn_outputs[key] = final_output
        self.turns.append(
            TurnRecord(
                id=key,
                role=role,
                mode=mode,
                started_at=started_at,
                ended_at=utc_now_iso(),
                input=input,
                output=final_output,
                metadata=metadata or {},
            )
        )
        return final_output

    def user(self, key: str, content: str, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.record_turn(key, role="user", content=content, mode="seed", metadata=metadata)

    def assistant_seed(self, key: str, content: str, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.record_turn(key, role="assistant", content=content, mode="seed", metadata=metadata)

    def action_seed(self, key: str, action: Any, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.record_turn(
            key,
            role="action",
            mode="seed",
            input=action,
            output=TaskOutput(text="", value={"role": "action", "mode": "seed", "action": action}),
            metadata=metadata,
        )

    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Any | None = None,
        result: Any | None = None,
        agent: str | None = None,
        turn: str | None = None,
        call_id: str | None = None,
        status: str = "success",
        duration_s: float = 0.0,
        error: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCallRecord:
        now = utc_now_iso()
        record = ToolCallRecord(
            name=name,
            arguments=arguments,
            result=result,
            agent=agent,
            turn_id=turn or (self._active_turn_stack[-1] if self._active_turn_stack else None),
            call_id=call_id,
            status=status,
            duration_s=duration_s,
            started_at=now,
            ended_at=now,
            error=exception_to_error(error) if isinstance(error, BaseException) else error,
            metadata=metadata or {},
        )
        self.tool_calls.append(record)
        return record

    def conversation(self, scenario: Any | None = None, *, id: str | None = None) -> "ConversationRecorder":
        return ConversationRecorder(self, scenario=scenario, conversation_id=id)


class ModelResolver:
    def __init__(self, model: ModelConfig | ModelVariant) -> None:
        if isinstance(model, ModelVariant):
            self.variant = model
        else:
            self.variant = ModelVariant(key=model.key, models={"default": model}, default_role="default")

    def __call__(self, role: str = "default") -> ModelConfig:
        return self.variant.model_for(role)

    @property
    def default(self) -> ModelConfig:
        return self.variant.default_model

    @property
    def key(self) -> str:
        return self.variant.key

    @property
    def model(self) -> str:
        return self.variant.model

    @property
    def params(self) -> dict[str, Any]:
        return self.variant.params

    @property
    def models(self) -> dict[str, ModelConfig]:
        return self.variant.models


class ConversationRecorder:
    def __init__(self, ctx: ExperimentContext, *, scenario: Any | None, conversation_id: str | None = None) -> None:
        self.ctx = ctx
        self.scenario = scenario if isinstance(scenario, dict) else {}
        self.id = conversation_id or str(self.scenario.get("id") or "")

    async def __aenter__(self) -> "ConversationRecorder":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def user(self, key: str, content: str, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.ctx.user(key, content, metadata=metadata)

    def assistant_seed(self, key: str, content: str, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.ctx.assistant_seed(key, content, metadata=metadata)

    def action_seed(self, key: str, action: Any, *, metadata: dict[str, Any] | None = None) -> TaskOutput:
        return self.ctx.action_seed(key, action, metadata=metadata)

    async def turn(
        self,
        key: str,
        callable_or_value: Any,
        *,
        role: str = "assistant",
        mode: str | None = None,
        input: Any | None = None,
        evals: list[tuple[Any, ...]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskOutput:
        return await self.ctx.turn(
            key,
            callable_or_value,
            role=role,
            mode=mode,
            input=input,
            evals=evals,
            metadata=metadata,
        )

    def task_output(self) -> TaskOutput:
        return TaskOutput(
            text=last_turn_text(self.ctx.turns),
            value={
                "conversation_id": self.id,
                "turns": [turn.model_dump(mode="json") for turn in self.ctx.turns],
                "tool_calls": [call.model_dump(mode="json") for call in self.ctx.tool_calls],
            },
        )


def last_turn_text(turns: list[TurnRecord]) -> str:
    for turn in reversed(turns):
        if turn.output and turn.output.text:
            return turn.output.text
    return ""


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
    if media_format == "wav":
        return "audio/wav"
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

    input_details = _get(usage, "input_tokens_details", None) or _get(usage, "input_token_details", {}) or {}
    output_details = _get(usage, "output_tokens_details", None) or _get(usage, "output_token_details", {}) or {}
    return TokenUsage(
        input_tokens=int(_get(usage, "input_tokens", 0) or 0),
        cached_tokens=int(_get(input_details, "cached_tokens", 0) or 0),
        output_tokens=int(_get(usage, "output_tokens", 0) or 0),
        reasoning_tokens=int(_get(output_details, "reasoning_tokens", 0) or 0),
        total_tokens=int(_get(usage, "total_tokens", 0) or 0),
    )


def pcm16_audio(audio: bytes | str | Path, *, default_rate: int) -> tuple[bytes, int]:
    if isinstance(audio, bytes):
        return audio, default_rate
    path = Path(audio).expanduser().resolve()
    if path.suffix.lower() != ".wav":
        return path.read_bytes(), default_rate
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError("Realtime audio fixtures must be mono WAV files")
        if handle.getsampwidth() != 2:
            raise ValueError("Realtime audio fixtures must be 16-bit PCM WAV files")
        rate = handle.getframerate()
        return handle.readframes(handle.getnframes()), rate


def wav_from_pcm16(audio: bytes, *, sample_rate: int) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio)
    return buffer.getvalue()


def redact_realtime_events(events: list[Any]) -> list[Any]:
    return [redact_realtime_payload(raw_payload(event)) for event in events]


def parse_realtime_tool_calls(events: list[Any], response: Any | None) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}

    def upsert(payload: Any, *, output_index: Any = None, source: str | None = None) -> None:
        item = raw_payload(payload)
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type not in {"function_call", "mcp_tool_call"}:
            return
        key = str(item.get("call_id") or item.get("id") or f"tool_call_{len(calls)}")
        current = calls.setdefault(key, {"type": item_type})
        for field_name in ("id", "call_id", "name", "arguments", "status"):
            value = item.get(field_name)
            if value not in (None, ""):
                current[field_name] = value
        if output_index is not None:
            current["output_index"] = output_index
        if source:
            current["source"] = source
        arguments = current.get("arguments")
        if isinstance(arguments, str):
            try:
                current["arguments_json"] = json_loads(arguments)
            except ValueError:
                pass

    for event in events:
        payload = raw_payload(event)
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            upsert(
                payload.get("item"),
                output_index=payload.get("output_index"),
                source=str(event_type),
            )
        elif event_type == "response.function_call_arguments.done":
            upsert(
                {
                    "type": "function_call",
                    "id": payload.get("item_id"),
                    "call_id": payload.get("call_id"),
                    "name": payload.get("name"),
                    "arguments": payload.get("arguments"),
                },
                output_index=payload.get("output_index"),
                source=str(event_type),
            )

    response_payload = raw_payload(response)
    if isinstance(response_payload, dict):
        for index, item in enumerate(response_payload.get("output") or []):
            upsert(item, output_index=index, source="response.done")

    return list(calls.values())


def redact_realtime_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [redact_realtime_payload(item) for item in value]
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key in {"audio", "delta"} and isinstance(item, str) and looks_like_base64_audio(item):
                redacted[key] = base64_marker(item)
            else:
                redacted[key] = redact_realtime_payload(item)
        return redacted
    return value


def looks_like_base64_audio(value: str) -> bool:
    compact = "".join(value.split())
    return len(compact) > 128 and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact) is not None


def base64_marker(value: str) -> str:
    compact = "".join(value.split())
    digest = hashlib.sha256(compact.encode("ascii", errors="ignore")).hexdigest()
    return f"<redacted base64 media sha256={digest} base64_chars={len(compact)}>"


def json_loads(value: str) -> Any:
    import json

    try:
        return json.loads(value)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


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
