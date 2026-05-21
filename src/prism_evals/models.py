from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScoreValue = bool | int | float
ScoreType = Literal["BOOLEAN", "NUMERIC"]
Status = Literal["success", "failed", "skipped"]
EvalFn = Callable[..., Any]


@dataclass(frozen=True)
class EvalDefinition:
    key: str
    func: EvalFn
    description: str | None = None


class ModelConfig(BaseModel):
    key: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def key_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model key must not be empty")
        return value


class ModelVariant(BaseModel):
    key: str
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    default_role: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def key_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("variant key must not be empty")
        return value

    @property
    def default_model(self) -> ModelConfig:
        if self.default_role in self.models:
            return self.models[self.default_role]
        if self.models:
            return next(iter(self.models.values()))
        raise ValueError(f"variant {self.key!r} has no models")

    @property
    def model(self) -> str:
        return self.default_model.model

    @property
    def params(self) -> dict[str, Any]:
        return self.default_model.params

    def model_for(self, role: str = "default") -> ModelConfig:
        if role in self.models:
            return self.models[role]
        if role == "default":
            return self.default_model
        raise KeyError(f"variant {self.key!r} has no model role {role!r}")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ErrorRecord(BaseModel):
    type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MediaArtifact(BaseModel):
    path: str
    mime_type: str
    format: str
    sha256: str
    bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskOutput(BaseModel):
    text: str = ""
    value: Any | None = None
    media: list[MediaArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def normalize(cls, value: Any) -> "TaskOutput":
        if isinstance(value, TaskOutput):
            return value
        raise TypeError(task_output_error_message("workflow or step", value))


class EvalResult(BaseModel):
    key: str | None = None
    score: ScoreValue | None = None
    data_type: ScoreType | None = None
    description: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: ErrorRecord | None = None

    def with_defaults(self, key: str, description: str | None = None) -> "EvalResult":
        data = self.model_dump()
        data["key"] = data["key"] or key
        data["description"] = data["description"] or description
        if data["data_type"] is None and data["score"] is not None:
            data["data_type"] = infer_score_type(data["score"])
        return EvalResult(**data)


class GenerationRecord(BaseModel):
    response_id: str | None = None
    latency_s: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_request: Any | None = None
    raw_response: Any | None = None
    output_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: ErrorRecord | None = None


class ToolCallRecord(BaseModel):
    name: str
    arguments: Any | None = None
    result: Any | None = None
    agent: str | None = None
    turn_id: str | None = None
    call_id: str | None = None
    status: Status = "success"
    duration_s: float = 0.0
    started_at: str | None = None
    ended_at: str | None = None
    error: ErrorRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    id: str
    role: str
    mode: str | None = None
    status: Status = "success"
    started_at: str
    ended_at: str
    duration_s: float = 0.0
    input: Any | None = None
    output: TaskOutput | None = None
    evals: list[EvalResult] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: ErrorRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepRecord(BaseModel):
    key: str
    status: Status
    started_at: str
    ended_at: str
    duration_s: float
    output: TaskOutput | None = None
    evals: list[EvalResult] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    response_id: str | None = None
    generations: list[GenerationRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: ErrorRecord | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ItemRunRecord(BaseModel):
    item_run_id: str
    run_id: str
    experiment_name: str
    item_index: int
    item_id: str
    item: dict[str, Any]
    model_key: str
    model: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    variant_key: str | None = None
    repetition: int
    status: Status
    started_at: str
    ended_at: str
    duration_s: float
    output: TaskOutput | None = None
    evals: list[EvalResult] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    response_id: str | None = None
    generations: list[GenerationRecord] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    turns: list[TurnRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    raw_input: Any | None = None
    raw_output: Any | None = None
    error: ErrorRecord | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    experiment_name: str
    started_at: str
    ended_at: str | None = None
    dataset_path: str
    dataset_sha256: str | None = None
    experiment_file: str | None = None
    experiment_sha256: str | None = None
    output_dir: str
    settings: dict[str, Any] = Field(default_factory=dict)
    model_configs: list[ModelConfig] = Field(default_factory=list)
    variant_configs: list[ModelVariant] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    python_version: str | None = None


def infer_score_type(score: ScoreValue) -> ScoreType:
    if isinstance(score, bool):
        return "BOOLEAN"
    return "NUMERIC"


def require_task_output(value: Any, *, context: str) -> TaskOutput:
    if isinstance(value, TaskOutput):
        return value
    raise TypeError(task_output_error_message(context, value))


def task_output_error_message(context: str, value: Any) -> str:
    return (
        f"{context} must return prism_evals.TaskOutput, got {type(value).__name__}. "
        "Prism now requires explicit outputs; use TaskOutput(text=...), "
        "TaskOutput(value=...), or TaskOutput(media=[...])."
    )
