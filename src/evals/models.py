from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evals._utils import to_jsonable


ScoreValue = bool | int | float
ScoreType = Literal["BOOLEAN", "NUMERIC"]
Status = Literal["success", "failed", "skipped"]


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


class TaskOutput(BaseModel):
    text: str = ""
    value: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def normalize(cls, value: Any) -> "TaskOutput":
        if isinstance(value, TaskOutput):
            return value
        if isinstance(value, str):
            return cls(text=value)
        if isinstance(value, dict):
            text = value.get("text", "")
            metadata = value.get("metadata", {})
            return cls(text=str(text), value=value.get("value", value), metadata=dict(metadata))
        text = getattr(value, "text", None)
        if isinstance(text, str):
            metadata = getattr(value, "metadata", {})
            return cls(text=text, value=to_jsonable(value), metadata=dict(metadata or {}))
        return cls(text=str(value), value=to_jsonable(value))


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
    error: ErrorRecord | None = None


class ExecutionRecord(BaseModel):
    execution_id: str
    run_id: str
    experiment_name: str
    row_index: int
    row_id: str
    row: dict[str, str]
    model_key: str
    model: str
    model_params: dict[str, Any] = Field(default_factory=dict)
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    python_version: str | None = None


def infer_score_type(score: ScoreValue) -> ScoreType:
    if isinstance(score, bool):
        return "BOOLEAN"
    return "NUMERIC"
