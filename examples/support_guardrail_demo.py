from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from prism_evals import EvalResult, Experiment, ModelConfig, TaskOutput


BASE_DIR = Path(__file__).parent
GUARDRAIL_PROMPT_PATH = BASE_DIR / "prompts" / "support_guardrail.md"
GUARDRAIL_PROMPT = GUARDRAIL_PROMPT_PATH.read_text(encoding="utf-8").strip()
ROUTES = ["order_status", "returns", "account_help", "product_advice", "blocked"]


def guardrail_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "support_guardrail_route",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ROUTES},
                "allowed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["route", "allowed", "reason"],
            "additionalProperties": False,
        },
    }


def response_params(model: ModelConfig) -> dict[str, Any]:
    params = copy.deepcopy(model.params)
    text_params = dict(params.pop("text", {}))
    text_params["format"] = guardrail_schema()
    return {**params, "text": text_params}


def parse_json_output(output_text: str) -> dict[str, Any]:
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return parsed


def selected_models() -> list[ModelConfig]:
    models = [
        ModelConfig(
            key="gpt55_low",
            model="gpt-5.5",
            params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        ),
        ModelConfig(
            key="gpt55_medium",
            model="gpt-5.5",
            params={"reasoning": {"effort": "medium"}, "text": {"verbosity": "low"}},
        ),
        ModelConfig(
            key="gpt54_mini_low",
            model="gpt-5.4-mini",
            params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        ),
    ]
    requested = os.getenv("PRISM_SUPPORT_MODEL_KEYS")
    if not requested:
        return models
    keys = {key.strip() for key in requested.split(",") if key.strip()}
    selected = [model for model in models if model.key in keys]
    if not selected:
        available = ", ".join(model.key for model in models)
        raise ValueError(f"PRISM_SUPPORT_MODEL_KEYS did not match any models. Available: {available}")
    return selected


exp = Experiment(
    name="support_guardrail_demo",
    dataset="datasets/support_agent_demo.csv",
    output_dir="runs",
    concurrency=25,
    resume=True,
    repetitions=1,
    artifacts=[GUARDRAIL_PROMPT_PATH.relative_to(BASE_DIR)],
    metadata={"demo_stage": "guardrail_only"},
    openai_client=AsyncOpenAI(),
)
exp.models(selected_models())


async def guardrail_only(item, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **response_params(model),
        input=(
            f"{GUARDRAIL_PROMPT}\n\n"
            f"Customer message:\n{item['request']}\n\n"
            "Return the route decision as JSON."
        ),
    )
    value = parse_json_output(response.output_text)
    return TaskOutput(text=response.output_text, value=value)


def route_accuracy(item, model, output, ctx):
    actual = output.value.get("route") if isinstance(output.value, dict) else None
    expected = item["expected_route"]
    score = actual == expected
    return EvalResult(
        score=score,
        comment=None if score else f"expected {expected}, got {actual}",
        metadata={"expected_route": expected, "actual_route": actual},
    )


def guardrail_returns_json(item, model, output, ctx):
    try:
        value = parse_json_output(output.text)
    except Exception as exc:
        return EvalResult(score=False, comment=f"response was not valid JSON: {exc}")

    has_route = value.get("route") in ROUTES
    has_allowed = isinstance(value.get("allowed"), bool)
    has_reason = isinstance(value.get("reason"), str) and bool(value["reason"].strip())
    score = has_route and has_allowed and has_reason
    comment = None if score else "JSON must include route, allowed, and reason"
    return EvalResult(
        score=score,
        comment=comment,
        metadata={"has_route": has_route, "has_allowed": has_allowed, "has_reason": has_reason},
    )


exp.workflow = guardrail_only
exp.eval(
    "guardrail_returns_json",
    guardrail_returns_json,
    description="Guardrail response is valid JSON with route, allowed, and reason",
)
exp.eval("route_accuracy", route_accuracy, description="Guardrail route matches expected route")
