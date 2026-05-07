from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from prism_evals import EvalResult, ModelConfig, TaskOutput


BASE_DIR = Path(__file__).parent
GUARDRAIL_PROMPT_PATH = BASE_DIR / "prompts" / "support_guardrail.md"
WORKLOAD_PROMPT_PATH = BASE_DIR / "prompts" / "support_workload.md"
GUARDRAIL_PROMPT = GUARDRAIL_PROMPT_PATH.read_text(encoding="utf-8").strip()
WORKLOAD_PROMPT = WORKLOAD_PROMPT_PATH.read_text(encoding="utf-8").strip()

ROUTES = ["order_status", "returns", "account_help", "product_advice", "blocked"]
ACTIONS = ["lookup_order", "start_return", "reset_account", "recommend_product", "refuse"]


def json_schema(name: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


GUARDRAIL_SCHEMA = json_schema(
    "support_guardrail_route",
    {
        "route": {"type": "string", "enum": ROUTES},
        "allowed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    ["route", "allowed", "reason"],
)

WORKLOAD_SCHEMA = json_schema(
    "support_workload_reply",
    {
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": ACTIONS},
    },
    ["reply", "action"],
)


def response_params(model: ModelConfig, schema: dict[str, Any]) -> dict[str, Any]:
    params = copy.deepcopy(model.params)
    text_params = dict(params.pop("text", {}))
    text_params["format"] = schema
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


async def classify_support_request(item, model, ctx) -> TaskOutput:
    response = await ctx.responses.create(
        model=model.model,
        **response_params(model, GUARDRAIL_SCHEMA),
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
