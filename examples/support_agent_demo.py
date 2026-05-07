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


exp = Experiment(
    name="support_agent_demo",
    dataset="datasets/support_agent_demo.csv",
    output_dir="runs",
    concurrency=25,
    resume=True,
    repetitions=1,
    artifacts=[GUARDRAIL_PROMPT_PATH.relative_to(BASE_DIR), WORKLOAD_PROMPT_PATH.relative_to(BASE_DIR)],
    metadata={"demo_stage": "full_agent"},
    openai_client=AsyncOpenAI(),
)
exp.models(selected_models())


async def support_agent(item, model, ctx):
    async def classify_support_request():
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

    guardrail = await ctx.step(
        "guardrail_triage",
        classify_support_request,
        evals=[("route_accuracy", route_accuracy, "Guardrail route matches expected route")],
    )

    async def write_customer_response():
        response = await ctx.responses.create(
            model=model.model,
            **response_params(model, WORKLOAD_SCHEMA),
            input=(
                f"{WORKLOAD_PROMPT}\n\n"
                f"Customer message:\n{item['request']}\n\n"
                f"Guardrail route JSON:\n{json.dumps(guardrail.value, sort_keys=True)}\n\n"
                "Return the final reply as JSON."
            ),
        )
        value = parse_json_output(response.output_text)
        return TaskOutput(text=value["reply"], value=value)

    return await ctx.step("customer_response", write_customer_response)


def route_accuracy(item, model, output, ctx):
    actual = output.value.get("route") if isinstance(output.value, dict) else None
    expected = item["expected_route"]
    score = actual == expected
    return EvalResult(
        score=score,
        comment=None if score else f"expected {expected}, got {actual}",
        metadata={"expected_route": expected, "actual_route": actual},
    )


def end_to_end_success(item, model, output, ctx):
    guardrail = ctx.step_outputs.get("guardrail_triage")
    route = guardrail.value.get("route") if guardrail and isinstance(guardrail.value, dict) else None
    action = output.value.get("action") if isinstance(output.value, dict) else None
    reply = output.value.get("reply", "") if isinstance(output.value, dict) else ""
    expected_phrase = item["expected_reply_phrase"]

    route_ok = route == item["expected_route"]
    action_ok = action == item["expected_action"]
    phrase_ok = expected_phrase.lower() in reply.lower()
    score = route_ok and action_ok and phrase_ok
    comment = None
    if not score:
        misses = []
        if not route_ok:
            misses.append(f"route expected {item['expected_route']}, got {route}")
        if not action_ok:
            misses.append(f"action expected {item['expected_action']}, got {action}")
        if not phrase_ok:
            misses.append(f"reply missing {expected_phrase!r}")
        comment = "; ".join(misses)
    return EvalResult(
        score=score,
        comment=comment,
        metadata={
            "route_ok": route_ok,
            "action_ok": action_ok,
            "phrase_ok": phrase_ok,
            "expected_phrase": expected_phrase,
        },
    )


exp.workflow = support_agent
exp.eval(
    "end_to_end_success",
    end_to_end_success,
    description="Full chain routes correctly and returns the expected support action",
)
