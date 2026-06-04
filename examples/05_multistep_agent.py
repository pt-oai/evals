from __future__ import annotations

import copy
import json
from typing import Any

from openai import AsyncOpenAI

from prism_evals import Equal, EvalResult, Experiment, ModelConfig, NonEmpty, TaskOutput, item, out, text


ROUTES = ["billing", "shipping", "technical", "product"]

ROUTE_SCHEMA = {
    "type": "json_schema",
    "name": "ticket_route",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ROUTES},
            "reason": {"type": "string"},
        },
        "required": ["route", "reason"],
        "additionalProperties": False,
    },
}

REPLY_SCHEMA = {
    "type": "json_schema",
    "name": "ticket_reply",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "next_step": {"type": "string"},
        },
        "required": ["reply", "next_step"],
        "additionalProperties": False,
    },
}


exp = Experiment(
    name="multistep_agent",
    dataset="datasets/support_tickets.csv",
    output_dir="runs",
    concurrency=5,
    resume=True,
    openai_client=AsyncOpenAI(),
)

exp.model(
    ModelConfig(
        key="gpt5_low",
        model="gpt-5",
        params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
    )
)


def with_schema(model: ModelConfig, schema: dict[str, Any]) -> dict[str, Any]:
    params = copy.deepcopy(model.params)
    text_params = dict(params.pop("text", {}))
    text_params["format"] = schema
    return {**params, "text": text_params}


def parse_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


async def route_ticket(item, model, ctx):
    async def classify():
        response = await ctx.responses.create(
            model=model.model,
            **with_schema(model, ROUTE_SCHEMA),
            input=(
                "Route this customer ticket to exactly one queue: "
                "billing, shipping, technical, or product.\n\n"
                f"Ticket: {item['message']}"
            ),
        )
        value = parse_object(response.output_text)
        return TaskOutput(text=response.output_text, value=value)

    route = await ctx.step(
        "route",
        classify,
        evals=[
            ("route_matches", Equal(actual=out("route"), expected=item("expected_route"))),
        ],
    )

    async def draft_reply():
        response = await ctx.responses.create(
            model=model.model,
            **with_schema(model, REPLY_SCHEMA),
            input=(
                "Write a short customer reply and name the next internal step.\n\n"
                f"Ticket: {item['message']}\n"
                f"Route JSON: {json.dumps(route.value, sort_keys=True)}"
            ),
        )
        value = parse_object(response.output_text)
        return TaskOutput(text=value["reply"], value=value, metadata={"raw_json": response.output_text})

    return await ctx.step("reply", draft_reply, evals=[("reply_non_empty", NonEmpty(text()))])


def next_step_mentions_expected(item, model, output, ctx):
    if not isinstance(output.value, dict):
        return EvalResult(score=False, comment="final output did not include structured value")
    next_step = str(output.value.get("next_step", ""))
    expected = item["expected_next_step"]
    score = expected.lower() in next_step.lower()
    return EvalResult(
        score=score,
        comment=None if score else f"expected next step to mention {expected!r}",
        metadata={"expected_next_step": expected, "actual_next_step": next_step},
    )


exp.workflow = route_ticket
exp.eval("next_step_mentions_expected", next_step_mentions_expected)
