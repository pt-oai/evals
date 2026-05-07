from __future__ import annotations

import json

from openai import AsyncOpenAI

from prism_evals import EvalResult, Experiment, TaskOutput

from support_demo_common import (
    BASE_DIR,
    GUARDRAIL_PROMPT_PATH,
    WORKLOAD_PROMPT,
    WORKLOAD_PROMPT_PATH,
    WORKLOAD_SCHEMA,
    classify_support_request,
    parse_json_output,
    response_params,
    route_accuracy,
    selected_models,
)


exp = Experiment(
    name="support_agent_demo",
    dataset="datasets/support_agent_demo.csv",
    output_dir="runs",
    concurrency=4,
    resume=True,
    repetitions=1,
    artifacts=[GUARDRAIL_PROMPT_PATH.relative_to(BASE_DIR), WORKLOAD_PROMPT_PATH.relative_to(BASE_DIR)],
    metadata={"demo_stage": "full_agent"},
    openai_client=AsyncOpenAI(),
)
exp.models(selected_models())


async def support_agent(item, model, ctx):
    guardrail = await ctx.step(
        "guardrail_triage",
        lambda: classify_support_request(item, model, ctx),
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
