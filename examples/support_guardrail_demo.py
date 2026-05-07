from __future__ import annotations

from openai import AsyncOpenAI

from prism_evals import Experiment

from support_demo_common import (
    BASE_DIR,
    GUARDRAIL_PROMPT_PATH,
    classify_support_request,
    route_accuracy,
    selected_models,
)


exp = Experiment(
    name="support_guardrail_demo",
    dataset="datasets/support_agent_demo.csv",
    output_dir="runs",
    concurrency=4,
    resume=True,
    repetitions=1,
    artifacts=[GUARDRAIL_PROMPT_PATH.relative_to(BASE_DIR)],
    metadata={"demo_stage": "guardrail_only"},
    openai_client=AsyncOpenAI(),
)
exp.models(selected_models())


async def guardrail_only(item, model, ctx):
    return await classify_support_request(item, model, ctx)


exp.workflow = guardrail_only
exp.eval("route_accuracy", route_accuracy, description="Guardrail route matches expected route")
