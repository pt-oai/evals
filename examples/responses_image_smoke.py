from typing import Any

from openai import AsyncOpenAI

from prism_evals import Experiment, ModelConfig, NonEmpty, TaskOutput, text


exp = Experiment(
    name="responses_image_smoke",
    dataset="datasets/image_prompts.csv",
    output_dir="runs",
    concurrency=1,
)

exp.model(ModelConfig(key="gpt5_image", model="gpt-5", params={"reasoning": {"effort": "low"}}))

client = AsyncOpenAI()


async def generate_image(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["prompt"],
        tools=[{"type": "image_generation"}],
    )
    image = ctx.media.from_base64(first_image_result(response), format="png", name=item["id"])
    return TaskOutput(text=response.output_text or "Generated image", media=[image])


def first_image_result(response: Any) -> str:
    for output in getattr(response, "output", []) or []:
        output_type = getattr(output, "type", None)
        if output_type == "image_generation_call":
            result = getattr(output, "result", None)
            if isinstance(result, str):
                return result
    raise ValueError("response did not include an image_generation_call result")


exp.workflow = generate_image
exp.eval("has_summary", NonEmpty(text()))
