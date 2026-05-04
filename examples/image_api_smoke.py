from openai import AsyncOpenAI

from prism_evals import Experiment, ModelConfig, NonEmpty, TaskOutput, text


exp = Experiment(
    name="image_api_smoke",
    dataset="datasets/image_prompts.csv",
    output_dir="runs",
    concurrency=1,
)

exp.model(ModelConfig(key="gpt_image_low", model="gpt-image-2", params={"quality": "low"}))

client = AsyncOpenAI()


async def generate_image(item, model, ctx):
    response = await client.images.generate(
        model=model.model,
        prompt=item["prompt"],
        response_format="b64_json",
        **model.params,
    )
    image = ctx.media.from_base64(response.data[0].b64_json, format="png", name=item["id"])
    return TaskOutput(text="Generated image", media=[image])


exp.workflow = generate_image
exp.eval("has_summary", NonEmpty(text()))
