from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, TaskOutput, item, text


exp = Experiment(
    name="qa_smoke",
    dataset="datasets/qa.csv",
    output_dir="runs",
    concurrency=5,
    resume=True,
    repetitions=1,
)

exp.model(
    ModelConfig(
        key="gpt5_low",
        model="gpt-5",
        params={"reasoning": {"effort": "low"}},
    )
)

client = AsyncOpenAI()


async def answer(item, model, ctx):
    response = await client.responses.create(
        model=model.model,
        **model.params,
        input=item["question"],
    )
    return TaskOutput(text=response.output_text)


exp.workflow = answer
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected answer appears",
)
exp.eval("brevity", LengthBetween(value=text(), max_len=200))
