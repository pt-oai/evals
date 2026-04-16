from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, item, text


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


async def answer(item, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=item["question"],
    )
    return response.output_text


exp.workflow = answer
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected answer appears",
)
exp.eval("brevity", LengthBetween(value=text(), max_len=200))


if __name__ == "__main__":
    exp.run()
