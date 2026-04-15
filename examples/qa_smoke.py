from evals import EvalResult, Experiment, ModelConfig


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


@exp.task
async def answer(row, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=row["question"],
    )
    return response.output_text


@exp.eval("contains_expected", description="Expected answer appears")
def contains_expected(row, model, output, ctx):
    return row["expected"].lower() in output.text.lower()


@exp.eval("brevity")
def brevity(row, model, output, ctx):
    return EvalResult(score=min(1.0, 200 / max(len(output.text), 1)))


if __name__ == "__main__":
    exp.run()

