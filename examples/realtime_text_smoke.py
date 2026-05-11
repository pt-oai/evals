from prism_evals import Contains, Experiment, ModelConfig, NonEmpty, item, text


exp = Experiment(
    name="realtime_text_smoke",
    dataset="datasets/realtime_text_smoke.csv",
    output_dir="runs",
    concurrency=1,
    resume=True,
    repetitions=1,
)

exp.model(
    ModelConfig(
        key="realtime2_low",
        model="gpt-realtime-2",
        params={"reasoning": {"effort": "low"}},
    )
)


async def answer(item, model, ctx):
    result = await ctx.realtime.run_text(
        item["prompt"],
        instructions="Answer briefly and follow the user's requested format exactly.",
    )
    return result.task_output()


exp.workflow = answer
exp.eval("non_empty", NonEmpty(value=text()))
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected"), case_sensitive=False),
    description="Expected phrase appears in the Realtime text response",
)
