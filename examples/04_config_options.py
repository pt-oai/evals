from pathlib import Path

from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, NonEmpty, TaskOutput, item, text


BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "config_system.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


exp = Experiment(
    name="config_options",
    dataset="datasets/config_options.csv",
    output_dir="runs",
    concurrency=2,
    resume=True,
    repetitions=2,
    max_retries=2,
    fail_fast=False,
    timestamp_output_dir=False,
    artifacts=[SYSTEM_PROMPT_PATH.relative_to(BASE_DIR)],
    display="progress",
    metadata={"example": "configuration_options"},
    openai_client=AsyncOpenAI(),
)

exp.variant(
    "fast_writer",
    models={
        "planner": {
            "model": "gpt-5.4-mini",
            "params": {"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        },
        "writer": {
            "model": "gpt-5",
            "params": {"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        },
    },
    default_role="writer",
    metadata={"goal": "low-latency baseline"},
)

exp.variant(
    "careful_writer",
    models={
        "planner": {
            "model": "gpt-5",
            "params": {"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        },
        "writer": {
            "model": "gpt-5",
            "params": {"reasoning": {"effort": "medium"}, "text": {"verbosity": "low"}},
        },
    },
    default_role="writer",
    metadata={"goal": "more reasoning on the final answer"},
)


async def summarize(item, model, ctx):
    writer = ctx.model("writer")
    response = await ctx.responses.create(
        model=writer.model,
        **writer.params,
        input=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Topic: {item['topic']}\n"
            f"Required phrase: {item['required_phrase']}"
        ),
    )
    return TaskOutput(
        text=response.output_text,
        metadata={"variant": model.key, "writer_model": writer.model},
    )


exp.workflow = summarize
exp.eval("non_empty", NonEmpty(text()))
exp.eval(
    "contains_required_phrase",
    Contains(container=text(), expected=item("required_phrase"), case_sensitive=False),
)
