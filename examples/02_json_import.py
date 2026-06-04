from __future__ import annotations

from openai import AsyncOpenAI

from prism_evals import Contains, Experiment, LengthBetween, ModelConfig, TaskOutput, item, text


exp = Experiment(
    name="json_import",
    dataset="datasets/json_cases",
    output_dir="runs",
    concurrency=3,
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


def catalog_text(products: list[dict[str, str]]) -> str:
    return "\n".join(
        f"- {product['name']}: {product['summary']}"
        for product in products
    )


async def answer_from_json(item, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **model.params,
        input=(
            "Answer the customer using only this catalog.\n\n"
            f"Catalog:\n{catalog_text(item['context']['products'])}\n\n"
            f"Customer question: {item['question']}"
        ),
    )
    return TaskOutput(
        text=response.output_text,
        value={"product_count": len(item["context"]["products"])},
    )


exp.workflow = answer_from_json
exp.eval(
    "contains_expected",
    Contains(container=text(), expected=item("expected.answer_contains"), case_sensitive=False),
)
exp.eval("brief_answer", LengthBetween(value=text(), max_len=240))
