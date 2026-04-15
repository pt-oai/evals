from __future__ import annotations

import pytest

from evals import ModelConfig
from evals.openai import ExperimentContext, extract_usage


@pytest.mark.parametrize("capture_raw", [True, False])
def test_responses_wrapper_captures_generation(fake_client, capture_raw):
    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            row={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            execution_id="exec",
            capture_raw=capture_raw,
            max_retries=1,
        )
        response = await ctx.responses.create(model="gpt-test", input="hello")
        return ctx, response

    import asyncio

    ctx, response = asyncio.run(run())
    assert response.output_text == "answer: hello"
    assert len(ctx.generations) == 1
    generation = ctx.generations[0]
    assert generation.response_id == "resp_test"
    assert generation.output_text == "answer: hello"
    assert generation.usage.input_tokens == 3
    assert generation.usage.cached_tokens == 1
    assert generation.usage.reasoning_tokens == 2
    assert generation.raw_request is not None if capture_raw else generation.raw_request is None


def test_extract_usage_supports_dict_response():
    response = {
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 16,
        }
    }
    usage = extract_usage(response)
    assert usage.input_tokens == 10
    assert usage.cached_tokens == 4
    assert usage.output_tokens == 6
    assert usage.reasoning_tokens == 2
    assert usage.total_tokens == 16

