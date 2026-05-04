from __future__ import annotations

import json
import base64

import pytest

from prism_evals import ModelConfig, TaskOutput
from prism_evals.openai import ExperimentContext, extract_usage


@pytest.mark.parametrize("capture_raw", [True, False])
def test_responses_wrapper_captures_generation(tmp_path, fake_client, capture_raw):
    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            item={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            item_run_id="item-run",
            run_dir=tmp_path,
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


def test_step_records_only_generations_inside_step(tmp_path, fake_client):
    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            item={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
        )
        await ctx.responses.create(model="gpt-test", input="outside")

        async def inside_step():
            response = await ctx.responses.create(model="gpt-test", input="inside")
            return TaskOutput(text=response.output_text)

        output = await ctx.step("inside", inside_step)
        return ctx, output

    import asyncio

    ctx, output = asyncio.run(run())
    assert output.text == "answer: inside"
    assert len(ctx.generations) == 2
    assert len(ctx.steps) == 1
    assert ctx.steps[0].key == "inside"
    assert len(ctx.steps[0].generations) == 1
    assert ctx.steps[0].generations[0].output_text == "answer: inside"


def test_raw_request_redacts_data_urls_by_default(tmp_path, fake_client):
    image_url = "data:image/jpeg;base64," + ("a" * 1024)

    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            item={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
        )
        await ctx.responses.create(
            model="gpt-test",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )
        return ctx

    import asyncio

    ctx = asyncio.run(run())
    raw_image_url = ctx.generations[0].raw_request["input"][0]["content"][1]["image_url"]
    assert raw_image_url.startswith("data:image/jpeg;base64,<redacted sha256=")
    assert "chars=1047" in raw_image_url
    assert "base64_chars=1024" in raw_image_url
    assert image_url not in json.dumps(ctx.generations[0].raw_request)
    assert fake_client.calls[0]["input"][0]["content"][1]["image_url"] == image_url


def test_raw_request_data_url_redaction_can_be_disabled(tmp_path, fake_client):
    image_url = "data:image/jpeg;base64," + ("a" * 1024)

    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            item={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
            redact_raw_data_urls=False,
        )
        await ctx.responses.create(
            model="gpt-test",
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": image_url}]}],
        )
        return ctx

    import asyncio

    ctx = asyncio.run(run())
    assert ctx.generations[0].raw_request["input"][0]["content"][0]["image_url"] == image_url


def test_raw_response_redacts_bare_image_base64(tmp_path, fake_client):
    payload = "a" * 1024

    async def run():
        ctx = ExperimentContext(
            client=fake_client,
            item={"question": "hello"},
            model=ModelConfig(key="m1", model="gpt-test"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
        )
        await ctx.responses.create(model="gpt-test", input="hello", extra_body={"b64_json": payload})
        return ctx

    import asyncio

    ctx = asyncio.run(run())
    raw_request = ctx.generations[0].raw_request
    assert raw_request["extra_body"]["b64_json"].startswith("<redacted base64 media sha256=")
    assert payload not in json.dumps(raw_request)


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


def test_media_store_writes_base64_bytes_and_path(tmp_path, fake_client):
    ctx = ExperimentContext(
        client=fake_client,
        item={"question": "hello"},
        model=ModelConfig(key="m1", model="gpt-test"),
        item_run_id="item-run",
        run_dir=tmp_path,
        capture_raw=True,
        max_retries=1,
    )

    artifact = ctx.media.from_base64(
        "data:image/png;base64," + base64.b64encode(b"png bytes").decode("ascii"),
        name="preview image",
        metadata={"alt": "Preview"},
    )

    assert artifact.path == "media/item-run-001-preview-image.png"
    assert artifact.mime_type == "image/png"
    assert artifact.format == "png"
    assert artifact.bytes == len(b"png bytes")
    assert artifact.metadata == {"alt": "Preview"}
    assert (tmp_path / artifact.path).read_bytes() == b"png bytes"


def test_media_store_copies_source_path(tmp_path, fake_client):
    source = tmp_path / "source.webp"
    source.write_bytes(b"webp bytes")
    ctx = ExperimentContext(
        client=fake_client,
        item={"question": "hello"},
        model=ModelConfig(key="m1", model="gpt-test"),
        item_run_id="item-run",
        run_dir=tmp_path,
        capture_raw=True,
        max_retries=1,
    )

    artifact = ctx.media.from_path(source)

    assert artifact.path == "media/item-run-001-source.webp"
    assert artifact.mime_type == "image/webp"
    assert (tmp_path / artifact.path).read_bytes() == b"webp bytes"


def test_media_store_writes_raw_bytes(tmp_path, fake_client):
    ctx = ExperimentContext(
        client=fake_client,
        item={"question": "hello"},
        model=ModelConfig(key="m1", model="gpt-test"),
        item_run_id="item-run",
        run_dir=tmp_path,
        capture_raw=True,
        max_retries=1,
    )

    artifact = ctx.media.from_bytes(b"jpeg bytes", format="jpg")

    assert artifact.path == "media/item-run-001.jpeg"
    assert artifact.mime_type == "image/jpeg"
    assert (tmp_path / artifact.path).read_bytes() == b"jpeg bytes"
