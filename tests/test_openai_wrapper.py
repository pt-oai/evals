from __future__ import annotations

import json
import base64
from types import SimpleNamespace

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


def test_extract_usage_supports_realtime_response_shape():
    response = {
        "usage": {
            "input_tokens": 10,
            "input_token_details": {"cached_tokens": 4, "audio_tokens": 3},
            "output_tokens": 6,
            "output_token_details": {"audio_tokens": 2},
            "total_tokens": 16,
        }
    }
    usage = extract_usage(response)
    assert usage.input_tokens == 10
    assert usage.cached_tokens == 4
    assert usage.output_tokens == 6
    assert usage.total_tokens == 16


def test_realtime_run_text_captures_generation(tmp_path):
    async def run():
        events = [
            {"type": "session.updated"},
            {"type": "conversation.item.done"},
            {"type": "response.output_text.delta", "delta": "pr"},
            {"type": "response.output_text.delta", "delta": "ism"},
            {
                "type": "response.done",
                "response": {
                    "id": "resp_rt",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 7,
                        "input_token_details": {"cached_tokens": 2},
                        "output_tokens": 3,
                        "total_tokens": 10,
                    },
                },
            },
        ]
        client = FakeRealtimeClient(events)
        ctx = ExperimentContext(
            client=client,
            item={"prompt": "hello"},
            model=ModelConfig(key="rt", model="gpt-realtime-2", params={"reasoning": {"effort": "low"}}),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
        )
        result = await ctx.realtime.run_text("Reply with prism", instructions="Be brief")
        return ctx, client, result

    import asyncio

    ctx, client, result = asyncio.run(run())
    assert result.text == "prism"
    assert result.response_id == "resp_rt"
    assert result.task_output().text == "prism"
    assert result.task_output().value["tool_call_count"] == 0
    assert client.calls[0] == ("connect", {"model": "gpt-realtime-2"})
    assert client.calls[1][0] == "session.update"
    assert client.calls[2][0] == "conversation.item.create"
    assert client.calls[3][0] == "response.create"
    assert ctx.generations[0].response_id == "resp_rt"
    assert ctx.generations[0].usage.cached_tokens == 2
    assert ctx.generations[0].metadata["api"] == "realtime"
    assert ctx.generations[0].raw_request[0]["session"]["reasoning"] == {"effort": "low"}


def test_realtime_run_text_exposes_tool_calls_for_scoring(tmp_path):
    async def run():
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_order",
                    "arguments": '{"order_id":"A19"}',
                    "status": "completed",
                },
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_1",
                "call_id": "call_1",
                "name": "lookup_order",
                "arguments": '{"order_id":"A19"}',
            },
            {
                "type": "response.done",
                "response": {
                    "id": "resp_tool",
                    "status": "completed",
                    "output": [
                        {
                            "id": "fc_1",
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "lookup_order",
                            "arguments": '{"order_id":"A19"}',
                            "status": "completed",
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            },
        ]
        client = FakeRealtimeClient(events)
        ctx = ExperimentContext(
            client=client,
            item={"prompt": "hello"},
            model=ModelConfig(key="rt", model="gpt-realtime-2"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=False,
            max_retries=1,
        )
        result = await ctx.realtime.run_text(
            "Look up order A19",
            response={
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup_order",
                        "description": "Look up an order",
                        "parameters": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        },
                    }
                ]
            },
        )
        return ctx, result

    import asyncio

    ctx, result = asyncio.run(run())
    assert result.tool_calls == [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup_order",
            "arguments": '{"order_id":"A19"}',
            "status": "completed",
            "output_index": 0,
            "source": "response.done",
            "arguments_json": {"order_id": "A19"},
        }
    ]
    output = result.task_output()
    assert output.value["tool_call_count"] == 1
    assert output.value["tool_calls"][0]["name"] == "lookup_order"
    assert output.value["tool_calls"][0]["arguments_json"] == {"order_id": "A19"}
    assert ctx.generations[0].metadata["tool_call_count"] == 1
    assert ctx.generations[0].metadata["tool_call_names"] == ["lookup_order"]


def test_realtime_run_audio_writes_wav_media_and_redacts_audio_events(tmp_path):
    output_pcm = b"\x00\x01" * 128
    encoded_output = base64.b64encode(output_pcm).decode("ascii")

    async def run():
        events = [
            {"type": "session.updated"},
            {"type": "input_audio_buffer.committed"},
            {"type": "response.output_audio_transcript.delta", "delta": "prism"},
            {"type": "response.output_audio.delta", "delta": encoded_output},
            {
                "type": "response.done",
                "response": {
                    "id": "resp_audio",
                    "status": "completed",
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                },
            },
        ]
        client = FakeRealtimeClient(events)
        ctx = ExperimentContext(
            client=client,
            item={"prompt": "hello"},
            model=ModelConfig(key="rt", model="gpt-realtime-2"),
            item_run_id="item-run",
            run_dir=tmp_path,
            capture_raw=True,
            max_retries=1,
        )
        result = await ctx.realtime.run_audio(b"\x00\x00\x01\x00", output_name="reply")
        return ctx, client, result

    import asyncio

    ctx, client, result = asyncio.run(run())
    assert result.transcript == "prism"
    assert result.audio == output_pcm
    assert result.media[0].path == "media/item-run-001-reply.wav"
    assert result.media[0].mime_type == "audio/wav"
    assert (tmp_path / result.media[0].path).read_bytes().startswith(b"RIFF")
    assert ("input_audio_buffer.append", {"audio": base64.b64encode(b"\x00\x00\x01\x00").decode("ascii")}) in client.calls
    assert ("input_audio_buffer.commit", {}) in client.calls
    raw_response_json = json.dumps(ctx.generations[0].raw_response)
    assert encoded_output not in raw_response_json
    assert "<redacted base64 media sha256=" in raw_response_json


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


def test_media_store_writes_audio_mime_type(tmp_path, fake_client):
    ctx = ExperimentContext(
        client=fake_client,
        item={"question": "hello"},
        model=ModelConfig(key="m1", model="gpt-test"),
        item_run_id="item-run",
        run_dir=tmp_path,
        capture_raw=True,
        max_retries=1,
    )

    artifact = ctx.media.from_bytes(b"wav bytes", format="wav")

    assert artifact.path == "media/item-run-001.wav"
    assert artifact.mime_type == "audio/wav"


class FakeRealtimeClient:
    def __init__(self, events):
        self.calls = []
        self.realtime = SimpleNamespace(connect=lambda **kwargs: FakeRealtimeManager(self.calls, kwargs, events))


class FakeRealtimeManager:
    def __init__(self, calls, connect_kwargs, events):
        self.calls = calls
        self.connect_kwargs = connect_kwargs
        self.events = events

    async def __aenter__(self):
        self.calls.append(("connect", self.connect_kwargs))
        return FakeRealtimeConnection(self.calls, self.events)

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeRealtimeConnection:
    def __init__(self, calls, events):
        self.calls = calls
        self.events = list(events)
        self.session = SimpleNamespace(update=self._session_update)
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=self._conversation_item_create))
        self.input_audio_buffer = SimpleNamespace(append=self._input_audio_append, commit=self._input_audio_commit)
        self.response = SimpleNamespace(create=self._response_create)

    async def _session_update(self, **kwargs):
        self.calls.append(("session.update", kwargs))

    async def _conversation_item_create(self, **kwargs):
        self.calls.append(("conversation.item.create", kwargs))

    async def _input_audio_append(self, **kwargs):
        self.calls.append(("input_audio_buffer.append", kwargs))

    async def _input_audio_commit(self, **kwargs):
        self.calls.append(("input_audio_buffer.commit", kwargs))

    async def _response_create(self, **kwargs):
        self.calls.append(("response.create", kwargs))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)
