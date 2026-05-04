# Migration Guide

## 0.7.0: `TaskOutput` Is Required

Prism workflows and step callables must now return `TaskOutput`. Strings and
dicts are no longer normalized as workflow outputs.

Prism also no longer owns OpenAI SDK calls for normal workflows. Import and use
the OpenAI SDK directly in your experiment file:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()
```

Use that client for Responses API, Image API, or any other OpenAI call, then
return a `TaskOutput` with the data Prism should evaluate and store.

Text output:

```python
from openai import AsyncOpenAI
from prism_evals import TaskOutput

client = AsyncOpenAI()

response = await client.responses.create(...)

# Old
return response.output_text

# New
return TaskOutput(text=response.output_text)
```

Structured output:

```python
# Old
return {"answer": parsed}

# New
return TaskOutput(text=json.dumps(parsed), value=parsed)
```

Step output:

```python
# Old
draft = await ctx.step("draft", lambda: "hello")

# New
draft = await ctx.step("draft", lambda: TaskOutput(text="hello"))
```

Generated media:

```python
from openai import AsyncOpenAI
from prism_evals import TaskOutput

client = AsyncOpenAI()

async def workflow(item, model, ctx):
    response = await client.images.generate(
        model=model.model,
        prompt=item["prompt"],
        response_format="b64_json",
        **model.params,
    )
    image = ctx.media.from_base64(response.data[0].b64_json, format="png")
    return TaskOutput(text="Generated image", media=[image])
```

Generated files now live in `media/`. The `artifacts/` directory remains for
files copied from the experiment source tree with `Experiment(...,
artifacts=[...])`.

Custom JSONL consumers should read generated media from `output.media` or
`steps[].output.media`. Prism no longer expects raw provider responses to be the
source of truth for generated images.

Direct SDK calls are not automatically recorded as Prism generation records, and
raw OpenAI request/response payloads are not captured by default. Store any
important call details in `TaskOutput.metadata` when your eval needs them.
