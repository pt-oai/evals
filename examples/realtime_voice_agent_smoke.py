from pathlib import Path

from prism_evals import Contains, EvalResult, Experiment, ModelConfig, NonEmpty, item, text


BASE_DIR = Path(__file__).parent

exp = Experiment(
    name="realtime_voice_agent_smoke",
    dataset="datasets/realtime_voice_smoke.csv",
    output_dir="runs",
    concurrency=1,
    resume=True,
    repetitions=1,
    artifacts=["audio/realtime_prism.wav"],
)

exp.model(
    ModelConfig(
        key="realtime2_low",
        model="gpt-realtime-2",
        params={"reasoning": {"effort": "low"}},
    )
)


async def answer(item, model, ctx):
    result = await ctx.realtime.run_audio(
        BASE_DIR / item["audio_path"],
        instructions="Reply briefly. If you hear the word prism, include the word prism in your response.",
        session={
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": None,
                },
                "output": {
                    "format": {"type": "audio/pcm"},
                    "voice": "marin",
                },
            }
        },
        output_name=item["id"],
    )
    return result.task_output()


def response_completed(item, model, output, ctx):
    status = output.value.get("status") if isinstance(output.value, dict) else None
    return EvalResult(
        score=status == "completed",
        comment=None if status == "completed" else f"expected completed response, got {status}",
        metadata={"status": status},
    )


def returned_audio(item, model, output, ctx):
    audio = [media for media in output.media if media.mime_type.startswith("audio/")]
    return EvalResult(score=bool(audio), comment=None if audio else "response did not include audio media")


exp.workflow = answer
exp.eval("response_completed", response_completed)
exp.eval("non_empty", NonEmpty(value=text()))
exp.eval(
    "contains_expected_phrase",
    Contains(container=text(), expected=item("expected_phrase"), case_sensitive=False),
    description="Expected phrase appears in the Realtime response transcript",
)
exp.eval("returned_audio", returned_audio)
