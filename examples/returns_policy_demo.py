from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, Type

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from prism_evals import EvalResult, Experiment, ModelConfig, TaskOutput


BASE_DIR = Path(__file__).parent
POLICY_PATH = BASE_DIR / "prompts" / "returns_policy.md"
PROMPT_PATH = BASE_DIR / "prompts" / "returns_policy_weak.md"
POLICY = POLICY_PATH.read_text(encoding="utf-8").strip()
PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
Decision = Literal["ELIGIBLE", "NOT_ELIGIBLE", "NEEDS_INFO"]


class PolicyAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    response: str = Field(description="Customer-facing answer grounded in the return policy.")


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=1)
    reason: str


def pydantic_json_schema(name: str, model: Type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": model.model_json_schema(),
    }


ANSWER_SCHEMA = pydantic_json_schema("return_policy_answer", PolicyAnswer)
JUDGE_SCHEMA = pydantic_json_schema("return_policy_judge", JudgeResult)


def response_params(model: ModelConfig, schema: dict[str, Any]) -> dict[str, Any]:
    params = copy.deepcopy(model.params)
    text_params = dict(params.pop("text", {}))
    text_params["format"] = schema
    return {**params, "text": text_params}


def selected_models() -> list[ModelConfig]:
    models = [
        ModelConfig(
            key="gpt54_mini_low",
            model="gpt-5.4-mini",
            params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        ),
        ModelConfig(
            key="gpt55_low",
            model="gpt-5.5",
            params={"reasoning": {"effort": "low"}, "text": {"verbosity": "low"}},
        ),
    ]
    requested = os.getenv("PRISM_RETURNS_MODEL_KEYS")
    if not requested:
        return [models[0]]
    keys = {key.strip() for key in requested.split(",") if key.strip()}
    selected = [model for model in models if model.key in keys]
    if not selected:
        available = ", ".join(model.key for model in models)
        raise ValueError(f"PRISM_RETURNS_MODEL_KEYS did not match any models. Available: {available}")
    return selected


exp = Experiment(
    name="returns_policy_demo",
    dataset="datasets/returns_policy_demo.csv",
    output_dir="runs",
    concurrency=4,
    resume=True,
    repetitions=1,
    artifacts=[POLICY_PATH.relative_to(BASE_DIR), PROMPT_PATH.relative_to(BASE_DIR)],
    metadata={"demo_stage": "single_step_policy_answer", "prompt_strength": "intentionally_weak"},
    openai_client=AsyncOpenAI(),
)
exp.models(selected_models())


async def answer_return_question(item, model, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **response_params(model, ANSWER_SCHEMA),
        input=(
            f"{PROMPT}\n\n"
            f"Policy:\n{POLICY}\n\n"
            f"Customer question:\n{item['customer_message']}"
        ),
    )
    answer = PolicyAnswer.model_validate_json(response.output_text)
    return TaskOutput(
        text=answer.response,
        value=answer.model_dump(mode="json"),
        metadata={"raw_json": response.output_text},
    )


def split_terms(value: str) -> list[str]:
    return [term.strip().lower() for term in value.split("|") if term.strip()]


def response_text(output: TaskOutput) -> str:
    if isinstance(output.value, dict):
        return str(output.value.get("response", ""))
    return output.text


def decision_exact_match(item, model, output, ctx):
    actual = output.value.get("decision") if isinstance(output.value, dict) else None
    expected = item["expected_decision"]
    return EvalResult(
        score=actual == expected,
        comment=None if actual == expected else f"expected {expected}, got {actual}",
        metadata={"expected_decision": expected, "actual_decision": actual},
    )


def required_terms_present(item, model, output, ctx):
    terms = split_terms(item["required_terms"])
    text = response_text(output).lower()
    missing = [term for term in terms if term not in text]
    return EvalResult(
        score=not missing,
        comment=None if not missing else f"missing required terms: {', '.join(missing)}",
        metadata={"required_terms": terms, "missing_terms": missing},
    )


def forbidden_terms_absent(item, model, output, ctx):
    terms = split_terms(item["forbidden_terms"])
    text = response_text(output).lower()
    present = [term for term in terms if term in text]
    return EvalResult(
        score=not present,
        comment=None if not present else f"forbidden terms present: {', '.join(present)}",
        metadata={"forbidden_terms": terms, "present_terms": present},
    )


def required_term_coverage(item, model, output, ctx):
    terms = split_terms(item["required_terms"])
    text = response_text(output).lower()
    if not terms:
        return EvalResult(score=1.0, metadata={"required_terms": [], "matched_terms": []})
    matched = [term for term in terms if term in text]
    score = len(matched) / len(terms)
    return EvalResult(
        score=score,
        comment=f"matched {len(matched)}/{len(terms)} required terms",
        metadata={"required_terms": terms, "matched_terms": matched},
    )


async def llm_policy_judge(item, model, output, ctx):
    response = await ctx.responses.create(
        model=model.model,
        **response_params(model, JUDGE_SCHEMA),
        input=(
            "You are grading a customer-support answer against a return policy.\n"
            "Give 1.0 only if the answer is grounded, clear, and does not overpromise.\n"
            "Give 0.5 for a partially useful answer with a missing caveat.\n"
            "Give 0.0 for an answer that contradicts the policy or guesses when it should ask for details.\n\n"
            "Caps: if the answer decision does not match the expected decision, the score must be at most 0.4.\n"
            "If a required policy fact is missing, the score must be at most 0.7.\n"
            "If the answer makes a forbidden promise, the score must be 0.0.\n\n"
            f"Policy:\n{POLICY}\n\n"
            f"Customer question:\n{item['customer_message']}\n\n"
            f"Expected decision:\n{item['expected_decision']}\n"
            f"Required policy facts:\n{item['required_terms']}\n"
            f"Forbidden promises:\n{item['forbidden_terms']}\n"
            f"Answer decision:\n{output.value.get('decision') if isinstance(output.value, dict) else None}\n"
            f"Answer text:\n{response_text(output)}"
        ),
    )
    judge = JudgeResult.model_validate_json(response.output_text)
    return EvalResult(
        score=judge.score,
        comment=judge.reason,
        metadata={"judge_model": model.model},
    )


exp.workflow = answer_return_question
exp.eval("decision_exact_match", decision_exact_match, description="Decision matches the expected label")
exp.eval("required_terms_present", required_terms_present, description="Answer includes required policy facts")
exp.eval("forbidden_terms_absent", forbidden_terms_absent, description="Answer avoids forbidden promises")
exp.eval("required_term_coverage", required_term_coverage, description="Fraction of required policy facts included")
exp.eval("llm_policy_judge", llm_policy_judge, description="LLM judge score for groundedness and usefulness")
