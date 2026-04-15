from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class FakeUsageDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class FakeUsage:
    def __init__(
        self,
        *,
        input_tokens: int = 3,
        cached_tokens: int = 1,
        output_tokens: int = 5,
        reasoning_tokens: int = 2,
    ) -> None:
        self.input_tokens = input_tokens
        self.input_tokens_details = {"cached_tokens": cached_tokens}
        self.output_tokens = output_tokens
        self.output_tokens_details = {"reasoning_tokens": reasoning_tokens}
        self.total_tokens = input_tokens + output_tokens


class FakeResponse:
    def __init__(self, text: str, response_id: str = "resp_test") -> None:
        self.id = response_id
        self.output_text = text
        self.usage = FakeUsage()

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "id": self.id,
            "output_text": self.output_text,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "input_tokens_details": self.usage.input_tokens_details,
                "output_tokens": self.usage.output_tokens,
                "output_tokens_details": self.usage.output_tokens_details,
                "total_tokens": self.usage.total_tokens,
            },
        }


class FakeResponses:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        text = kwargs.get("input", "")
        if "fail" in str(text).lower():
            raise RuntimeError("fake failure")
        return FakeResponse(text=f"answer: {text}")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses = FakeResponses(self.calls)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()

