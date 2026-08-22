"""Test doubles that stand in for a real OpenRouter/LangChain chat model."""

from __future__ import annotations

from typing import Any

import pytest

from statement_classifier.models import Classification


class FakeStructuredLLM:
    """Returned by FakeLLM.with_structured_output(); always yields the same result."""

    def __init__(self, result: Classification) -> None:
        self.result = result
        self.invocations: list[Any] = []

    def invoke(self, messages: Any) -> Classification:
        self.invocations.append(messages)
        return self.result


class FakeLLM:
    """Duck-types the subset of a LangChain chat model this package relies on."""

    def __init__(self, result: Classification) -> None:
        self.result = result
        self.structured: FakeStructuredLLM | None = None

    def with_structured_output(self, schema: type) -> FakeStructuredLLM:
        assert schema is Classification
        self.structured = FakeStructuredLLM(self.result)
        return self.structured


@pytest.fixture
def fake_fact_llm() -> FakeLLM:
    return FakeLLM(Classification(class_="fact", confidence=0.9))


@pytest.fixture
def fake_opinion_llm() -> FakeLLM:
    return FakeLLM(Classification(class_="opinion", confidence=0.8))
