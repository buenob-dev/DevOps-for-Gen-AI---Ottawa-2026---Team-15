from types import SimpleNamespace

import httpx2
import pytest
from openai import APITimeoutError, AuthenticationError
from pydantic import ValidationError

from statement_classifier import classifier as classifier_module
from statement_classifier.models import (
    ClassifierError,
    ClassifierInput,
    StatementInput,
)


class FakeStructuredModel:
    """Stands in for `ChatOpenAI(...).with_structured_output(...)` — the LLM boundary."""

    def __init__(self, handlers: dict[str, callable]):
        self.handlers = handlers
        self.calls: list[str] = []

    async def ainvoke(self, prompt: str):
        self.calls.append(prompt)
        for needle, handler in self.handlers.items():
            if needle in prompt:
                return handler()
        raise AssertionError(f"FakeStructuredModel: no handler matched prompt: {prompt!r}")


def _ok(classification: str, confidence: float = 0.9):
    def handler():
        return SimpleNamespace(classification=classification, confidence=confidence)

    return handler


def _fail(exc: BaseException):
    def handler():
        raise exc

    return handler


def _patch_model(monkeypatch, fake_model: FakeStructuredModel) -> FakeStructuredModel:
    monkeypatch.setattr(classifier_module, "_get_structured_model", lambda: fake_model)
    return fake_model


def _statement(text: str, context: str | None = None) -> StatementInput:
    return StatementInput(surroundingContext=context or text, statement=text)


async def test_classify_single_fact_statement(monkeypatch):
    fake = _patch_model(
        monkeypatch,
        FakeStructuredModel({"The sky is blue": _ok("fact", 0.95)}),
    )
    input_ = ClassifierInput(statements=[_statement("The sky is blue")])

    output = await classifier_module.classify_statements(input_)

    result = output.statements[0]
    assert result.classification is not None
    assert result.classification.class_ == "fact"
    assert 0.0 <= result.classification.confidence <= 1.0
    assert result.error is None
    assert len(fake.calls) == 1


async def test_classify_single_opinion_statement(monkeypatch):
    _patch_model(
        monkeypatch,
        FakeStructuredModel({"Pizza is the best food": _ok("opinion", 0.8)}),
    )
    input_ = ClassifierInput(statements=[_statement("Pizza is the best food")])

    output = await classifier_module.classify_statements(input_)

    result = output.statements[0]
    assert result.classification.class_ == "opinion"
    assert 0.0 <= result.classification.confidence <= 1.0
    assert result.error is None


async def test_mixed_batch_classifies_each_statement_independently(monkeypatch):
    _patch_model(
        monkeypatch,
        FakeStructuredModel(
            {
                "Water boils at 100C": _ok("fact", 0.99),
                "Opinions are the best": _ok("opinion", 0.7),
            }
        ),
    )
    input_ = ClassifierInput(
        statements=[
            _statement("Water boils at 100C"),
            _statement("Opinions are the best"),
        ]
    )

    output = await classifier_module.classify_statements(input_)

    classes = {s.statement: s.classification.class_ for s in output.statements}
    assert classes == {
        "Water boils at 100C": "fact",
        "Opinions are the best": "opinion",
    }


async def test_empty_batch_returns_empty_result_without_error(monkeypatch):
    fake = _patch_model(monkeypatch, FakeStructuredModel({}))
    input_ = ClassifierInput(statements=[])

    output = await classifier_module.classify_statements(input_)

    assert output.statements == []
    assert fake.calls == []


async def test_one_statement_failing_does_not_affect_siblings(monkeypatch):
    _patch_model(
        monkeypatch,
        FakeStructuredModel(
            {
                "This one is broken": _fail(RuntimeError("boom")),
                "This one is fine": _ok("fact", 0.6),
            }
        ),
    )
    input_ = ClassifierInput(
        statements=[
            _statement("This one is broken"),
            _statement("This one is fine"),
        ]
    )

    output = await classifier_module.classify_statements(input_)

    by_statement = {s.statement: s for s in output.statements}

    broken = by_statement["This one is broken"]
    assert broken.classification is None
    assert broken.error is not None
    assert broken.error.code == "LLM_ERROR"

    fine = by_statement["This one is fine"]
    assert fine.classification is not None
    assert fine.classification.class_ == "fact"
    assert fine.error is None


async def test_timeout_after_retries_exhausted_yields_llm_timeout_error(monkeypatch):
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    _patch_model(
        monkeypatch,
        FakeStructuredModel({"Slow statement": _fail(APITimeoutError(request))}),
    )
    input_ = ClassifierInput(statements=[_statement("Slow statement")])

    output = await classifier_module.classify_statements(input_)

    result = output.statements[0]
    assert result.classification is None
    assert result.error.code == "LLM_TIMEOUT"


async def test_bad_model_output_yields_parse_error(monkeypatch):
    def raise_validation_error():
        try:
            classifier_module._LLMClassification(classification="fact", confidence=2.0)
        except ValidationError as exc:
            raise exc

    _patch_model(
        monkeypatch,
        FakeStructuredModel({"Weird statement": raise_validation_error}),
    )
    input_ = ClassifierInput(statements=[_statement("Weird statement")])

    output = await classifier_module.classify_statements(input_)

    result = output.statements[0]
    assert result.classification is None
    assert result.error.code == "PARSE_ERROR"


async def test_invalid_api_key_aborts_whole_batch(monkeypatch):
    request = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx2.Response(401, request=request)
    _patch_model(
        monkeypatch,
        FakeStructuredModel(
            {"Anything": _fail(AuthenticationError("invalid key", response=response, body=None))}
        ),
    )
    input_ = ClassifierInput(statements=[_statement("Anything")])

    with pytest.raises(ClassifierError) as exc_info:
        await classifier_module.classify_statements(input_)

    assert exc_info.value.code == "AUTH_ERROR"


async def test_missing_api_key_raises_before_any_llm_call(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []

    def fail_if_called():
        calls.append(True)
        raise AssertionError("should not build the chat model without an API key")

    monkeypatch.setattr(classifier_module, "ChatOpenAI", lambda **kwargs: fail_if_called())

    input_ = ClassifierInput(statements=[_statement("Anything")])

    with pytest.raises(ClassifierError) as exc_info:
        await classifier_module.classify_statements(input_)

    assert exc_info.value.code == "MISSING_API_KEY"
    assert calls == []


async def test_non_positive_concurrency_rejected_before_any_llm_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        classifier_module,
        "_get_structured_model",
        lambda: calls.append(True) or FakeStructuredModel({}),
    )
    input_ = ClassifierInput(statements=[_statement("Anything")])

    with pytest.raises(ClassifierError) as exc_info:
        await classifier_module.classify_statements(input_, concurrency=0)

    assert exc_info.value.code == "INVALID_INPUT"
    assert calls == []


def test_parse_classifier_input_rejects_malformed_input():
    with pytest.raises(ClassifierError) as exc_info:
        classifier_module.parse_classifier_input({"statements": [{"surroundingContext": "x"}]})

    assert exc_info.value.code == "INVALID_INPUT"


def test_parse_classifier_input_accepts_valid_input():
    parsed = classifier_module.parse_classifier_input(
        {"statements": [{"surroundingContext": "ctx", "statement": "s"}]}
    )
    assert parsed.statements[0].statement == "s"
