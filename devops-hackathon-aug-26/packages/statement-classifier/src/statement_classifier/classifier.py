from __future__ import annotations

import asyncio
import os
from typing import Any, Literal, Optional

from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, AuthenticationError
from pydantic import BaseModel, Field, SecretStr, ValidationError

from .models import (
    Classification,
    ClassifierError,
    ClassifierInput,
    ClassifierOutput,
    ErrorDetail,
    StatementInput,
    StatementResult,
)

DEFAULT_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CONCURRENCY = 5

_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_SECONDS = 0.1


class _LLMClassification(BaseModel):
    """Schema the LLM must satisfy for a single statement, via structured output."""

    classification: Literal["fact", "opinion"]
    confidence: float = Field(ge=0.0, le=1.0)


def parse_classifier_input(raw: Any) -> ClassifierInput:
    """Validate raw (already-JSON-decoded) input, raising ClassifierError(INVALID_INPUT) on failure."""
    try:
        return ClassifierInput.model_validate(raw)
    except ValidationError as exc:
        raise ClassifierError(code="INVALID_INPUT", message=str(exc)) from exc


def _get_structured_model() -> Any:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ClassifierError(
            code="MISSING_API_KEY",
            message="OPENROUTER_API_KEY environment variable is not set",
        )

    model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)

    chat = ChatOpenAI(model=model_name, api_key=SecretStr(api_key), base_url=base_url)
    return chat.with_structured_output(_LLMClassification)


def _build_prompt(statement: StatementInput) -> str:
    return (
        "You are classifying a single statement extracted from a larger piece of "
        "text as either a checkable factual claim (\"fact\") or a subjective, "
        "non-checkable statement (\"opinion\"). Use the surrounding context only "
        "to disambiguate the statement; do not classify the surrounding context "
        "itself.\n\n"
        f"Surrounding context: {statement.surroundingContext}\n"
        f"Statement to classify: {statement.statement}"
    )


async def _classify_one(
    statement: StatementInput,
    structured_model: Any,
    semaphore: asyncio.Semaphore,
) -> StatementResult:
    async with semaphore:
        last_exc: Optional[BaseException] = None
        error_code: Literal["LLM_ERROR", "LLM_TIMEOUT", "PARSE_ERROR"] = "LLM_ERROR"

        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = await structured_model.ainvoke(_build_prompt(statement))
                return StatementResult(
                    surroundingContext=statement.surroundingContext,
                    statement=statement.statement,
                    classification=Classification.model_validate(
                        {"class": result.classification, "confidence": result.confidence}
                    ),
                    error=None,
                )
            except AuthenticationError as exc:
                # Invalid credentials fail the whole batch, not just this item.
                raise ClassifierError(code="AUTH_ERROR", message=str(exc)) from exc
            except APITimeoutError as exc:
                last_exc = exc
                error_code = "LLM_TIMEOUT"
            except (ValidationError, OutputParserException) as exc:
                last_exc = exc
                error_code = "PARSE_ERROR"
            except Exception as exc:  # noqa: BLE001 - isolate any other per-item failure
                last_exc = exc
                error_code = "LLM_ERROR"

            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        return StatementResult(
            surroundingContext=statement.surroundingContext,
            statement=statement.statement,
            classification=None,
            error=ErrorDetail(code=error_code, message=str(last_exc)),
        )


async def classify_statements(
    input: ClassifierInput, *, concurrency: int = DEFAULT_CONCURRENCY
) -> ClassifierOutput:
    if concurrency < 1:
        raise ClassifierError(
            code="INVALID_INPUT", message=f"concurrency must be >= 1, got {concurrency}"
        )

    if not input.statements:
        return ClassifierOutput(statements=[])

    structured_model = _get_structured_model()
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        asyncio.ensure_future(_classify_one(statement, structured_model, semaphore))
        for statement in input.statements
    ]
    try:
        results = await asyncio.gather(*tasks)
    except ClassifierError:
        for task in tasks:
            task.cancel()
        # Let cancellation land before re-raising, so no task's exception is
        # dropped silently ("Task exception was never retrieved").
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return ClassifierOutput(statements=list(results))


def classify_statements_sync(
    input: ClassifierInput, *, concurrency: int = DEFAULT_CONCURRENCY
) -> ClassifierOutput:
    return asyncio.run(classify_statements(input, concurrency=concurrency))
