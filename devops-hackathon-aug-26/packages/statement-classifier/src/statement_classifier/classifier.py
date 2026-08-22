"""Fact/opinion classification for statements."""

from __future__ import annotations

from typing import Any

from .llm import get_openrouter_chat_model
from .models import (
    Classification,
    ClassifiedStatement,
    StatementInput,
    StatementsInput,
    StatementsOutput,
)
from .prompts import CLASSIFICATION_PROMPT


def classify_statement(
    statement: str,
    surrounding_context: str,
    *,
    llm: Any | None = None,
) -> Classification:
    """Classify a single statement as "fact" or "opinion".

    `llm` defaults to an OpenRouter chat model (see `llm.py`). Pass one in
    for testing, or to point at a different model. It only needs to
    support LangChain's `with_structured_output`.
    """
    llm = llm or get_openrouter_chat_model()
    structured_llm = llm.with_structured_output(Classification)
    messages = CLASSIFICATION_PROMPT.format_messages(
        statement=statement, surrounding_context=surrounding_context
    )
    return structured_llm.invoke(messages)


def classify(item: StatementInput, *, llm: Any | None = None) -> ClassifiedStatement:
    """Classify one statement, returning it paired with its verdict."""
    classification = classify_statement(
        item.statement, item.surrounding_context, llm=llm
    )
    return ClassifiedStatement(
        surrounding_context=item.surrounding_context,
        statement=item.statement,
        classification=classification,
    )


def classify_batch(
    batch: StatementsInput, *, llm: Any | None = None
) -> StatementsOutput:
    """Classify every statement in a `{"statements": [...]}` batch."""
    llm = llm or get_openrouter_chat_model()
    return StatementsOutput(
        statements=[classify(item, llm=llm) for item in batch.statements]
    )
