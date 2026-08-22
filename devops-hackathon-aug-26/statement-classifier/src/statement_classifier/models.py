"""Data contracts for the statement classifier.

These mirror the fact-checking pipeline's shared JSON shape: a statement is
paired with the surrounding context it was found in, and (after this
package runs) a classification of whether it is a checkable "fact" or an
unverifiable "opinion".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Classification(BaseModel):
    """The fact/opinion verdict for a single statement."""

    model_config = ConfigDict(populate_by_name=True)

    class_: Literal["fact", "opinion"] = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)


class StatementInput(BaseModel):
    """A single statement awaiting classification."""

    model_config = ConfigDict(populate_by_name=True)

    surrounding_context: str = Field(alias="surroundingContext")
    statement: str


class ClassifiedStatement(BaseModel):
    """A statement together with its classification verdict."""

    model_config = ConfigDict(populate_by_name=True)

    surrounding_context: str = Field(alias="surroundingContext")
    statement: str
    classification: Classification


class StatementsInput(BaseModel):
    """The `{"statements": [...]}` batch input shape."""

    statements: list[StatementInput]


class StatementsOutput(BaseModel):
    """The `{"statements": [...]}` batch output shape."""

    statements: list[ClassifiedStatement]
