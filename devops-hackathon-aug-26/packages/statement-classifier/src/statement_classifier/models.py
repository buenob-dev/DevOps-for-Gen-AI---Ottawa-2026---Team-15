from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ErrorCode = Literal[
    "INVALID_INPUT",
    "MISSING_API_KEY",
    "AUTH_ERROR",
    "LLM_ERROR",
    "LLM_TIMEOUT",
    "PARSE_ERROR",
    "INTERNAL_ERROR",
]


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str


class ClassifierError(Exception):
    """Raised for batch-level failures: the whole call aborts, nothing partial is returned."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def to_error_detail(self) -> ErrorDetail:
        return ErrorDetail(code=self.code, message=self.message)


class Classification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_: Literal["fact", "opinion"] = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)


class StatementInput(BaseModel):
    surroundingContext: str
    statement: str


class StatementResult(BaseModel):
    surroundingContext: str
    statement: str
    classification: Optional[Classification] = None
    error: Optional[ErrorDetail] = None


class ClassifierInput(BaseModel):
    statements: list[StatementInput]


class ClassifierOutput(BaseModel):
    statements: list[StatementResult]
