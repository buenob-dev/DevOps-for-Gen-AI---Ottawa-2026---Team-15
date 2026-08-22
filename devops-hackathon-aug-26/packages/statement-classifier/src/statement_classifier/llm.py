"""LLM setup: LangChain's ChatOpenAI pointed at OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_openrouter_chat_model(
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Build a LangChain chat model backed by OpenRouter.

    Reads the API key from `OPENROUTER_API_KEY` and, unless overridden, the
    model id from `OPENROUTER_MODEL` (falling back to `DEFAULT_MODEL`).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Get a key from https://openrouter.ai/keys "
            "and export it, or add it to a .env file."
        )

    return ChatOpenAI(
        model=model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )
