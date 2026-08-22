import pytest


@pytest.fixture(autouse=True)
def openrouter_api_key(monkeypatch):
    """Most tests mock the LLM boundary directly, but _get_structured_model still
    checks for an API key before building the chat model, so give every test one
    unless it explicitly deletes it to test the missing-key path."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
