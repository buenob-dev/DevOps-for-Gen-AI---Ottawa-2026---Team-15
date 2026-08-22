# statement-classifier

Classifies a statement as a checkable **fact** or an unverifiable **opinion**,
given the surrounding context it appeared in. One stage of the hackathon's
fact-checking pipeline — a separate component is responsible for turning
`"fact"` statements into a researched ruling.

Built with [LangChain](https://python.langchain.com/) against
[OpenRouter](https://openrouter.ai/) as the model gateway, so the backing
model is just a string (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`)
and swapping providers doesn't touch any code.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

## Usage

```python
from statement_classifier import classify_statement

result = classify_statement(
    statement="This is a test",
    surrounding_context="We are testing. This is a test. Test is now over.",
)
print(result.class_, result.confidence)  # e.g. "opinion" 0.65
```

Or classify a whole batch matching the pipeline's JSON shape:

```python
from statement_classifier import StatementsInput, classify_batch

batch = StatementsInput.model_validate({
    "statements": [
        {
            "surroundingContext": "We are testing. This is a test. Test is now over.",
            "statement": "This is a test",
        }
    ]
})
result = classify_batch(batch)
print(result.model_dump(by_alias=True))
```

which produces:

```json
{
  "statements": [
    {
      "surroundingContext": "We are testing. This is a test. Test is now over.",
      "statement": "This is a test",
      "classification": {
        "class": "opinion",
        "confidence": 0.65
      }
    }
  ]
}
```

`classify_batch` re-uses one OpenRouter connection across every statement in
the batch. `class_`/`"class"` is the field name inside `Classification` -
`class` is a reserved word in Python, so the attribute is `class_` and the
JSON alias (used by `model_dump(by_alias=True)`) is `class`.

## Configuration

| Env var | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENROUTER_API_KEY` | yes | - | Auth for OpenRouter. Get one at https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | no | `openai/gpt-4o-mini` | Any [OpenRouter model id](https://openrouter.ai/models) |

## Testing

```bash
uv run pytest
```

Tests run against a fake chat model (see `tests/conftest.py`) so they need
no API key and make no network calls.
