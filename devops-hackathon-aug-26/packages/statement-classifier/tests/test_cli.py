import json
from types import SimpleNamespace

from click.testing import CliRunner

from statement_classifier import classifier as classifier_module
from statement_classifier.cli import main


class FakeStructuredModel:
    async def ainvoke(self, prompt: str):
        if "fact statement" in prompt:
            return SimpleNamespace(classification="fact", confidence=0.9)
        return SimpleNamespace(classification="opinion", confidence=0.6)


def test_classify_file_to_file_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier_module, "_get_structured_model", lambda: FakeStructuredModel())

    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(
        json.dumps(
            {
                "statements": [
                    {"surroundingContext": "ctx", "statement": "This is a fact statement"},
                ]
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["classify", "--input", str(input_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output_data = json.loads(output_path.read_text())
    assert output_data["statements"][0]["classification"]["class"] == "fact"
    assert output_data["statements"][0]["error"] is None


def test_classify_stdin_to_stdout_exits_zero(monkeypatch):
    monkeypatch.setattr(classifier_module, "_get_structured_model", lambda: FakeStructuredModel())

    input_json = json.dumps(
        {"statements": [{"surroundingContext": "ctx", "statement": "an opinion statement"}]}
    )

    runner = CliRunner()
    result = runner.invoke(main, ["classify"], input=input_json)

    assert result.exit_code == 0, result.output
    output_data = json.loads(result.output)
    assert output_data["statements"][0]["classification"]["class"] == "opinion"


def test_classify_malformed_json_exits_nonzero_with_stderr_error(monkeypatch):
    monkeypatch.setattr(classifier_module, "_get_structured_model", lambda: FakeStructuredModel())

    runner = CliRunner()
    result = runner.invoke(main, ["classify"], input="{not valid json")

    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error["code"] == "INVALID_INPUT"


def test_classify_missing_api_key_exits_three(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    input_json = json.dumps(
        {"statements": [{"surroundingContext": "ctx", "statement": "anything"}]}
    )

    runner = CliRunner()
    result = runner.invoke(main, ["classify"], input=input_json)

    assert result.exit_code == 3
    error = json.loads(result.stderr)
    assert error["code"] == "MISSING_API_KEY"


def test_classify_concurrency_override_is_passed_through(tmp_path, monkeypatch):
    seen_concurrency = {}
    real_classify_sync = classifier_module.classify_statements_sync

    def spy(input_, *, concurrency=classifier_module.DEFAULT_CONCURRENCY):
        seen_concurrency["value"] = concurrency
        return real_classify_sync(input_, concurrency=concurrency)

    monkeypatch.setattr(classifier_module, "_get_structured_model", lambda: FakeStructuredModel())
    monkeypatch.setattr("statement_classifier.cli.classify_statements_sync", spy)

    input_json = json.dumps(
        {"statements": [{"surroundingContext": "ctx", "statement": "a fact statement"}]}
    )

    runner = CliRunner()
    result = runner.invoke(main, ["classify", "--concurrency", "2"], input=input_json)

    assert result.exit_code == 0, result.output
    assert seen_concurrency["value"] == 2
