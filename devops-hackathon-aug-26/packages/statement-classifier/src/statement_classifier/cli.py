from __future__ import annotations

import json
import sys
from typing import Optional

import click

from .classifier import classify_statements_sync, parse_classifier_input
from .models import ClassifierError

EXIT_SUCCESS = 0
EXIT_INTERNAL_ERROR = 1
EXIT_INVALID_INPUT = 2
EXIT_AUTH_ERROR = 3

_EXIT_CODE_BY_ERROR_CODE = {
    "INVALID_INPUT": EXIT_INVALID_INPUT,
    "MISSING_API_KEY": EXIT_AUTH_ERROR,
    "AUTH_ERROR": EXIT_AUTH_ERROR,
}


@click.group()
def main() -> None:
    """statement-classifier: classify statements as fact or opinion."""


@main.command()
@click.option(
    "--input",
    "input_path",
    default="-",
    help="Input JSON file to read a ClassifierInput document from. '-' (default) reads stdin.",
)
@click.option(
    "--output",
    "output_path",
    default="-",
    help="Output JSON file to write the ClassifierOutput document to. '-' (default) writes stdout.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=None,
    help="Override the default classification concurrency.",
)
def classify(input_path: str, output_path: str, concurrency: Optional[int]) -> None:
    """Classify a batch of statements as fact or opinion."""
    exit_code = _run_classify(input_path, output_path, concurrency)
    if exit_code != EXIT_SUCCESS:
        sys.exit(exit_code)


def _read_input(input_path: str) -> str:
    if input_path == "-":
        return sys.stdin.read()
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def _write_output(output_path: str, content: str) -> None:
    if output_path == "-":
        sys.stdout.write(content)
        sys.stdout.write("\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)


def _write_error(error: ClassifierError) -> None:
    sys.stderr.write(json.dumps(error.to_error_detail().model_dump()) + "\n")


def _run_classify(input_path: str, output_path: str, concurrency: Optional[int]) -> int:
    try:
        raw_text = _read_input(input_path)
    except OSError as exc:
        _write_error(ClassifierError(code="INVALID_INPUT", message=str(exc)))
        return EXIT_INVALID_INPUT

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _write_error(ClassifierError(code="INVALID_INPUT", message=f"invalid JSON: {exc}"))
        return EXIT_INVALID_INPUT

    try:
        classifier_input = parse_classifier_input(raw)
        kwargs = {} if concurrency is None else {"concurrency": concurrency}
        output = classify_statements_sync(classifier_input, **kwargs)
    except ClassifierError as exc:
        _write_error(exc)
        return _EXIT_CODE_BY_ERROR_CODE.get(exc.code, EXIT_INTERNAL_ERROR)
    except Exception as exc:  # noqa: BLE001 - unexpected/internal error, still reported as JSON
        _write_error(ClassifierError(code="INTERNAL_ERROR", message=str(exc)))
        return EXIT_INTERNAL_ERROR

    _write_output(output_path, output.model_dump_json(by_alias=True, indent=2))
    return EXIT_SUCCESS


if __name__ == "__main__":
    main()
