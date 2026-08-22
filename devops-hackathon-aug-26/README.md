# devops-hackathon-aug-26

Monorepo for the DevOps Hackathon (Aug 2026) fact-checking tool. Each
subdirectory is an independent, `uv`-managed Python package owned by a
different part of the pipeline - `cd` into one and run `uv sync` there.

## Pipeline

1. Statement extraction (elsewhere) produces `{surroundingContext, statement}`
   pairs from source text.
2. **[`statement-classifier`](./statement-classifier)** - classifies each
   statement as `fact` or `opinion`.
3. A downstream component (elsewhere) web-searches `fact` statements and
   attaches a `ruling` (justification + references).

## Subpackages

- [`statement-classifier`](./statement-classifier) - fact/opinion
  classification via LangChain + OpenRouter.
