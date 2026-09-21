# Causal Evaluation

Evaluating causal reasoning in language models with the CLadder benchmark. The pipeline validates benchmark records, samples by causal level, queries configurable providers, stores auditable raw responses, scores answers, and creates thesis-ready tables.

## Setup

Use Python 3.11+ and `uv`:

```sh
uv venv
uv pip install -e '.[dev]'
cp .env.example .env
```

The expected JSONL or CSV dataset fields are `question`, `answer`, and optional `question_id`, `causal_level`, `question_type`, and `topology`. Put the immutable source in `data/raw/`; never overwrite it. Provider secrets such as `LLM_API_KEY` are environment variables only.

## Configuration and run

Edit `configs/models.yaml`, `configs/experiments.yaml`, and `configs/prompts.yaml`. The default documented seed is `20260903`. A complete dry run is:

```sh
python -m causal_evaluation.cli validate-data --input data/raw/cladder.jsonl
python -m causal_evaluation.cli sample-data
python -m causal_evaluation.cli run --config configs/experiments.yaml --dry-run
python -m causal_evaluation.cli evaluate
python -m causal_evaluation.cli generate-reports
```

The dry-run provider follows the same storage and evaluation path without network calls. Real providers should implement `LLMClient` and read credentials from the environment. Raw responses are stored in `data/interim/responses.jsonl`; normalized evaluations are in `data/processed/evaluations.jsonl`. Reports use deterministic paths under `reports/`.

Runs are resumable: request IDs are UUID5 values derived from run, model, and question IDs, and duplicate records are skipped. For academic reproducibility, retain configuration files, seeds, dataset version/hash, provider/model parameters, timestamps, and raw responses. API studies should budget costs, respect provider terms and rate limits, minimize sensitive data, and report model/version and sampling parameters.

## Quality checks

```sh
pytest
ruff check .
ruff format --check .
```
