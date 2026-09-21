"""Load CLadder exports without modifying raw files."""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .validation import BenchmarkQuestion


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """Load and validate JSON array, JSONL, or CSV benchmark records."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        records = pd.read_csv(source).to_dict(orient="records")
    elif source.suffix.lower() == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
    else:
        with source.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    questions = [BenchmarkQuestion.model_validate(record) for record in records]
    ids = [question.stable_id() for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark question identifiers")
    return questions


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
