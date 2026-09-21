"""JSONL storage for resumable experiment artifacts."""

import json
from pathlib import Path
from typing import Any


def append_unique(path: str | Path, record: dict[str, Any], key: str) -> bool:
    """Append a record unless its key already exists; return whether written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if destination.exists():
        with destination.open(encoding="utf-8") as handle:
            existing = {json.loads(line).get(key) for line in handle if line.strip()}
    if record.get(key) in existing:
        return False
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    return True
