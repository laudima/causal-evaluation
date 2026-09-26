"""JSONL storage for resumable experiment artifacts."""

import json
from pathlib import Path
from typing import Any


def load_existing_keys(path: str | Path, key: str) -> set[Any]:
    """Return the set of existing `key` values already stored at `path` (empty if absent)."""
    destination = Path(path)
    if not destination.exists():
        return set()
    with destination.open(encoding="utf-8") as handle:
        return {json.loads(line).get(key) for line in handle if line.strip()}


def append_unique(
    path: str | Path, record: dict[str, Any], key: str, existing: set[Any] | None = None
) -> bool:
    """Append a record unless its key already exists; return whether written.

    Pass a pre-loaded `existing` set (see `load_existing_keys`) when writing
    many records in a loop - re-reading the whole file on every call (the
    fallback when `existing` is omitted) turns a large resumed run into an
    O(n^2) file scan. `existing` is updated in place as records are written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if existing is None:
        existing = load_existing_keys(path, key)
    if record.get(key) in existing:
        return False
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    existing.add(record.get(key))
    return True
