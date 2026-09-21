"""Deterministic benchmark subset selection."""

import random
from collections import defaultdict

from .validation import BenchmarkQuestion


def sample_questions(
    questions: list[BenchmarkQuestion], size: int, seed: int
) -> list[BenchmarkQuestion]:
    """Select a reproducible, approximately stratified subset by causal level."""
    if size < 0:
        raise ValueError("size must be non-negative")
    if size >= len(questions):
        return list(questions)
    groups: dict[str, list[BenchmarkQuestion]] = defaultdict(list)
    for question in questions:
        groups[question.causal_level].append(question)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[BenchmarkQuestion] = []
    while groups and len(selected) < size:
        for level in sorted(list(groups)):
            if groups[level] and len(selected) < size:
                selected.append(groups[level].pop())
            if not groups[level]:
                del groups[level]
    return selected
