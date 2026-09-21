"""Response scoring."""


def normalize_answer(value: str) -> str:
    """Normalize an answer for exact-match evaluation."""
    return " ".join(value.strip().casefold().split())


def score_response(expected: str, actual: str) -> bool:
    """Return whether a model answer matches the expected answer."""
    return normalize_answer(expected) == normalize_answer(actual)
