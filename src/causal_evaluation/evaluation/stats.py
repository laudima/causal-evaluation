"""Small statistics helpers for accuracy comparisons."""

import math
import re

import numpy as np

_Z95 = 1.959963984540054


def wilson_ci(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_mean_ci(
    values, n_boot: int = 10_000, seed: int = 20260903, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of `values` with a percentile bootstrap interval."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_boot, data.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return (
        float(data.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def diff_proportions_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = _Z95
) -> tuple[float, float, float]:
    """Difference p1 - p2 of independent proportions with Newcombe's hybrid Wilson interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, lower, upper)


_LEADING = re.compile(r"^\W*(?:a|answer)?\s*[:\-]?\s*\W*(yes|no)\b(?!\s+or\b)", re.IGNORECASE)
_ANY = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def extract_yes_no(text: str | None) -> str | None:
    """Recover a yes/no answer from free text, or None if it is ambiguous.

    Accepts a leading answer ("Yes.", "A: no", "**No**"); otherwise accepts the
    text only if every yes/no word in it agrees.
    """
    if not text:
        return None
    leading = _LEADING.match(text.strip())
    if leading:
        return leading.group(1).lower()
    found = {match.lower() for match in _ANY.findall(text)}
    return found.pop() if len(found) == 1 else None
