import math

import pandas as pd
import pytest

from causal_evaluation.evaluation.failure_graph import failure_graph
from causal_evaluation.evaluation.stats import (
    bootstrap_mean_ci,
    diff_proportions_ci,
    extract_yes_no,
    mcnemar_exact,
    wilson_ci,
)
from causal_evaluation.evaluation.variant_analysis import p_yes_shift, paired_accuracy_gap


def test_wilson_matches_reference_value():
    lo, hi = wilson_ci(81, 263)
    assert lo == pytest.approx(0.2553, abs=1e-4)
    assert hi == pytest.approx(0.3662, abs=1e-4)
    assert all(math.isnan(v) for v in wilson_ci(0, 0))


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)


def test_bootstrap_and_newcombe():
    mean, lo, hi = bootstrap_mean_ci([1, 1, 1, 1])
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)
    d, lo, hi = diff_proportions_ci(56, 70, 48, 80)
    assert d == pytest.approx(0.2)
    assert lo == pytest.approx(0.0524, abs=1e-3)
    assert hi == pytest.approx(0.3339, abs=1e-3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("Yes.", "yes"),
        ("A: no", "no"),
        ("**No**", "no"),
        ("Answer: Yes, because", "yes"),
        ("The answer is no.", "no"),
        ("yes or no? It depends", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_yes_no(text, expected):
    assert extract_yes_no(text) == expected


def _scored(rows):
    return pd.DataFrame(
        rows, columns=["model", "pair_id", "variant", "query_type", "answer", "p_yes", "correct"]
    )


def test_paired_gap_and_shift():
    rows = []
    for i in range(4):
        pid = f"p{i}"
        rows.append(("m", pid, "commonsense", "ate", "yes", 0.9, True))
        rows.append(("m", pid, "anticommonsense", "ate", "yes", 0.4, i == 0))
    df = _scored(rows)
    gap = paired_accuracy_gap(df).iloc[0]
    assert gap["acc_common"] == 1.0 and gap["acc_anti"] == 0.25
    assert gap["only_common_correct"] == 3 and gap["only_anti_correct"] == 0
    shift = p_yes_shift(df).iloc[0]
    assert shift["shift"] == pytest.approx(0.5)


def test_failure_graph_lift():
    # In each group, A and B fail together; C is independent of both.
    rows = []
    for g in range(40):
        fail_ab = g % 2 == 0
        rows += [
            {
                "rung": 1,
                "query_type": "A",
                "variant": "v",
                "story_id": g,
                "graph_id": "x",
                "correct": not fail_ab,
            },
            {
                "rung": 2,
                "query_type": "B",
                "variant": "v",
                "story_id": g,
                "graph_id": "x",
                "correct": not fail_ab,
            },
            {
                "rung": 3,
                "query_type": "C",
                "variant": "v",
                "story_id": g,
                "graph_id": "x",
                "correct": g % 4 < 2,
            },
        ]
    nodes, edges = failure_graph(pd.DataFrame(rows), min_pairs=10)
    assert set(nodes["error_rate"].round(2)) == {0.5}
    ab = edges[(edges["source"] == (1, "A")) & (edges["target"] == (2, "B"))]
    assert ab["lift"].iloc[0] == pytest.approx(2.0)
    assert ((edges["source"] == (1, "A")) & (edges["target"] == (3, "C"))).sum() == 0
