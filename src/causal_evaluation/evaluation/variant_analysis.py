"""Tables for the commonsense-alignment hypotheses (H1, H1b, H2)."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ..data.variants import RELATIONAL_QUERY_TYPES
from .stats import bootstrap_mean_ci, diff_proportions_ci, mcnemar_exact, wilson_ci


def load_scored(sample_path: str | Path, result_paths: Iterable[str | Path]) -> pd.DataFrame:
    """Join sampled items with model scores; a prediction is "yes" when P(Yes) > 0.5.

    Results from a non-direct prompt strategy are labelled "<model>-<strategy>",
    so every per-model table keeps the strategies apart.
    """
    items = pd.read_json(sample_path, lines=True)
    frames = [pd.read_json(path, lines=True) for path in result_paths]
    if not frames:
        raise FileNotFoundError("No result files given")
    results = pd.concat(frames, ignore_index=True)
    if "prompt_strategy" not in results:
        results["prompt_strategy"] = "direct"
    results["prompt_strategy"] = results["prompt_strategy"].fillna("direct")
    results = results.drop_duplicates(["item_id", "model", "prompt_strategy"], keep="last")
    other = results["prompt_strategy"] != "direct"
    results.loc[other, "model"] = (
        results.loc[other, "model"] + "-" + results.loc[other, "prompt_strategy"]
    )
    df = results.merge(items, on="item_id", how="inner", validate="many_to_one")
    df["pred"] = (df["p_yes"] > 0.5).map({True: "yes", False: "no"})
    df["correct"] = df["pred"] == df["answer"]
    return df


def rate_table(df: pd.DataFrame, by: list[str], flag: str = "correct") -> pd.DataFrame:
    """Proportion of `flag` per group with 95% Wilson intervals."""
    table = df.groupby(by)[flag].agg(n="size", k="sum").reset_index()
    bounds = [wilson_ci(int(k), int(n)) for k, n in zip(table["k"], table["n"], strict=True)]
    table["rate"] = table["k"] / table["n"]
    table["lo"] = [b[0] for b in bounds]
    table["hi"] = [b[1] for b in bounds]
    return table


def paired_accuracy_gap(df: pd.DataFrame) -> pd.DataFrame:
    """H1, paired part: commonsense minus anticommonsense accuracy on matched items."""
    rows = []
    for model, group in df[df["pair_id"].notna()].groupby("model"):
        wide = group.pivot_table(
            index="pair_id", columns="variant", values="correct", aggfunc="first"
        )
        if not {"commonsense", "anticommonsense"} <= set(wide.columns):
            continue
        wide = wide.dropna(subset=["commonsense", "anticommonsense"]).astype(bool)
        common, anti = wide["commonsense"], wide["anticommonsense"]
        mean, lo, hi = bootstrap_mean_ci(common.astype(int) - anti.astype(int))
        b = int((common & ~anti).sum())
        c = int((~common & anti).sum())
        rows.append(
            {
                "model": model,
                "pairs": len(wide),
                "acc_common": common.mean(),
                "acc_anti": anti.mean(),
                "gap": mean,
                "gap_lo": lo,
                "gap_hi": hi,
                "only_common_correct": b,
                "only_anti_correct": c,
                "mcnemar_p": mcnemar_exact(b, c),
            }
        )
    return pd.DataFrame(rows)


def nonsense_gap(df: pd.DataFrame) -> pd.DataFrame:
    """H1, unpaired part: commonsense minus noncommonsense accuracy."""
    rows = []
    for model, group in df.groupby("model"):
        common = group[group["variant"] == "commonsense"]["correct"]
        nonsense = group[group["variant"] == "noncommonsense"]["correct"]
        if common.empty or nonsense.empty:
            continue
        gap, lo, hi = diff_proportions_ci(
            int(common.sum()), len(common), int(nonsense.sum()), len(nonsense)
        )
        rows.append(
            {
                "model": model,
                "acc_common": common.mean(),
                "acc_nonsense": nonsense.mean(),
                "gap": gap,
                "gap_lo": lo,
                "gap_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def p_yes_shift(df: pd.DataFrame) -> pd.DataFrame:
    """H1b: mean P(Yes) commonsense minus anticommonsense on matched relational items.

    Reported separately for items whose correct answer is yes and no.
    """
    subset = df[df["pair_id"].notna() & df["query_type"].isin(RELATIONAL_QUERY_TYPES)]
    rows = []
    for (model, answer), group in subset.groupby(["model", "answer"]):
        wide = group.pivot_table(
            index="pair_id", columns="variant", values="p_yes", aggfunc="first"
        )
        if not {"commonsense", "anticommonsense"} <= set(wide.columns):
            continue
        wide = wide.dropna(subset=["commonsense", "anticommonsense"])
        if wide.empty:
            continue
        mean, lo, hi = bootstrap_mean_ci(wide["commonsense"] - wide["anticommonsense"])
        rows.append(
            {
                "model": model,
                "answer": answer,
                "pairs": len(wide),
                "p_yes_common": wide["commonsense"].mean(),
                "p_yes_anti": wide["anticommonsense"].mean(),
                "shift": mean,
                "shift_lo": lo,
                "shift_hi": hi,
            }
        )
    return pd.DataFrame(rows)
