"""Question-only priors: do the variable names carry a prior, and do in-context answers follow it?

The question-only run scores each matched item without the background (graph) or
the given data, so its P(Yes) is the model's prior for the question as worded.
"""

import numpy as np
import pandas as pd

from .stats import bootstrap_mean_ci

EPS = 1e-6


def logit(p, eps: float = EPS):
    """Logit with P clipped to [eps, 1 - eps]; the 4-bit models reach P below 1e-10."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def join_prior(context: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, item) with in-context and question-only P(Yes).

    `context` and `prior` are `load_scored` frames; model names may carry a
    "-<strategy>" suffix, which is stripped so both sides share a model key.
    """
    keep = ["item_id", "pair_id", "variant", "query_type", "rung", "answer"]
    ctx = context.assign(
        base=context["model"].str.replace(r"-(causal_cot|question_only)$", "", regex=True)
    )
    pri = prior.assign(base=prior["model"].str.replace(r"-question_only$", "", regex=True))
    joined = ctx[["base", *keep, "p_yes"]].merge(
        pri[["base", "item_id", "p_yes"]].rename(columns={"p_yes": "p_prior"}),
        on=["base", "item_id"],
        how="inner",
        validate="one_to_one",
    )
    joined = joined[joined["pair_id"].notna()].copy()
    joined["logit_ctx"] = logit(joined["p_yes"])
    joined["logit_prior"] = logit(joined["p_prior"])
    joined["y"] = (joined["answer"] == "yes").astype(int)
    return joined


def paired_gaps(joined: pd.DataFrame, query_types=None) -> pd.DataFrame:
    """Commonsensical minus anti-commonsensical mean P(Yes), question-only vs in context.

    A positive prior gap means the anti-commonsensical question, asked alone, gets
    less "Yes": the names carry the prior the plausibility account relies on.
    """
    data = joined if query_types is None else joined[joined["query_type"].isin(query_types)]
    rows = []
    for base, group in data.groupby("base"):
        wide = group.pivot_table(index="pair_id", columns="variant", values=["p_prior", "p_yes"])
        wide = wide.dropna()
        row = {"model": base, "pairs": len(wide)}
        for label, col in (("prior", "p_prior"), ("context", "p_yes")):
            diff = wide[(col, "commonsense")] - wide[(col, "anticommonsense")]
            row[f"{label}_gap"], row[f"{label}_lo"], row[f"{label}_hi"] = bootstrap_mean_ci(diff)
        rows.append(row)
    return pd.DataFrame(rows)


def _ols(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return coef, r2


def prior_vs_structure(
    joined: pd.DataFrame, n_boot: int = 2000, seed: int = 20260903
) -> pd.DataFrame:
    """Per model: logit P(Yes | context) ~ 1 + logit P(Yes | question only) + correct answer.

    `b_prior` is how far the in-context logit moves per unit of prior logit;
    `b_answer` is how far it moves when the correct answer is Yes, i.e. how much
    the given structure is used. Intervals come from a bootstrap over pairs.
    `r2_unique_prior` / `r2_unique_answer` are the drops in R^2 when that term is
    removed. Also reports the within-pair slope: the difference in context logits
    between the two names regressed on their difference in prior logits, which
    measures how much of a name's prior passes through the given structure.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for base, group in joined.groupby("base"):
        group = group.reset_index(drop=True)
        y = group["logit_ctx"].to_numpy()
        X = np.column_stack([np.ones(len(group)), group["logit_prior"], group["y"]])
        coef, r2 = _ols(y, X)
        _, r2_no_prior = _ols(y, X[:, [0, 2]])
        _, r2_no_answer = _ols(y, X[:, [0, 1]])

        wide = group.pivot_table(
            index="pair_id", columns="variant", values=["logit_ctx", "logit_prior"]
        ).dropna()
        d_ctx = (
            wide[("logit_ctx", "commonsense")] - wide[("logit_ctx", "anticommonsense")]
        ).to_numpy()
        d_pri = (
            wide[("logit_prior", "commonsense")] - wide[("logit_prior", "anticommonsense")]
        ).to_numpy()
        within = np.polyfit(d_pri, d_ctx, 1)[0] if d_pri.var() > 0 else np.nan

        pairs = group["pair_id"].unique()
        index = {p: np.flatnonzero(group["pair_id"].to_numpy() == p) for p in pairs}
        pair_pos = {p: i for i, p in enumerate(wide.index)}
        boot = []
        for _ in range(n_boot):
            sample = rng.choice(pairs, size=len(pairs), replace=True)
            rows_b = np.concatenate([index[p] for p in sample])
            c, _ = _ols(y[rows_b], X[rows_b])
            wp = [pair_pos[p] for p in sample if p in pair_pos]
            w = np.polyfit(d_pri[wp], d_ctx[wp], 1)[0] if np.var(d_pri[wp]) > 0 else np.nan
            boot.append([c[1], c[2], w])
        lo, hi = np.nanquantile(np.array(boot), [0.025, 0.975], axis=0)
        rows.append(
            {
                "model": base,
                "items": len(group),
                "b_prior": coef[1],
                "b_prior_lo": lo[0],
                "b_prior_hi": hi[0],
                "b_answer": coef[2],
                "b_answer_lo": lo[1],
                "b_answer_hi": hi[1],
                "r2": r2,
                "r2_unique_prior": r2 - r2_no_prior,
                "r2_unique_answer": r2 - r2_no_answer,
                "within_pair_slope": within,
                "within_lo": lo[2],
                "within_hi": hi[2],
            }
        )
    return pd.DataFrame(rows)


def interpret(gaps: pd.DataFrame, band: float = 0.04) -> pd.DataFrame:
    """Reading rule for each model, from the prior gap and the in-context gap."""

    def verdict(r):
        prior_large = r.prior_lo > band
        prior_null = r.prior_lo > -band and r.prior_hi < band
        ctx_null = r.context_lo > -band and r.context_hi < band
        if prior_large and ctx_null:
            return "names carry a prior, but the context overrides it"
        if prior_large and r.context_lo > 0:
            return "names carry a prior and part of it survives the context"
        if prior_null:
            return "no prior gap: the test cannot separate the hypotheses for this model"
        return "inconclusive: widen the sample or read the regression"

    return gaps.assign(reading=gaps.apply(verdict, axis=1))
