import json

from causal_evaluation.evaluation.analysis import evaluate_records, load_evaluations
from causal_evaluation.evaluation.metrics import score_response


def test_scoring_normalizes_case_and_whitespace():
    assert score_response(" Yes ", "yes")


def test_evaluation_preserves_metadata():
    result = evaluate_records(
        [{"question_id": "q1", "model": "dry-run", "response": {"text": "yes"}}],
        [
            {
                "question_id": "q1",
                "answer": "yes",
                "causal_level": "hard",
                "question_type": "effect",
                "topology": "chain",
            }
        ],
    )
    assert bool(result.iloc[0]["correct"])
    assert result.iloc[0]["causal_level"] == "hard"


def test_evaluation_skips_responses_from_a_prior_sample():
    # responses.jsonl accumulates historically across re-samples; a response
    # for a question_id no longer in the current sample must be skipped, not
    # raise, so evaluate keeps working after `sample-data` is re-run.
    result = evaluate_records(
        [
            {"question_id": "stale", "model": "dry-run", "response": {"text": "yes"}},
            {"question_id": "q1", "model": "dry-run", "response": {"text": "yes"}},
        ],
        [{"question_id": "q1", "answer": "yes", "causal_level": "hard"}],
    )
    assert list(result["question_id"]) == ["q1"]


def test_evaluation_carries_forward_run_provenance_fields():
    result = evaluate_records(
        [
            {
                "question_id": "q1",
                "model": "gpt-4o-mini",
                "run_id": "run-1",
                "provider": "openai",
                "prompt_strategy": "causal_cot",
                "response": {"text": "yes"},
            }
        ],
        [{"question_id": "q1", "answer": "yes", "causal_level": "hard"}],
    )
    assert result.iloc[0]["run_id"] == "run-1"
    assert result.iloc[0]["provider"] == "openai"
    assert result.iloc[0]["prompt_strategy"] == "causal_cot"


def test_evaluation_defaults_missing_provenance_fields_for_old_responses():
    # responses.jsonl written before run_id/provider/prompt_strategy were
    # first-class fields must still evaluate cleanly.
    result = evaluate_records(
        [{"question_id": "q1", "model": "gpt-4o-mini", "response": {"text": "yes"}}],
        [{"question_id": "q1", "answer": "yes", "causal_level": "hard"}],
    )
    assert result.iloc[0]["run_id"] == "unknown"
    assert result.iloc[0]["provider"] == "unknown"
    assert result.iloc[0]["prompt_strategy"] == "unknown"


def test_load_evaluations_concatenates_multiple_files(tmp_path):
    first = tmp_path / "evaluations-a.jsonl"
    second = tmp_path / "evaluations-b.jsonl"
    first.write_text(json.dumps({"question_id": "q1", "model": "model-a", "correct": True}) + "\n")
    second.write_text(
        json.dumps({"question_id": "q1", "model": "model-b", "correct": False}) + "\n"
    )

    combined = load_evaluations([first, second])

    assert sorted(combined["model"]) == ["model-a", "model-b"]


def test_prior_analysis_recovers_prior_and_answer_effects():
    import numpy as np
    import pandas as pd

    from causal_evaluation.evaluation.prior_analysis import (
        interpret,
        join_prior,
        paired_gaps,
        prior_vs_structure,
    )

    rng = np.random.default_rng(0)
    rows_ctx, rows_pri = [], []
    for k in range(300):
        answer = "yes" if k % 2 else "no"
        for variant, shift in (("commonsense", 1.0), ("anticommonsense", -1.0)):
            item = f"{variant}-{k}"
            prior_logit = shift + rng.normal(0, 1)
            ctx_logit = 0.5 * prior_logit + 2.0 * (answer == "yes") - 1.0 + rng.normal(0, 0.3)
            base = {
                "item_id": item,
                "pair_id": f"pair-{k}",
                "variant": variant,
                "query_type": "ate",
                "rung": 2,
                "answer": answer,
            }
            rows_ctx.append({**base, "model": "m", "p_yes": 1 / (1 + np.exp(-ctx_logit))})
            rows_pri.append(
                {**base, "model": "m-question_only", "p_yes": 1 / (1 + np.exp(-prior_logit))}
            )
    joined = join_prior(pd.DataFrame(rows_ctx), pd.DataFrame(rows_pri))
    assert len(joined) == 600
    fit = prior_vs_structure(joined, n_boot=200).iloc[0]
    assert abs(fit.b_prior - 0.5) < 0.05 and abs(fit.b_answer - 2.0) < 0.1
    assert fit.b_prior_lo < 0.5 < fit.b_prior_hi
    assert abs(fit.within_pair_slope - 0.5) < 0.05
    gaps = paired_gaps(joined)
    assert gaps.iloc[0].prior_gap > 0.2
    assert "survives" in interpret(gaps).iloc[0].reading
