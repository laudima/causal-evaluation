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
