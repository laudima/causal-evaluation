"""Join sampled questions with model responses/evaluations into viewer records."""

import json
from pathlib import Path
from typing import Any

from .causal_diagram import layout_for_topology, parse_reasoning_graph


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_by_question_id(
    records: list[dict[str, Any]], model: str | None
) -> dict[str, dict[str, Any]]:
    """Index by question_id, optionally restricted to one model.

    responses.jsonl/evaluations.jsonl accumulate rows across every model run
    ever executed against a given path. Without a `model` filter, indexing by
    question_id alone means whichever row happens to be last in the file wins
    for a given question - silently mixing models. Passing `model` avoids
    that collision.
    """
    filtered = (r for r in records if model is None or r.get("model") == model)
    return {r["question_id"]: r for r in filtered}


def build_question_records(
    sample_path: str | Path,
    responses_path: str | Path | None = None,
    evaluations_path: str | Path | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Join sampled questions with their model responses/evaluations, one record per question.

    `model` restricts responses/evaluations to a single model's rows, so a
    shared responses/evaluations path holding multiple models' results
    doesn't silently collide (see `_index_by_question_id`).
    """
    questions = _read_jsonl(sample_path)
    responses = _index_by_question_id(_read_jsonl(responses_path), model)
    evaluations = _index_by_question_id(_read_jsonl(evaluations_path), model)

    records = []
    for question in questions:
        question_id = question["question_id"]
        meta = question.get("meta") or {}
        treatment = meta.get("treatment", "X")
        outcome = meta.get("outcome", "Y")
        topology = question.get("topology", "unknown")

        variables, edges = parse_reasoning_graph(question.get("reasoning"))
        layout = layout_for_topology(topology, variables, edges, treatment, outcome)

        response = responses.get(question_id)
        response_body = (response or {}).get("response") or {}
        reasoning_body = (response or {}).get("reasoning") or {}
        evaluation = evaluations.get(question_id)

        records.append(
            {
                "question_id": question_id,
                "story_id": meta.get("story_id", ""),
                "topology": topology,
                "query_type": question.get("question_type", "unknown"),
                "causal_level": question.get("causal_level", "unknown"),
                "question": question.get("question", ""),
                "given_info": question.get("given_info", ""),
                "answer": question.get("answer", ""),
                "treatment": treatment,
                "outcome": outcome,
                "variables": variables,
                "edges": edges,
                "layout": layout,
                "model": (response or {}).get("model"),
                "response_text": response_body.get("text"),
                "response_error": response_body.get("error"),
                "reasoning_text": reasoning_body.get("text"),
                "correct": (evaluation or {}).get("correct"),
            }
        )
    return records
