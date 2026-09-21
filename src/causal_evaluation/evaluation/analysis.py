"""Evaluation result construction and grouping."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .metrics import score_response


def evaluate_records(responses: list[dict], questions: list[dict]) -> pd.DataFrame:
    """Join raw responses to benchmark answers and produce normalized results.

    Responses accumulate historically across runs/re-samples (see `run`'s
    append-only storage), so `responses` may include question_ids from a
    prior sample no longer present in `questions` - those are skipped
    rather than raising, so evaluate keeps working after re-sampling.

    `run_id`/`provider`/`prompt_strategy` are carried forward from each
    response (falling back to "unknown" for responses written before these
    fields existed) so evaluations.jsonl is self-sufficient for grouping and
    graphing across multiple model/provider/prompt-strategy runs, without
    re-joining responses.jsonl.
    """
    answers = {item["question_id"]: item for item in questions}
    rows = []
    for record in responses:
        question = answers.get(record["question_id"])
        if question is None:
            continue
        text = record["response"].get("text", "")
        rows.append(
            {
                **{
                    key: question.get(key, "unknown")
                    for key in ("question_id", "causal_level", "question_type", "topology")
                },
                "model": record["model"],
                "run_id": record.get("run_id", "unknown"),
                "provider": record.get("provider", "unknown"),
                "prompt_strategy": record.get("prompt_strategy", "unknown"),
                "correct": score_response(question["answer"], text),
                "error": record["response"].get("error"),
            }
        )
    return pd.DataFrame(rows)


def load_evaluations(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read and concatenate multiple evaluations.jsonl files into one frame.

    Each configured experiment writes its own evaluations file (e.g. one per
    model/provider run); this combines them for cross-run comparisons and
    graphing without hand-rolled globbing/concatenation at each call site.
    """
    frames = [pd.read_json(path, lines=True) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
