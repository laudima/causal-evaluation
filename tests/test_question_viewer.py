import json

from causal_evaluation.viewer.records import build_question_records
from causal_evaluation.viewer.render import render_viewer_html


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_build_question_records_handles_missing_response_and_evaluation(tmp_path):
    sample_path = tmp_path / "sample.jsonl"
    _write_jsonl(sample_path, [{"question_id": "q1", "question": "Q?", "answer": "yes"}])

    records = build_question_records(sample_path)

    assert len(records) == 1
    assert records[0]["question_id"] == "q1"
    assert records[0]["model"] is None
    assert records[0]["response_text"] is None
    assert records[0]["correct"] is None


def test_build_question_records_joins_response_and_evaluation(tmp_path):
    sample_path = tmp_path / "sample.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    evaluations_path = tmp_path / "evaluations.jsonl"
    _write_jsonl(sample_path, [{"question_id": "q1", "question": "Q?", "answer": "yes"}])
    _write_jsonl(
        responses_path,
        [{"question_id": "q1", "model": "gpt-4o-mini", "response": {"text": "yes"}}],
    )
    _write_jsonl(evaluations_path, [{"question_id": "q1", "model": "gpt-4o-mini", "correct": True}])

    records = build_question_records(sample_path, responses_path, evaluations_path)

    assert records[0]["model"] == "gpt-4o-mini"
    assert records[0]["response_text"] == "yes"
    assert records[0]["correct"] is True


def test_build_question_records_model_filter_avoids_multi_model_collision(tmp_path):
    # Two models answered the same question_id into a shared responses/evaluations
    # path; without filtering by model, whichever row is last would silently win.
    sample_path = tmp_path / "sample.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    evaluations_path = tmp_path / "evaluations.jsonl"
    _write_jsonl(sample_path, [{"question_id": "q1", "question": "Q?", "answer": "yes"}])
    _write_jsonl(
        responses_path,
        [
            {"question_id": "q1", "model": "model-a", "response": {"text": "yes"}},
            {"question_id": "q1", "model": "model-b", "response": {"text": "no"}},
        ],
    )
    _write_jsonl(
        evaluations_path,
        [
            {"question_id": "q1", "model": "model-a", "correct": True},
            {"question_id": "q1", "model": "model-b", "correct": False},
        ],
    )

    records = build_question_records(sample_path, responses_path, evaluations_path, model="model-a")

    assert records[0]["model"] == "model-a"
    assert records[0]["response_text"] == "yes"
    assert records[0]["correct"] is True


def test_render_viewer_html_embeds_records_json():
    records = [
        {"question_id": "q1", "story_id": "s", "topology": "chain", "variables": {}, "edges": []}
    ]

    html = render_viewer_html(records)

    assert "<html" in html
    assert '"question_id": "q1"' in html
    assert "__RECORDS__" not in html
    assert "__SUPPORTED__" not in html
    assert "__CSS__" not in html
    assert "__JS__" not in html
