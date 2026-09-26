import json

from causal_evaluation.data.validation import BenchmarkQuestion
from causal_evaluation.experiments.runner import run_causal_cot_experiment, run_experiment
from causal_evaluation.experiments.schema import ResponseRecord
from causal_evaluation.models.base import ModelResponse


class FakeClient:
    """Returns a fixed answer for every prompt, recording what it was asked."""

    def __init__(self, text: str = "yes"):
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> ModelResponse:
        self.prompts.append(prompt)
        return ModelResponse(text=self.text)


def _read_records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_run_experiment_writes_schema_valid_default_strategy_records(tmp_path):
    questions = [BenchmarkQuestion(question="Q1?", answer="yes", question_id="q1")]
    output_path = tmp_path / "responses.jsonl"

    written = run_experiment(
        questions,
        FakeClient(),
        "{question}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )

    assert written == 1
    lines = _read_records(output_path)
    assert len(lines) == 1
    record = ResponseRecord.model_validate(lines[0])
    assert record.model == "test-model"
    assert record.provider == "test-provider"
    assert record.prompt_strategy == "default"
    assert record.reasoning is None
    assert record.reasoning_prompt is None
    assert record.response.text == "yes"


def test_run_causal_cot_experiment_writes_schema_valid_records_with_reasoning_prompt(tmp_path):
    questions = [
        BenchmarkQuestion(question="Q1?", answer="yes", question_id="q1", given_info="Info.")
    ]
    output_path = tmp_path / "responses.jsonl"

    written = run_causal_cot_experiment(
        questions,
        FakeClient(),
        "{given_info} {question}",
        "{given_info} {question} {reasoning}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )

    assert written == 1
    record = ResponseRecord.model_validate(_read_records(output_path)[0])
    assert record.prompt_strategy == "causal_cot"
    assert record.reasoning is not None
    assert record.reasoning.text == "yes"
    # The step-1 prompt must be preserved, not discarded.
    assert record.reasoning_prompt == "Info. Q1?"
    assert record.response.text == "yes"


def test_run_experiment_is_resumable_by_request_id(tmp_path):
    questions = [BenchmarkQuestion(question="Q1?", answer="yes", question_id="q1")]
    output_path = tmp_path / "responses.jsonl"

    run_experiment(
        questions,
        FakeClient(),
        "{question}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )
    second_client = FakeClient()
    second_written = run_experiment(
        questions,
        second_client,
        "{question}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )

    assert second_written == 0
    assert len(_read_records(output_path)) == 1
    # Resuming must skip already-answered questions before ever calling the
    # model, not just dedup at write time - a large resumed run must not
    # re-query every previously-completed question.
    assert second_client.prompts == []


def test_run_causal_cot_experiment_skips_answered_questions_without_calling_client(tmp_path):
    questions = [
        BenchmarkQuestion(question="Q1?", answer="yes", question_id="q1", given_info="Info."),
        BenchmarkQuestion(question="Q2?", answer="yes", question_id="q2", given_info="Info."),
    ]
    output_path = tmp_path / "responses.jsonl"

    run_causal_cot_experiment(
        [questions[0]],
        FakeClient(),
        "{given_info} {question}",
        "{given_info} {question} {reasoning}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )
    second_client = FakeClient()
    second_written = run_causal_cot_experiment(
        questions,
        second_client,
        "{given_info} {question}",
        "{given_info} {question} {reasoning}",
        "test-model",
        "test-provider",
        str(output_path),
        "run-1",
    )

    assert second_written == 1
    assert len(_read_records(output_path)) == 2
    # Only q2's two calls (reasoning + final) should happen - q1 must be
    # skipped entirely, not silently re-queried and discarded at write time.
    assert len(second_client.prompts) == 2
