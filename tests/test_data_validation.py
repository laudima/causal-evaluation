import pytest
from pydantic import ValidationError

from causal_evaluation.data.loader import load_questions
from causal_evaluation.data.validation import BenchmarkQuestion


def test_malformed_question_rejected():
    with pytest.raises(ValidationError):
        BenchmarkQuestion(question="", answer="yes")


def test_duplicate_questions_rejected(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"same","answer":"yes"}\n{"question":"same","answer":"no"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(path)
