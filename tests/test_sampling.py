from causal_evaluation.data.sampling import sample_questions
from causal_evaluation.data.validation import BenchmarkQuestion


def test_sampling_is_reproducible_and_stratified():
    questions = [
        BenchmarkQuestion(question=f"q{i}", answer="yes", causal_level=str(i % 2))
        for i in range(10)
    ]
    first = [q.stable_id() for q in sample_questions(questions, 4, 7)]
    second = [q.stable_id() for q in sample_questions(questions, 4, 7)]
    assert first == second
    assert {q.causal_level for q in sample_questions(questions, 4, 7)} == {"0", "1"}
