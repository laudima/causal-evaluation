import json

import pytest

from causal_evaluation.evaluation.variant_analysis import load_scored
from causal_evaluation.models.logprob import (
    COT_FINAL_QUESTION,
    NO_VARIANTS,
    YES_VARIANTS,
    answer_token_ids,
    cot_messages,
    done_item_ids,
    item_text,
    question_only_text,
    render_messages,
    render_prompt,
)

ITEM = {"background": "X causes Y.", "given_info": "P=0.4.", "question": "Does X?"}


class FakeTokenizer:
    chat_template = None
    vocab = {"Yes": 1, " Yes": 2, "yes": 3, "No": 4, " no": 5}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [90, 91]


def test_item_text_includes_graph_data_and_instruction():
    text = item_text({"background": "X causes Y.", "given_info": "P=0.4.", "question": "Does X?"})
    assert text.splitlines() == ["X causes Y.", "P=0.4.", "Does X?", "Answer with Yes or No only."]


def test_render_prompt_without_chat_template_adds_special_tokens():
    prompt, add_special = render_prompt(FakeTokenizer(), "hello")
    assert prompt.endswith("Answer:") and add_special


def test_answer_token_ids_keeps_single_token_forms():
    tok = FakeTokenizer()
    assert answer_token_ids(tok, YES_VARIANTS) == [1, 2, 3]
    assert answer_token_ids(tok, NO_VARIANTS) == [4, 5]
    with pytest.raises(ValueError):
        answer_token_ids(tok, ("maybe",))


def test_cot_messages_two_turns():
    first = cot_messages(ITEM)
    assert len(first) == 1
    assert first[0]["content"].startswith("X causes Y.\nP=0.4.\nDoes X?\n\nGuidance:")
    assert "Step 6) Calculate the estimand" in first[0]["content"]
    full = cot_messages(ITEM, "reasoning text")
    assert [m["role"] for m in full] == ["user", "assistant", "user"]
    assert full[1]["content"] == "reasoning text"
    assert full[2]["content"] == COT_FINAL_QUESTION


def test_render_messages_fallback():
    prompt, add_special = render_messages(FakeTokenizer(), cot_messages(ITEM, "r"))
    assert prompt.startswith("User: X causes Y.") and prompt.endswith("Assistant:")
    assert add_special


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_done_ids_and_labels_separate_strategies(tmp_path):
    results = tmp_path / "r.jsonl"
    _write(
        results,
        [
            {"item_id": "a", "model": "m", "p_yes": 0.9},
            {"item_id": "a", "model": "m", "prompt_strategy": "causal_cot", "p_yes": 0.2},
        ],
    )
    assert done_item_ids(results, "m") == {"a"}
    assert done_item_ids(results, "m", "causal_cot") == {"a"}
    assert done_item_ids(results, "m", "other") == set()
    sample = tmp_path / "s.jsonl"
    _write(sample, [{"item_id": "a", "answer": "yes"}])
    df = load_scored(sample, [results]).set_index("model")
    assert df.loc["m", "correct"]
    assert not df.loc["m-causal_cot", "correct"]


def test_higher_token_limit_redoes_only_truncated(tmp_path):
    results = tmp_path / "cot.jsonl"
    base = {"model": "m", "prompt_strategy": "causal_cot", "max_new_tokens": 512}
    _write(
        results,
        [
            {**base, "item_id": "finished", "reasoning_truncated": False},
            {**base, "item_id": "cut", "reasoning_truncated": True},
        ],
    )
    assert done_item_ids(results, "m", "causal_cot", 512) == {"finished", "cut"}
    assert done_item_ids(results, "m", "causal_cot", 1024) == {"finished"}
    with results.open("a") as handle:
        handle.write(
            json.dumps(
                {**base, "item_id": "cut", "max_new_tokens": 1024, "reasoning_truncated": True}
            )
            + "\n"
        )
    assert done_item_ids(results, "m", "causal_cot", 1024) == {"finished", "cut"}


def test_load_scored_keeps_latest_row_per_item(tmp_path):
    results = tmp_path / "cot.jsonl"
    base = {"item_id": "a", "model": "m", "prompt_strategy": "causal_cot"}
    _write(
        results,
        [
            {**base, "p_yes": 0.1, "max_new_tokens": 512},
            {**base, "p_yes": 0.9, "max_new_tokens": 1024},
        ],
    )
    sample = tmp_path / "s.jsonl"
    _write(sample, [{"item_id": "a", "answer": "yes"}])
    df = load_scored(sample, [results])
    assert len(df) == 1 and df["max_new_tokens"].iloc[0] == 1024 and df["correct"].iloc[0]


def test_question_only_text_drops_graph_and_data():
    assert question_only_text(ITEM).splitlines() == ["Does X?", "Answer with Yes or No only."]
