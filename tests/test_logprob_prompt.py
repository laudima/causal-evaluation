import pytest

from causal_evaluation.models.logprob import (
    NO_VARIANTS,
    YES_VARIANTS,
    answer_token_ids,
    item_text,
    render_prompt,
)


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
