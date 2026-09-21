import json

import pytest
from pydantic import ValidationError

from causal_evaluation.experiments.schema import ResponseRecord
from causal_evaluation.models.base import ModelResponse


def _base_kwargs(**overrides):
    kwargs = {
        "request_id": "req-1",
        "run_id": "run-1",
        "question_id": "q1",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "prompt_strategy": "default",
        "prompt": "Question?",
        "response": ModelResponse(text="yes"),
        "created_at": "2026-09-19T00:00:00Z",
    }
    kwargs.update(overrides)
    return kwargs


def test_response_record_requires_core_fields():
    with pytest.raises(ValidationError):
        ResponseRecord(request_id="req-1")


def test_response_record_defaults_reasoning_and_reasoning_prompt_to_none():
    record = ResponseRecord(**_base_kwargs())
    assert record.reasoning is None
    assert record.reasoning_prompt is None


def test_response_record_accepts_causal_cot_reasoning():
    record = ResponseRecord(
        **_base_kwargs(
            prompt_strategy="causal_cot",
            reasoning_prompt="step 1 prompt",
            reasoning=ModelResponse(text="reasoning text"),
        )
    )
    assert record.reasoning_prompt == "step 1 prompt"
    assert record.reasoning.text == "reasoning text"


def test_response_record_round_trips_through_json_dump():
    record = ResponseRecord(**_base_kwargs())
    dumped = record.model_dump(mode="json")
    # Must be plain-JSON-serializable, as required by storage.append_unique.
    line = json.dumps(dumped, sort_keys=True)
    reloaded = ResponseRecord.model_validate(json.loads(line))
    assert reloaded == record
