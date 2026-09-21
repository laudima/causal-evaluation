"""Validated schema for a stored experiment response record."""

from datetime import datetime

from pydantic import BaseModel

from ..models.base import ModelResponse


class ResponseRecord(BaseModel):
    """One resumable-storage row in responses.jsonl.

    `provider` and `prompt_strategy` are explicit top-level fields (rather
    than left buried in `response.metadata` or inferred from which optional
    keys happen to be present) so responses accumulated across many
    model/provider/prompt-strategy runs can be grouped and compared later
    without re-parsing nested, provider-defined metadata.
    """

    request_id: str
    run_id: str
    question_id: str
    model: str
    provider: str
    prompt_strategy: str
    prompt: str
    reasoning_prompt: str | None = None
    reasoning: ModelResponse | None = None
    response: ModelResponse
    created_at: datetime
