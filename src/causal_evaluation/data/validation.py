"""Validated benchmark schemas."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BenchmarkQuestion(BaseModel):
    """A normalized CLadder question and its expected answer."""

    model_config = ConfigDict(extra="allow")

    question_id: str | None = None
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    given_info: str = ""
    causal_level: str = "unknown"
    question_type: str = "unknown"
    topology: str = "unknown"

    @field_validator("question", "answer", mode="before")
    @classmethod
    def require_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value.strip()

    @field_validator("question_id", mode="before")
    @classmethod
    def coerce_question_id(cls, value: Any) -> str | None:
        return None if value is None else str(value)

    @model_validator(mode="before")
    @classmethod
    def derive_from_meta(cls, data: Any) -> Any:
        """Pull causal_level/question_type/topology out of the CLadder "meta" block."""
        if not isinstance(data, dict):
            return data
        meta = data.get("meta")
        if not isinstance(meta, dict):
            return data
        data = dict(data)
        if meta.get("rung") is not None:
            data["causal_level"] = str(meta["rung"])
        if meta.get("query_type") is not None:
            data["question_type"] = meta["query_type"]
        if meta.get("graph_id") is not None:
            data["topology"] = meta["graph_id"]
        return data

    def stable_id(self) -> str:
        """Return the supplied stable identifier or a deterministic content ID."""
        if self.question_id:
            return self.question_id
        import hashlib

        return hashlib.sha256(self.question.encode()).hexdigest()[:16]
