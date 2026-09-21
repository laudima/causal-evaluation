"""Provider protocol and response schema."""

from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    """Raw provider response retained for auditability."""

    text: str = ""
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str] = {}


class LLMClient(Protocol):
    """Minimal interface implemented by every provider."""

    def complete(self, prompt: str) -> ModelResponse: ...
