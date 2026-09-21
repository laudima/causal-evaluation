"""Built-in provider implementations."""

import os

from openai import OpenAI

from .base import ModelResponse


class DryRunClient:
    """Provider that returns the benchmark answer embedded in test metadata."""

    def complete(self, prompt: str) -> ModelResponse:
        marker = "[DRY_RUN_ANSWER="
        if marker not in prompt:
            return ModelResponse(text="unknown", metadata={"provider": "dry-run"})
        answer = prompt.split(marker, 1)[1].split("]", 1)[0]
        return ModelResponse(text=answer, metadata={"provider": "dry-run"})


class OpenAICompatibleClient:
    """Provider for OpenAI and any OpenAI-compatible API (e.g. DeepSeek)."""

    def __init__(
        self,
        model: str,
        provider_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._model = model
        self._provider_name = provider_name
        self._client = OpenAI(api_key=api_key or os.environ["LLM_API_KEY"], base_url=base_url)

    def complete(self, prompt: str) -> ModelResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            return ModelResponse(
                text=response.choices[0].message.content or "",
                metadata={"provider": self._provider_name, "model": self._model},
            )
        except Exception as exc:
            return ModelResponse(
                error=str(exc),
                metadata={"provider": self._provider_name, "model": self._model},
            )


def build_client(provider: str, model: str = ""):
    """Build a configured client; external providers can be registered here."""
    if provider == "dry-run":
        return DryRunClient()
    if provider == "openai":
        return OpenAICompatibleClient(model=model, provider_name="openai")
    if provider == "deepseek":
        return OpenAICompatibleClient(
            model=model,
            provider_name="deepseek",
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        )
    if provider == "docker":
        # Docker Model Runner serves local models over an OpenAI-compatible API
        # and ignores the API key, so no credential is read from the environment.
        return OpenAICompatibleClient(
            model=model,
            provider_name="docker",
            base_url=os.environ.get("DOCKER_MODEL_BASE_URL", "http://localhost:12434/engines/v1"),
            api_key="docker-model-runner",
        )
    raise ValueError(f"Unsupported provider: {provider}")
