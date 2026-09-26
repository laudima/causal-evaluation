"""Built-in provider implementations."""

import os

from openai import OpenAI

from .base import ModelResponse

# A request that never times out can hang a `run` indefinitely - e.g. a stale
# TCP connection left over from a Docker Model Runner restart. Every
# OpenAI-compatible client gets this ceiling unless overridden.
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))


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
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._model = model
        self._provider_name = provider_name
        self._client = OpenAI(
            api_key=api_key or os.environ["LLM_API_KEY"], base_url=base_url, timeout=timeout
        )

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
        # 127.0.0.1 (not "localhost") avoids httpx blocking on a dead IPv6
        # (::1) connection attempt before falling back to IPv4 - observed
        # adding ~60s of dead time per request that curl's resolver doesn't
        # hit against the same host.
        return OpenAICompatibleClient(
            model=model,
            provider_name="docker",
            base_url=os.environ.get("DOCKER_MODEL_BASE_URL", "http://127.0.0.1:12434/engines/v1"),
            api_key="docker-model-runner",
        )
    raise ValueError(f"Unsupported provider: {provider}")
