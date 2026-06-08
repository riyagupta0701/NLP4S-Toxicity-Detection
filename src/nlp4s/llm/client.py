"""LLM client abstraction over hosted APIs (Role C).

A single ``LLMClient`` interface with two backends:
- ``CohereClient``     — Aya models (uses COHERE_API_KEY).
- ``OpenAICompatibleClient`` — Mistral/Llama via any OpenAI-compatible endpoint
  (uses OPENAI_COMPATIBLE_API_KEY / OPENAI_COMPATIBLE_BASE_URL).

Keys are read from the environment (load a .env via python-dotenv at startup).
"""

from __future__ import annotations

import abc


class LLMClient(abc.ABC):
    """Minimal text-completion interface used by classification and generation."""

    @abc.abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Return the model's text completion for ``prompt``."""
        raise NotImplementedError


class CohereClient(LLMClient):
    """Aya backend via the Cohere API (Chat v2 endpoint).

    Minimal Role-A implementation so `nlp4s generate` is runnable end-to-end
    against real Aya before Role C lands their multi-backend version. Role C
    is free to rewrite this — the public surface is just ``complete``.
    """

    def __init__(self, model_id: str) -> None:
        import os

        import cohere  # local import: heavy dep

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set; copy .env.example to .env and fill it in"
            )
        self.model_id = model_id
        self._client = cohere.ClientV2(api_key=api_key)

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self._client.chat(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # ClientV2 returns response.message.content as a list of content blocks.
        content = response.message.content or []
        return "".join(getattr(block, "text", "") for block in content)


class OpenAICompatibleClient(LLMClient):
    """Mistral/Llama backend via an OpenAI-compatible chat endpoint.

    Targets a local server (vLLM / Ollama / llama.cpp / LM Studio) by default —
    set OPENAI_COMPATIBLE_BASE_URL to e.g. ``http://localhost:8000/v1`` and
    OPENAI_COMPATIBLE_API_KEY to any non-empty value (the openai SDK requires
    the field; local servers ignore it).
    """

    def __init__(self, model_id: str) -> None:
        import os

        from openai import OpenAI  # local import: heavy dep

        base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or "local"
        if not base_url:
            raise RuntimeError(
                "OPENAI_COMPATIBLE_BASE_URL is not set; point it at your local "
                "OpenAI-compatible server (e.g. http://localhost:8000/v1)."
            )
        self.model_id = model_id
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return choice.message.content or ""


def build_client(provider: str, model_id: str) -> LLMClient:
    """Factory: map a config ``provider`` string to a client instance."""
    if provider == "cohere":
        return CohereClient(model_id)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(model_id)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
