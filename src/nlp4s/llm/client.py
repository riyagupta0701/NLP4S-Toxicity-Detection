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
    """Aya backend via the Cohere API.

    TODO(Role C): construct a cohere client from COHERE_API_KEY in ``__init__``
    and implement ``complete`` against the chat endpoint with ``model_id``.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # TODO(Role C): initialise cohere.Client(api_key=os.environ["COHERE_API_KEY"]).

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        raise NotImplementedError("TODO(Role C): implement Cohere/Aya completion")


class OpenAICompatibleClient(LLMClient):
    """Mistral/Llama backend via an OpenAI-compatible chat endpoint.

    TODO(Role C): construct an openai client pointed at OPENAI_COMPATIBLE_BASE_URL
    with OPENAI_COMPATIBLE_API_KEY, and implement ``complete`` via chat.completions.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # TODO(Role C): initialise openai.OpenAI(base_url=..., api_key=...).

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        raise NotImplementedError("TODO(Role C): implement OpenAI-compatible completion")


def build_client(provider: str, model_id: str) -> LLMClient:
    """Factory: map a config ``provider`` string to a client instance."""
    if provider == "cohere":
        return CohereClient(model_id)
    if provider == "openai_compatible":
        return OpenAICompatibleClient(model_id)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
