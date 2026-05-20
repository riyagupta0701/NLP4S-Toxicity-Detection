"""ToxiGen-style synthetic generation of implicit hate / non-hate pairs.

Uses a multilingual LLM (Aya via the LLMClient) and demonstration-based
prompting, seeded from MHC functionality definitions, to produce implicit
examples in MHC languages that lack training data. ToxiGen is English-only, so
generation is the route to *implicit multilingual* coverage.

Role A. See docs/assignment.md "Synthetic implicit-multilingual data generation".
"""

from __future__ import annotations

from nlp4s.llm.client import LLMClient
from nlp4s.schema import Example


def generate_for_language(
    client: LLMClient,
    language: str,
    demonstrations: list[Example],
    n: int,
) -> list[Example]:
    """Generate ``n`` synthetic examples in ``language`` via demonstration prompting.

    Args:
        client: LLM backend (e.g. Aya through CohereClient).
        language: target ISO 639-1 code.
        demonstrations: seed examples conditioning the generation.
        n: number of examples to request.

    Returns:
        Example records with split="synthetic" and an appropriate functionality
        (e.g. "derog_impl_h" for implicit) / matched non-hate controls.

    TODO(Role A): build the demonstration prompt, call ``client.complete``,
    parse the output into Example records, and tag language/split/functionality.
    """
    raise NotImplementedError("TODO(Role A): implement synthetic generation")
