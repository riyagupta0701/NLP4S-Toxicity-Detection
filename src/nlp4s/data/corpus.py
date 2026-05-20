"""Assemble the training corpus and report the MHC-vs-training language gap.

Role A. The coverage report feeds the Phase-2 baseline decision point
(which languages need synthetic augmentation). See docs/implementation_plan.md.
"""

from __future__ import annotations

from nlp4s.functionalities import MHC_LANGUAGES
from nlp4s.schema import Example


def assemble_corpus(sources: list[list[Example]]) -> list[Example]:
    """Concatenate and deduplicate training examples from multiple sources.

    TODO(Role A): merge sources (HASOC + optional synthetic), dedupe on text,
    and validate every record against the frozen schema.
    """
    raise NotImplementedError("TODO(Role A): implement corpus assembly")


def coverage_report(corpus: list[Example]) -> dict[str, int]:
    """Count training examples per language, across all MHC languages.

    Returns:
        Mapping ISO 639-1 code -> example count, with 0 for MHC languages that
        have no training data (the gap to be closed by synthetic generation).

    TODO(Role A): tally ``corpus`` by language; ensure all MHC_LANGUAGES appear
    as keys (defaulting to 0).
    """
    raise NotImplementedError("TODO(Role A): implement coverage report")


def missing_languages(corpus: list[Example]) -> list[str]:
    """MHC languages with zero training examples in ``corpus``.

    TODO(Role A): derive from coverage_report; these are the generation targets.
    """
    _ = MHC_LANGUAGES  # generation targets are drawn from here
    raise NotImplementedError("TODO(Role A): implement missing-language detection")
