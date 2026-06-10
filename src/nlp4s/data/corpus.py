"""Assemble the training corpus and report the MHC-vs-training language gap.

The coverage report identifies which languages need synthetic augmentation.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from nlp4s.functionalities import MHC_LANGUAGES
from nlp4s.schema import Example, validate


def _dedup_key(example: Example) -> tuple[str, str]:
    """Stable key for near-duplicate detection: (language, normalised text)."""
    text = unicodedata.normalize("NFKC", example.text).strip().lower()
    text = " ".join(text.split())  # collapse whitespace
    return (example.language, text)


def assemble_corpus(sources: Iterable[Iterable[Example]]) -> list[Example]:
    """Concatenate and deduplicate training examples from multiple sources.

    Sources are merged in the order given; the first occurrence of each
    ``(language, normalised-text)`` key wins. Every record is validated against
    the frozen schema before inclusion, so malformed inputs fail fast.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Example] = []
    for source in sources:
        for example in source:
            validate(example)
            key = _dedup_key(example)
            if key in seen:
                continue
            seen.add(key)
            out.append(example)
    return out


def coverage_report(corpus: Iterable[Example]) -> dict[str, int]:
    """Count training examples per language, across all MHC languages.

    Returns a mapping ISO 639-1 code → example count. Every MHC language is
    present as a key (defaulting to 0), so the result directly highlights the
    coverage gap to be closed by synthetic generation.
    """
    counts: dict[str, int] = {lang: 0 for lang in MHC_LANGUAGES}
    for example in corpus:
        if example.language in counts:
            counts[example.language] += 1
        else:
            counts[example.language] = counts.get(example.language, 0) + 1
    return counts


def missing_languages(corpus: Iterable[Example]) -> list[str]:
    """MHC languages with zero training examples in ``corpus``.

    These are the targets for synthetic implicit-multilingual generation.
    Order matches ``MHC_LANGUAGES`` for stable downstream behaviour.
    """
    counts = coverage_report(corpus)
    return [lang for lang in MHC_LANGUAGES if counts.get(lang, 0) == 0]
