"""Quality filtering and deduplication for synthetically generated examples.

Role A.
"""

from __future__ import annotations

from nlp4s.schema import Example


def deduplicate(examples: list[Example]) -> list[Example]:
    """Remove near-duplicate generated examples.

    TODO(Role A): exact + near-duplicate removal (e.g. normalised text or
    embedding similarity threshold).
    """
    raise NotImplementedError("TODO(Role A): implement deduplication")


def quality_filter(examples: list[Example]) -> list[Example]:
    """Drop low-quality / off-target generations.

    TODO(Role A): filter by language-id check, length bounds, and a
    label/functionality consistency heuristic.
    """
    raise NotImplementedError("TODO(Role A): implement quality filtering")
