"""Few-shot demonstration selection strategies (RQ3.4).

Compares random selection against selective retrieval (BM25 + target-group,
adaptive threshold, in the spirit of ARIIHA). Role C.
"""

from __future__ import annotations

from nlp4s.schema import Example


def select_random(pool: list[Example], k: int, seed: int = 42) -> list[Example]:
    """Pick ``k`` demonstrations uniformly at random from ``pool``.

    TODO(Role C): seeded random sampling.
    """
    raise NotImplementedError("TODO(Role C): implement random selection")


def select_bm25(pool: list[Example], query: str, k: int) -> list[Example]:
    """Pick the top-``k`` demonstrations by BM25 similarity to ``query``.

    TODO(Role C): build a rank_bm25 index over the pool's texts and retrieve.
    """
    raise NotImplementedError("TODO(Role C): implement BM25 selection")


def select_target_group(pool: list[Example], query: str, k: int) -> list[Example]:
    """Target-prioritised selection (exact/similar target group, then similarity).

    TODO(Role C): implement the target-group hierarchy with an adaptive
    similarity threshold fallback to BM25.
    """
    raise NotImplementedError("TODO(Role C): implement target-group selection")


def select(strategy: str, pool: list[Example], query: str, k: int) -> list[Example]:
    """Dispatch to a selection strategy by name ("random" | "bm25" | "target_group")."""
    if k <= 0 or not pool:
        return []
    if strategy == "random":
        return select_random(pool, k)
    if strategy == "bm25":
        return select_bm25(pool, query, k)
    if strategy == "target_group":
        return select_target_group(pool, query, k)
    raise ValueError(f"Unknown few-shot selection strategy: {strategy!r}")
