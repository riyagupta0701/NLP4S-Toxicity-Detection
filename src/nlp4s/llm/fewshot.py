"""Few-shot demonstration selection strategies (RQ3.4).

Compares random selection against selective retrieval. ``target_group`` uses
the ``Example.target`` field (MHC ``target_ident``); when the query lacks
a target it falls back to BM25 within the same language, then global BM25.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

from nlp4s.schema import Example

_STRATEGIES = ("random", "bm25", "target_group")


def select_random(pool: list[Example], k: int, seed: int = 42) -> list[Example]:
    """Pick ``k`` demonstrations uniformly at random from ``pool``."""
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    return rng.sample(pool, min(k, len(pool)))


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bm25_rank(pool: Sequence[Example], query: str) -> list[Example]:
    """Score the pool against ``query`` with BM25 and return it sorted desc."""
    from rank_bm25 import BM25Okapi  # local import: heavy dep

    tokenised = [_tokenise(e.text) for e in pool]
    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(_tokenise(query))
    ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)
    return [example for example, _ in ranked]


def select_bm25(pool: list[Example], query: str, k: int) -> list[Example]:
    """Pick the top-``k`` demonstrations by BM25 similarity to ``query``."""
    if k <= 0 or not pool:
        return []
    return _bm25_rank(pool, query)[:k]


def select_target_group(
    pool: list[Example],
    query: str,
    k: int,
    *,
    query_target: str | None = None,
    query_language: str | None = None,
) -> list[Example]:
    """Target-prioritised selection.

    Hierarchy:
      1. pool examples whose ``target`` matches ``query_target`` exactly,
         ordered by BM25 against the query text,
      2. then pool examples in ``query_language`` (any target),
         ordered by BM25,
      3. then the rest of the pool, ordered by BM25.

    When ``query_target`` is None, the first tier is skipped — this is the
    expected case at inference time, where the query is unlabeled. The
    behaviour is then "same-language BM25 with a global BM25 fallback", which
    is still a meaningful contrast against ``select_bm25`` (no language
    preference) and ``select_random``.
    """
    if k <= 0 or not pool:
        return []

    tier_target: list[Example] = []
    tier_lang: list[Example] = []
    tier_rest: list[Example] = []
    for e in pool:
        if query_target and e.target and e.target == query_target:
            tier_target.append(e)
        elif query_language and e.language == query_language:
            tier_lang.append(e)
        else:
            tier_rest.append(e)

    out: list[Example] = []
    for tier in (tier_target, tier_lang, tier_rest):
        if len(out) >= k:
            break
        if not tier:
            continue
        out.extend(_bm25_rank(tier, query)[: k - len(out)])
    return out[:k]


def select(
    strategy: str,
    pool: list[Example],
    query: str,
    k: int,
    *,
    query_target: str | None = None,
    query_language: str | None = None,
    seed: int = 42,
) -> list[Example]:
    """Dispatch to a selection strategy by name."""
    if k <= 0 or not pool:
        return []
    if strategy == "random":
        return select_random(pool, k, seed=seed)
    if strategy == "bm25":
        return select_bm25(pool, query, k)
    if strategy == "target_group":
        return select_target_group(
            pool, query, k, query_target=query_target, query_language=query_language
        )
    raise ValueError(
        f"Unknown few-shot selection strategy: {strategy!r}; expected one of {_STRATEGIES}"
    )
