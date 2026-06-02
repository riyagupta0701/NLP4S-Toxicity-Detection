"""Hold out a portion of MHC as the few-shot demonstration pool (Role C).

MHC is the evaluation benchmark, so we cannot use the same rows for both
prompting demonstrations and scoring. ``split_mhc`` partitions MHC
deterministically per (language x functionality) cell into a small ``pool``
slice (demos only) and a large ``eval`` slice (the test set). ``subsample_eval``
then balances the eval slice down to a budget the API can afford.

The split is keyed on Example.id so re-runs across processes are stable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from nlp4s.schema import Example


def _cell_key(e: Example) -> tuple[str, str]:
    return (e.language, e.functionality)


def split_mhc(
    examples: Iterable[Example],
    *,
    per_cell_pool_size: int = 5,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Stratified split into (eval, pool).

    For each (language, functionality) cell, draw ``per_cell_pool_size`` rows
    into the pool (or all rows if the cell is smaller) and keep the rest as
    eval. Stable across runs because we sort by id and use a seeded RNG.

    Returns ``(eval_examples, pool_examples)`` — eval first because it's the
    bigger, more-often-consumed slice.
    """
    by_cell: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for e in examples:
        by_cell[_cell_key(e)].append(e)

    rng = random.Random(seed)
    eval_out: list[Example] = []
    pool_out: list[Example] = []
    for cell, items in sorted(by_cell.items()):
        items_sorted = sorted(items, key=lambda e: e.id or "")
        rng.shuffle(items_sorted)
        take = min(per_cell_pool_size, len(items_sorted))
        pool_out.extend(_retag(items_sorted[:take], "pool"))
        eval_out.extend(items_sorted[take:])
    return eval_out, pool_out


def subsample_eval(
    eval_examples: Iterable[Example],
    *,
    per_cell: int,
    seed: int = 42,
) -> list[Example]:
    """Cap each (language, functionality) cell at ``per_cell`` examples.

    ``per_cell <= 0`` returns the input unchanged (no subsampling).
    """
    eval_list = list(eval_examples)
    if per_cell <= 0:
        return eval_list
    by_cell: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for e in eval_list:
        by_cell[_cell_key(e)].append(e)
    rng = random.Random(seed)
    out: list[Example] = []
    for cell, items in sorted(by_cell.items()):
        items_sorted = sorted(items, key=lambda e: e.id or "")
        rng.shuffle(items_sorted)
        out.extend(items_sorted[:per_cell])
    return out


def _retag(examples: list[Example], split: str) -> list[Example]:
    return [
        Example(
            text=e.text,
            language=e.language,
            label=e.label,
            functionality=e.functionality,
            split=split,
            id=e.id,
            target=e.target,
        )
        for e in examples
    ]
