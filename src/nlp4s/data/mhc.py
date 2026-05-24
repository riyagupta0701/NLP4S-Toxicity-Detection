"""Load Multilingual HateCheck (evaluation benchmark) into the frozen schema.

MHC is evaluation-only (no training split). This module loads it and keeps only
rows whose functionality falls in the studied implicit/explicit/control subset
(see ``nlp4s.functionalities``).

Source columns on HuggingFace (``mteb/multi-hatecheck``):
    ``functionality``  — MHC functionality label, e.g. ``derog_impl_h``.
    ``text``           — the test-case sentence.
    ``is_hateful``     — string label, ``"hateful"`` / ``"non-hateful"``.
    ``lang``           — ISO 639-3 code (e.g. ``"ara"``, ``"cmn"``).

Role A. See docs/assignment.md "Dataset".
"""

from __future__ import annotations

from typing import Any, Iterable

from nlp4s.functionalities import CONTROL, EXPLICIT, IMPLICIT, MHC_LANGUAGES
from nlp4s.schema import LABELS, Example

# Functionalities kept in the final set (Role A column filter).
KEEP_FUNCTIONALITIES: frozenset[str] = EXPLICIT | IMPLICIT | CONTROL

# MHC ships ISO 639-3 codes; the frozen schema uses ISO 639-1.
_LANG_639_3_TO_1: dict[str, str] = {
    "ara": "ar",
    "cmn": "zh",
    "deu": "de",
    "eng": "en",
    "fra": "fr",
    "hin": "hi",
    "ita": "it",
    "nld": "nl",
    "pol": "pl",
    "por": "pt",
    "spa": "es",
}


def _normalise_language(code: str) -> str:
    """Map an MHC ``lang`` value to ISO 639-1; pass through if already 639-1."""
    if code in _LANG_639_3_TO_1:
        return _LANG_639_3_TO_1[code]
    if code in MHC_LANGUAGES:
        return code
    raise ValueError(f"Unknown MHC language code: {code!r}")


def _normalise_label(value: Any) -> str:
    """Map MHC ``is_hateful`` (string or 0/1) onto schema.LABELS."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in LABELS:
            return v
        if v in {"hate", "hateful_h"}:
            return "hateful"
        if v in {"non_hateful", "nonhateful", "not_hateful"}:
            return "non-hateful"
    if isinstance(value, bool):
        return "hateful" if value else "non-hateful"
    if isinstance(value, (int, float)):
        return "hateful" if int(value) == 1 else "non-hateful"
    raise ValueError(f"Cannot interpret is_hateful={value!r}")


def to_example(row: dict[str, Any], *, split: str = "test", idx: int | None = None) -> Example:
    """Map a single raw MHC row to an Example.

    Translates ``is_hateful`` to ``schema.LABELS`` and normalises ``lang`` from
    ISO 639-3 to ISO 639-1. Caller is responsible for filtering on functionality.
    """
    language = _normalise_language(row["lang"])
    label = _normalise_label(row["is_hateful"])
    functionality = row["functionality"]
    text = row["text"]
    example_id = row.get("id")
    if example_id is None and idx is not None:
        example_id = f"mhc-{language}-{idx}"
    return Example(
        text=text,
        language=language,
        label=label,
        functionality=functionality,
        split=split,
        id=example_id,
    )


def _iter_rows(hf_dataset: str, cache_dir: str | None) -> Iterable[dict[str, Any]]:
    """Yield raw MHC rows from HuggingFace ``datasets``."""
    from datasets import load_dataset  # local import: heavy dep

    ds = load_dataset(hf_dataset, cache_dir=cache_dir)
    split_name = "test" if "test" in ds else next(iter(ds.keys()))
    for row in ds[split_name]:
        yield row


def load_mhc(hf_dataset: str, cache_dir: str | None = None) -> list[Example]:
    """Load the MHC test suite as Example records (split="test").

    Keeps only rows whose ``functionality`` is in the studied subset
    (EXPLICIT ∪ IMPLICIT ∪ CONTROL — i.e. ``derog_neg_attrib_h``,
    ``derog_dehum_h``, ``derog_impl_h``, ``slur_h``, ``profanity_h``,
    ``profanity_nh``). All other functionalities are dropped.
    """
    examples: list[Example] = []
    for idx, row in enumerate(_iter_rows(hf_dataset, cache_dir)):
        if row.get("functionality") not in KEEP_FUNCTIONALITIES:
            continue
        examples.append(to_example(row, split="test", idx=idx))
    return examples
