"""Load Multilingual HateCheck (evaluation benchmark) into the frozen schema.

MHC is evaluation-only (no training split). This module loads it and maps each
row's functionality label into the implicit/explicit/control grouping.

Role A. See docs/assignment.md "Dataset".
"""

from __future__ import annotations

from typing import Any

from nlp4s.schema import Example


def load_mhc(hf_dataset: str, cache_dir: str | None = None) -> list[Example]:
    """Load the MHC test suite as Example records (split="test").

    Args:
        hf_dataset: HuggingFace dataset id (see configs/data.yaml).
        cache_dir: optional local cache directory.

    Returns:
        One Example per MHC case, with ``functionality`` populated.

    TODO: load via ``datasets.load_dataset``, map MHC columns
    (test_case/label_gold/functionality/lang) onto Example fields, and keep
    only the functionalities in EXPLICIT | IMPLICIT | CONTROL.
    """
    raise NotImplementedError("TODO(Role A): implement MHC loading")


def to_example(row: dict[str, Any]) -> Example:
    """Map a single raw MHC row to an Example.

    TODO(Role A): translate MHC's gold-label encoding to schema.LABELS and
    normalise the language code to ISO 639-1.
    """
    raise NotImplementedError("TODO(Role A): implement MHC row mapping")
