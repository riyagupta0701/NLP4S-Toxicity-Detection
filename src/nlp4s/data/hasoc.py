"""Load and normalise HASOC training data into the frozen schema.

HASOC is a coarse binary HOF/NOT task; only English/German/Hindi overlap with
the MHC evaluation languages, and there is no implicit/explicit annotation
(``functionality`` is left empty).

Role A. See docs/assignment.md "Dataset language coverage".
"""

from __future__ import annotations

from nlp4s.schema import Example


def load_hasoc(root: str, languages: list[str]) -> list[Example]:
    """Load HASOC editions for the requested languages as Example records.

    Args:
        root: directory containing the HASOC files.
        languages: ISO 639-1 codes to load (typically the MHC overlap: en/de/hi).

    Returns:
        Example records with split="train" and functionality="".

    TODO(Role A): parse the per-edition TSV files, map the HOF/NOT label onto
    schema.LABELS ("hateful"/"non-hateful"), and tag the language.
    """
    raise NotImplementedError("TODO(Role A): implement HASOC loading")
