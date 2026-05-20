"""MHC functionality groupings and language constants (shared contract).

Operationalises the implicit-vs-explicit distinction via Multilingual HateCheck
functionality labels, exactly as defined in docs/assignment.md.
"""

from __future__ import annotations

# --- Functionality groupings (the operational implicit/explicit definition) ---

EXPLICIT: frozenset[str] = frozenset(
    {"derog_neg_attrib_h", "derog_dehum_h", "slur_h", "profanity_h"}
)
IMPLICIT: frozenset[str] = frozenset({"derog_impl_h"})
CONTROL: frozenset[str] = frozenset({"profanity_nh"})

Group = str  # one of: "explicit", "implicit", "control"


def group_of(functionality: str) -> Group:
    """Map an MHC functionality label to its category.

    Returns one of ``"explicit"``, ``"implicit"``, or ``"control"``.

    Raises:
        KeyError: if the functionality is not part of the studied subset.
    """
    if functionality in EXPLICIT:
        return "explicit"
    if functionality in IMPLICIT:
        return "implicit"
    if functionality in CONTROL:
        return "control"
    raise KeyError(f"Unknown / out-of-scope functionality: {functionality!r}")


# --- Languages ---

# All 11 MHC evaluation languages (ISO 639-1), English included.
MHC_LANGUAGES: tuple[str, ...] = (
    "ar", "nl", "fr", "de", "hi", "it", "zh", "pl", "pt", "es", "en",
)

# MHC languages for which HASOC provides training data (the coverage gap: 3/11).
HASOC_OVERLAP: frozenset[str] = frozenset({"en", "de", "hi"})


def languages_without_training_data() -> tuple[str, ...]:
    """MHC languages with no HASOC training coverage — synthetic-generation targets."""
    return tuple(lang for lang in MHC_LANGUAGES if lang not in HASOC_OVERLAP)
