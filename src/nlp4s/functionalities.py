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

# Synthetic-only tag for ToxiGen-style neutral mentions of minority groups
# produced by the synthetic generator. Not an MHC functionality — no MHC category
# fits this cleanly — but downstream filters / consumers accept it.
SYNTHETIC: frozenset[str] = frozenset({"neutral_mention_nh"})

Group = str  # one of: "explicit", "implicit", "control", "synthetic"


def group_of(functionality: str) -> Group:
    """Map a functionality label to its category.

    Returns one of ``"explicit"``, ``"implicit"``, ``"control"``, or
    ``"synthetic"`` (the last covers synthetic-only tags like
    ``neutral_mention_nh``).

    Raises:
        KeyError: if the functionality is not known.
    """
    if functionality in EXPLICIT:
        return "explicit"
    if functionality in IMPLICIT:
        return "implicit"
    if functionality in CONTROL:
        return "control"
    if functionality in SYNTHETIC:
        return "synthetic"
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
