"""Classification metrics broken down by functionality and language (RQ1/RQ3).

Role D.
"""

from __future__ import annotations

from nlp4s.schema import Example, Prediction


def score(
    examples: list[Example],
    predictions: list[Prediction],
    group_by: list[str],
) -> dict[tuple[str, ...], dict[str, float]]:
    """Compute F1/precision/recall grouped by the given axes.

    Args:
        examples: gold Example records (carry functionality + language).
        predictions: model Prediction records (joined on example id).
        group_by: axes to break down by, e.g. ["functionality", "language"].

    Returns:
        Mapping from a group key (tuple of axis values) to a metrics dict
        with keys "f1", "precision", "recall", "support".

    TODO(Role D): join predictions to examples by id, group, and compute
    per-group metrics with sklearn (binary, positive class = "hateful").
    """
    raise NotImplementedError("TODO(Role D): implement grouped metrics")


def implicit_explicit_gap(
    grouped: dict[tuple[str, ...], dict[str, float]],
) -> dict[str, float]:
    """Summarise the implicit-vs-explicit performance gap per language.

    TODO(Role D): aggregate grouped metrics into explicit vs implicit means
    (via functionalities.group_of) and return the gap.
    """
    raise NotImplementedError("TODO(Role D): implement implicit/explicit gap")
