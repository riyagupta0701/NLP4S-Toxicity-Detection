"""Cross-model and cross-language analysis (RQ3.2/3.3).

Role D.
"""

from __future__ import annotations

from nlp4s.schema import Prediction


def explanation_effect(
    no_expl: dict[tuple[str, ...], dict[str, float]],
    with_expl: dict[tuple[str, ...], dict[str, float]],
) -> dict[tuple[str, ...], float]:
    """Per (model, functionality, language) delta from adding explanation.

    TODO(Role D): subtract no-explanation metrics from explanation metrics to
    quantify where explanation helps or hurts.
    """
    raise NotImplementedError("TODO(Role D): implement explanation-effect analysis")


def model_disagreement(predictions_by_model: dict[str, list[Prediction]]) -> dict[str, float]:
    """Where models diverge most across functionality-language combinations.

    TODO(Role D): compute pairwise/aggregate disagreement to surface the
    combinations flagged in RQ3.2.
    """
    raise NotImplementedError("TODO(Role D): implement model-disagreement analysis")
