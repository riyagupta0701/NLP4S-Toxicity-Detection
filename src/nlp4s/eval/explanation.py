"""Reference-free explanation-quality scoring and its correlation with
classification correctness (RQ2.3).

Role D.
"""

from __future__ import annotations

from nlp4s.schema import Prediction


def score_explanations(predictions: list[Prediction], metric: str) -> dict[str, float]:
    """Score generated rationales with a reference-free metric.

    Args:
        predictions: predictions whose ``rationale`` is set (explanation condition).
        metric: "bertscore" or "g_eval".

    Returns:
        Mapping example_id -> quality score.

    TODO(Role D): implement BERTScore (self/source-based, reference-free variant)
    and a G-Eval LLM-judge option.
    """
    raise NotImplementedError("TODO(Role D): implement explanation scoring")


def correlate_with_correctness(
    quality: dict[str, float],
    correctness: dict[str, bool],
) -> float:
    """Correlation between explanation quality and whether the label was correct.

    TODO(Role D): align by example_id and compute a correlation coefficient.
    """
    raise NotImplementedError("TODO(Role D): implement correlation analysis")
