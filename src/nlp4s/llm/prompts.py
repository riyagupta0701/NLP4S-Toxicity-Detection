"""Prompt templates for the no-explanation vs explanation conditions (RQ3).

Role C.
"""

from __future__ import annotations

from nlp4s.schema import Example


def build_prompt(
    text: str,
    language: str,
    condition: str,
    demonstrations: list[Example] | None = None,
) -> str:
    """Build the classification prompt for one instance.

    Args:
        text: the instance to classify.
        language: ISO 639-1 code of the instance.
        condition: "no_explanation" (direct classify) or "explanation"
            ("classify and explain why it is or is not hateful").
        demonstrations: optional few-shot examples to prepend.

    TODO(Role C): assemble system/instruction + optional demonstrations + the
    target text. The explanation condition must request a rationale alongside the
    label; the no-explanation condition must request the label only.
    """
    raise NotImplementedError("TODO(Role C): implement prompt construction")


def parse_response(raw: str, condition: str) -> tuple[str, str | None]:
    """Parse a model response into (pred_label, rationale).

    Returns ``rationale=None`` for the no-explanation condition.

    TODO(Role C): extract the label (mapped to schema.LABELS) and, for the
    explanation condition, the rationale text.
    """
    raise NotImplementedError("TODO(Role C): implement response parsing")
