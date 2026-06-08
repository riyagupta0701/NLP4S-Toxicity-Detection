"""The frozen data schema (shared contract).

Every data and prediction record in this project uses these types so the four
workstreams (data, encoder, llm, eval) can develop independently against
fixtures. See docs/implementation_plan.md. Do not diverge silently.

Data record fields: ``text, language, label, functionality, split``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Binary task labels.
LABELS: frozenset[str] = frozenset({"hateful", "non-hateful"})

# Prompting conditions for the LLM experiments (RQ3).
CONDITIONS: frozenset[str] = frozenset({"no_explanation", "explanation"})


@dataclass
class Example:
    """A single input instance in the frozen schema.

    Attributes:
        text: the raw text.
        language: ISO 639-1 code (e.g. "en", "hi"); see functionalities.MHC_LANGUAGES.
        label: gold label, one of schema.LABELS.
        functionality: MHC functionality label (e.g. "derog_impl_h"); empty
            string for training data that has no functionality annotation.
        split: dataset split / origin (e.g. "test", "train", "synthetic", "pool").
        id: optional stable identifier.
        target: MHC target-identity group (e.g. "women", "Muslims"); None when
            the source dataset does not carry one (HASOC, synthetic).
    """

    text: str
    language: str
    label: str
    functionality: str = ""
    split: str = ""
    id: str | None = None
    target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        return cls(
            text=d["text"],
            language=d["language"],
            label=d["label"],
            functionality=d.get("functionality", ""),
            split=d.get("split", ""),
            id=d.get("id"),
            target=d.get("target"),
        )


@dataclass
class Prediction:
    """A model prediction for one Example.

    Attributes:
        example_id: id of the Example this prediction is for.
        model: model name (e.g. "xlm-roberta-base", "aya-23").
        condition: prompting condition (one of schema.CONDITIONS) or "" for the encoder.
        pred_label: predicted label, one of schema.LABELS.
        rationale: generated explanation text (LLM explanation condition), else None.
    """

    example_id: str
    model: str
    pred_label: str
    condition: str = ""
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        return cls(
            example_id=d["example_id"],
            model=d["model"],
            pred_label=d["pred_label"],
            condition=d.get("condition", ""),
            rationale=d.get("rationale"),
        )


def validate(example: Example) -> Example:
    """Validate an Example against the frozen schema.

    Performs basic field checks and returns the example unchanged on success.

    Raises:
        ValueError: if a required field is empty or a label is not in LABELS.

    TODO: extend with cross-field validation, e.g. checking ``language`` against
    functionalities.MHC_LANGUAGES and ``functionality`` against the known
    EXPLICIT/IMPLICIT/CONTROL sets where applicable.
    """
    if not example.text:
        raise ValueError("Example.text must be non-empty")
    if not example.language:
        raise ValueError("Example.language must be set (ISO 639-1)")
    if example.label not in LABELS:
        raise ValueError(f"Example.label must be one of {sorted(LABELS)}, got {example.label!r}")
    return example
