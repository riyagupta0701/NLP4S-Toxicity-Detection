"""Fine-tune XLM-RoBERTa for binary hate speech classification.

Role B. Produces the first baseline that gates the Phase-2 decision, and is
re-run with synthetic augmentation later. See docs/implementation_plan.md.
"""

from __future__ import annotations

from typing import Any


def train(config: dict[str, Any]) -> str:
    """Fine-tune the encoder per the ``encoder.yaml`` config.

    Args:
        config: parsed YAML config (see configs/encoder.yaml).

    Returns:
        Path to the saved model directory.

    TODO(Role B): read training data via nlp4s.io_utils.read_jsonl, tokenize,
    run the HuggingFace Trainer with the configured hyperparameters, and save to
    ``config["train"]["output_dir"]``.
    """
    raise NotImplementedError("TODO(Role B): implement fine-tuning loop")
