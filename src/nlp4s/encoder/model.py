"""XLM-RoBERTa model + tokenizer wrappers.

Role B.
"""

from __future__ import annotations

from typing import Any


def load_model_and_tokenizer(name: str, num_labels: int) -> tuple[Any, Any]:
    """Load an XLM-RoBERTa sequence-classification model and its tokenizer.

    TODO(Role B): use transformers
    ``AutoModelForSequenceClassification.from_pretrained`` and
    ``AutoTokenizer.from_pretrained``.
    """
    raise NotImplementedError("TODO(Role B): implement model/tokenizer loading")
