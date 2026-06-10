"""XLM-RoBERTa model + tokenizer wrappers."""

from __future__ import annotations
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from typing import Any


def load_model_and_tokenizer(name: str, num_labels: int) -> tuple[Any, Any]:
    """Load an XLM-RoBERTa sequence-classification model and its tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tokenizer
    
