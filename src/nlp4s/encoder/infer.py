"""Run the fine-tuned encoder over MHC and emit Prediction records."""

from __future__ import annotations
import torch
from typing import Any
from transformers import pipeline
from nlp4s.schema import Example, Prediction
from nlp4s.io_utils import read_jsonl, write_jsonl
from datasets import load_from_disk

from pathlib import Path
PROJECT_ROOT = Path(__file__).parents[3]


def predict(model_dir: str, examples: list[Example], model_name: str) -> list[Prediction]:
    """Classify each Example, returning Prediction records (condition=""). """

    device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
    )
    print(f"Using device: {device}")
    
    classifier = pipeline(
        "text-classification", 
        model=str(model_dir), 
        tokenizer=model_dir, 
        device=device
    )
    
    texts = [ex.text for ex in examples]
    # Process in batches to prevent memory limits
    results = classifier(texts, batch_size=16, truncation=True, max_length=128)
    
    predictions = []
    for ex, res in zip(examples, results):
        # HuggingFace defaults to LABEL_0 and LABEL_1. 1 maps to "hateful".
        pred_label = "hateful" if res["label"] == "LABEL_1" else "non-hateful"
        predictions.append(Prediction(
            example_id=str(ex.id),
            model=model_name,
            pred_label=pred_label,
            condition=""
        ))
        
    return predictions


def run(config: dict[str, Any]) -> str:

    eval_path = PROJECT_ROOT / config["infer"]["eval_path"]
    predictions_out = PROJECT_ROOT / config["infer"]["predictions_out"]
    model_dir = PROJECT_ROOT / config["train"]["output_dir"]
    model_name = config["model"]["name"]
        
    TRAIN_LANGS = set(config["languages"])
    examples = [ex for ex in read_jsonl(eval_path) if ex.language in TRAIN_LANGS]

    predictions = predict(str(model_dir), examples, model_name)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_out, predictions)
    return predictions_out