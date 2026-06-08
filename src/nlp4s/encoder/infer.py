"""Run the fine-tuned encoder over MHC and emit Prediction records.

Role B.
"""

from __future__ import annotations
import torch
from typing import Any
from transformers import pipeline
from nlp4s.schema import Example, Prediction
from nlp4s.io_utils import read_jsonl, write_jsonl


def predict(model_dir: str, examples: list[Example], model_name: str) -> list[Prediction]:
    """Classify each Example, returning Prediction records (condition=""). """

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    classifier = pipeline(
        "text-classification", 
        model=model_dir, 
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
    #raise NotImplementedError("TODO(Role B): implement encoder inference")


def run(config: dict[str, Any]) -> str:
    """Inference entrypoint: load MHC, predict, write predictions JSONL.

    Returns the predictions output path. """

    eval_path = config["infer"]["eval_path"]
    predictions_out = config["infer"]["predictions_out"]
    model_dir = config["train"]["output_dir"]
    model_name = config["model"]["name"]
    
    examples = read_jsonl(eval_path)
    predictions = predict(model_dir, examples, model_name)
    
    # Write predictions utilizing standard dict conversion
    write_jsonl(predictions_out, [p.to_dict() for p in predictions])
    
    return predictions_out
    raise NotImplementedError("TODO(Role B): implement inference entrypoint")
