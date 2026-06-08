"""Fine-tune XLM-RoBERTa for binary hate speech classification.

Role B. Produces the first baseline that gates the Phase-2 decision, and is
re-run with synthetic augmentation later. See docs/implementation_plan.md.
"""

from __future__ import annotations
from typing import Any
from datasets import Dataset, load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from nlp4s.io_utils import read_jsonl
from nlp4s.encoder.model import load_model_and_tokenizer
import torch
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
import numpy as np


PROJECT_ROOT = Path(__file__).parents[3]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }
   

def train(config: dict[str, Any]) -> str:
    output_dir = config["train"]["output_dir"]
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    
    #load raw data and model
    train_df = pd.read_csv(PROJECT_ROOT / config["train"]["train_path"])
    val_df   = pd.read_csv(PROJECT_ROOT / config["train"]["val_path"])
    model, tokenizer = load_model_and_tokenizer(model_name, config["model"]["num_labels"])
    
    train_dataset = Dataset.from_dict({"text": train_df["text"].tolist(),
                                       "label": train_df["label"].tolist()})
    val_dataset   = Dataset.from_dict({"text": val_df["text"].tolist(),
                                       "label": val_df["label"].tolist()})
    
    #tokenization func
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset   = val_dataset.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["train"]["epochs"],
        per_device_train_batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        seed=config["train"]["seed"],
        save_strategy="epoch",
        eval_strategy="epoch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir
    
    #raise NotImplementedError("TODO(Role B): implement fine-tuning loop")
