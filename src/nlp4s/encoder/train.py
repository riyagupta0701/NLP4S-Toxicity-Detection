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

def load_and_standardize_data(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    """Load MLMA, Multi3Hate, and HASOC, filter for the 7 target languages,
    and return unified train and validation Dataset objects.
    """
    target_langs = set(config["languages"])
    
    # 1. Load from Hugging Face Hub
    mlma_raw = load_dataset(config["datasets"]["mlma"], split="train")
    m3hate_raw = load_dataset(config["datasets"]["multi3hate"], split="train")
    
    # 2. Extract and format to standard columns: 'text', 'language', 'label'
    # (Note: Exact lambda mapping depends on each dataset's internal column keys)
    
    # 3. Stratified splitting needs to happen here for MLMA and Multi3Hate
    # before merging with HASOC's pre-defined validation sets.
    
    # TODO: Combine datasets into final_train and final_val
    pass

def train(config: dict[str, Any]) -> str:
    output_dir = config["train"]["output_dir"]
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    
    #load raw data and model
    examples = read_jsonl(config["train"]["data_path"])
    model, tokenizer = load_model_and_tokenizer(model_name, config["model"]["num_labels"])
    
    texts = [ex.text for ex in examples]
    labels = [1 if ex.label == "hateful" else 0 for ex in examples]
    raw_dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    #tokenization func
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
        
    tokenized_dataset = raw_dataset.map(tokenize_fn, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["train"]["epochs"],
        per_device_train_batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        seed=config["train"]["seed"],
        use_mps_device=True,  # Enables Apple Silicon GPU acceleration
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir
    
    #raise NotImplementedError("TODO(Role B): implement fine-tuning loop")
