"""Convenience runner: train the encoder and immediately run inference.

Equivalent to:
    nlp4s train --config configs/encoder.yaml
    nlp4s infer --config configs/encoder.yaml

Run from the project root:
    python src/nlp4s/encoder/run.py
"""

from pathlib import Path

from nlp4s.config import load_yaml
from nlp4s.encoder import infer, train

PROJECT_ROOT = Path(__file__).parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "encoder.yaml"

if __name__ == "__main__":
    config = load_yaml(CONFIG_PATH)
    train.train(config)
    infer.run(config)