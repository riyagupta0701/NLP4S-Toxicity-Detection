import yaml
from nlp4s.encoder import train, infer
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]  # src/nlp4s/encoder/run.py → up 3 → root
CONFIG_PATH  = PROJECT_ROOT / "configs" / "encoder.yaml"

if __name__ == "__main__":

    with open(CONFIG_PATH) as f:    
        config = yaml.safe_load(f)

    train.train(config)
    infer.run(config)