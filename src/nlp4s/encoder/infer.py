"""Run the fine-tuned encoder over MHC and emit Prediction records.

Role B.
"""

from __future__ import annotations

from typing import Any

from nlp4s.schema import Example, Prediction


def predict(model_dir: str, examples: list[Example], model_name: str) -> list[Prediction]:
    """Classify each Example, returning Prediction records (condition="").

    TODO(Role B): load the saved model, batch-tokenize, run inference, and map
    logits to schema.LABELS. ``example.id`` populates ``Prediction.example_id``.
    """
    raise NotImplementedError("TODO(Role B): implement encoder inference")


def run(config: dict[str, Any]) -> str:
    """Inference entrypoint: load MHC, predict, write predictions JSONL.

    Returns the predictions output path.

    TODO(Role B): wire MHC loading -> predict -> io_utils.write_jsonl using
    ``config["infer"]``.
    """
    raise NotImplementedError("TODO(Role B): implement inference entrypoint")
