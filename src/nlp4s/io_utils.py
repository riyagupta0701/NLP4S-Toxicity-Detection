"""JSONL read/write helpers for the frozen schema (shared contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from nlp4s.schema import Example, Prediction


def read_jsonl(path: str | Path) -> list[Example]:
    """Read a JSONL file of Example records."""
    path = Path(path)
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            examples.append(Example.from_dict(json.loads(line)))
    return examples


def write_jsonl(path: str | Path, records: Iterable[Example | Prediction]) -> None:
    """Write Example/Prediction records to JSONL, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_predictions(path: str | Path) -> list[Prediction]:
    """Read a JSONL file of Prediction records."""
    path = Path(path)
    preds: list[Prediction] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            preds.append(Prediction.from_dict(json.loads(line)))
    return preds
