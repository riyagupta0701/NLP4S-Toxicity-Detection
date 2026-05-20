"""YAML config loading + argparse helpers (shared contract)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping, got {type(data).__name__}")
    return data


def config_parser(description: str) -> argparse.ArgumentParser:
    """Build a parser that takes a single ``--config PATH`` argument.

    Reused by every CLI subcommand so config handling is uniform.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to the YAML config file for this step.",
    )
    return parser
