"""Results matrix export and visualizations.

Role D.
"""

from __future__ import annotations

from typing import Any


def write_matrix(grouped: dict[tuple[str, ...], dict[str, float]], out_path: str) -> None:
    """Write the functionality x language results matrix to CSV.

    TODO(Role D): flatten ``grouped`` into rows and save with pandas.
    """
    raise NotImplementedError("TODO(Role D): implement matrix export")


def plot_heatmap(grouped: dict[tuple[str, ...], dict[str, float]], figures_dir: str) -> None:
    """Render a functionality x language F1 heatmap.

    TODO(Role D): pivot to a matrix and save a matplotlib heatmap.
    """
    raise NotImplementedError("TODO(Role D): implement heatmap plotting")


def run(config: dict[str, Any]) -> str:
    """Evaluation entrypoint per ``eval.yaml``; writes metrics + figures.

    Returns the metrics output path.

    TODO(Role D): load examples + predictions, call metrics.score, write the
    matrix and figures using ``config``.
    """
    raise NotImplementedError("TODO(Role D): implement evaluation entrypoint")
