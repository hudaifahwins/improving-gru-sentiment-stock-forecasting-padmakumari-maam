"""Baseline GRU stock forecasting model (stub — implementation pending).

Planned interface:
    model = BaselineGRU(input_dim, hidden_dim, num_layers)
    preds = model(price_window)
"""

from __future__ import annotations


class BaselineGRU:
    """Placeholder for the baseline GRU forecaster."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, num_layers: int = 2) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def __call__(self, x):
        raise NotImplementedError("Baseline GRU training code not yet implemented.")


def train_baseline(data_path: str, config: dict | None = None) -> None:
    """Train the baseline model. Stub for now."""
    raise NotImplementedError("train_baseline not yet implemented.")
