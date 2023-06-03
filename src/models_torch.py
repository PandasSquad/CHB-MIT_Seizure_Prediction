"""Models to be trained and tested on the dataset"""
import torch.nn as nn
import torch


class MLP(nn.Module):
    """Simple MLP model."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(22 * 256*60, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(tensor)
