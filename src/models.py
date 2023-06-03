"""Models definition."""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch import Tensor
from typing import Any


class MLP(pl.LightningModule):
    """Simple MLP model."""

    def __init__(self, train_data: Any, val_data: Any, test_data: Any) -> None:
        """
        Initialize the MLP.

        :return: None
        """
        super().__init__()

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(22 * 256 * 60, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Forward pass through the network.

        :param tensor: Input tensor
        :return: Output tensor
        """
        return self.network(tensor)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        Perform one step of training.

        :param batch: Batch data
        :param batch_idx: Index of the batch
        :return: Loss tensor
        """
        x, y = batch
        y_hat = self.network(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Perform one step of validation.

        :param batch: Batch data
        :param batch_idx: Index of the batch
        :return: None
        """
        x, y = batch
        y_hat = self.network(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        :return: Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=64)
