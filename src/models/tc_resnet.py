from __future__ import annotations

import torch
from torch import nn


class TCBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TCResNet(nn.Module):
    """Lightweight temporal convolutional residual network for log-Mel features."""

    def __init__(
        self,
        num_classes: int = 12,
        n_mels: int = 40,
        channels: int = 48,
        num_blocks: int = 4,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=(n_mels, 5),
                padding=(0, 2),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[TCBlock(channels, kernel_size=kernel_size, dropout=dropout) for _ in range(num_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x).squeeze(2)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x)
