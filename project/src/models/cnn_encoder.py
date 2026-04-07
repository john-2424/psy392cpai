from __future__ import annotations

import torch
import torch.nn as nn


class GridCNNEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(32 * 8 * 8, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 3:
            observation = observation.unsqueeze(0)
            squeeze_back = True
            leading_shape = None
        elif observation.ndim == 4:
            squeeze_back = False
            leading_shape = observation.shape[:-3]
        else:
            squeeze_back = False
            leading_shape = observation.shape[:-3]
            observation = observation.reshape(-1, *observation.shape[-3:])

        x = self.conv(observation)
        x = self.head(x)

        if squeeze_back:
            x = x.squeeze(0)
        elif leading_shape is not None and len(leading_shape) > 1:
            x = x.reshape(*leading_shape, x.shape[-1])

        return x