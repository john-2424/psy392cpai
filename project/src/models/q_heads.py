from __future__ import annotations

import torch
import torch.nn as nn


class QValueHead(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)