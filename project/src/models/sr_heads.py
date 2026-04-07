from __future__ import annotations

import torch
import torch.nn as nn


class SuccessorFeatureHead(nn.Module):
    """
    Maps state features phi(s) to successor features psi(s, a)
    for each discrete action.

    Input:
        state_features: [B, d]
    Output:
        psi: [B, A, d]
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        n_actions: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * feature_dim),
        )

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        batch_size = state_features.shape[0]
        out = self.net(state_features)
        out = out.view(batch_size, self.n_actions, self.feature_dim)
        return out