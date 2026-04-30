"""Action-conditioned feature head for deep SF (Barreto 2017, Lehnert 2024).

Implements the slide-15 future-work fix #3: instead of a single shared
phi(s) that is the same regardless of action, project the encoder output
into n_actions distinct per-action feature vectors phi(s, a). This is the
canonical successor-features parameterization in Barreto et al. 2017
("Successor Features for Transfer in Reinforcement Learning").

Why it matters here. The slide-14 diagnosis showed SR's representation
collapses the column axis (agent_col R^2 = -0.15), which makes Q-values
tie along the column action dimension and breaks argmax. With per-action
features, phi(s, "left") and phi(s, "right") are forced to differ by
construction, so column ties cannot survive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedFeatureHead(nn.Module):
    """Map a shared state feature phi_shared(s) to per-action features
    phi(s, a) for each discrete action.

    Input:  phi_shared [B, d]
    Output: phi_sa     [B, A, d]
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        # One hidden MLP layer then a per-action linear projection. The
        # hidden layer gives the network capacity to mix the shared phi
        # before splitting into n_actions parallel heads.
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * feature_dim),
        )

    def forward(self, phi_shared: torch.Tensor) -> torch.Tensor:
        batch_size = phi_shared.shape[0]
        out = self.net(phi_shared)
        return out.view(batch_size, self.n_actions, self.feature_dim)


class PerActionSRHead(nn.Module):
    """Per-action SR head: maps phi(s, a) -> psi(s, a) with one MLP per action.

    Used together with ActionConditionedFeatureHead to implement the
    Barreto-style action-conditional SF: each action has its own (phi, psi)
    parameterization. Compared to a single shared SF head, this gives the
    network direct architectural capacity to differentiate Q-values along
    the action axis.

    Input:  phi_sa [B, A, d]
    Output: psi_sa [B, A, d]
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
            for _ in range(n_actions)
        ])

    def forward(self, phi_sa: torch.Tensor) -> torch.Tensor:
        outputs = [self.heads[a](phi_sa[:, a, :]) for a in range(self.n_actions)]
        return torch.stack(outputs, dim=1)
