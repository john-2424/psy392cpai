"""Action-conditioned deep successor features (slide-15 future-work fix #3).

Architecture:
    obs       -> phi_shared(s)        [GridCNNEncoder, shared]
    phi_shared(s) -> phi(s, a)        [ActionConditionedFeatureHead]
    phi(s, a) -> psi(s, a)            [PerActionSRHead]
    Q(s, a) = <psi(s, a), w>

Compared with the baseline SR (src/algorithms/sr.py) the key difference
is that phi is per-action by construction, which forces Q-values to
differ along the action axis even when the underlying state geometry
collapses two actions onto identical features. This is the architectural
fix for the policy-extraction fragility diagnosed in slide 14.

Loss:
    SR Bellman:    psi(s, a) ~ phi(s, a) + gamma * psi(s', a*)  [a* from target Q]
    Reward model:  r(s, a)   ~ <phi(s, a), w>
    Optional Q-margin hinge (same form as compute_sr_loss).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import GridCNNEncoder
from src.models.sr_heads_ac import ActionConditionedFeatureHead, PerActionSRHead


class SRACNet(nn.Module):
    """Action-conditioned deep successor feature network."""

    def __init__(
        self,
        obs_channels: int = 3,
        grid_size: int = 8,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        self.encoder = GridCNNEncoder(feature_dim=feature_dim)
        self.action_feature_head = ActionConditionedFeatureHead(
            feature_dim=feature_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        )
        self.sr_head = PerActionSRHead(
            feature_dim=feature_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        )
        self.reward_weights = nn.Parameter(torch.zeros(feature_dim))

    def encode_shared(self, obs: torch.Tensor) -> torch.Tensor:
        """phi_shared(s) with the same L2-norm bound as the baseline SR."""
        phi = self.encoder(obs)
        return F.normalize(phi, p=2, dim=-1)

    def action_features(self, obs: torch.Tensor) -> torch.Tensor:
        """phi(s, a) for all actions. Per-action L2-normalized so the SR
        Bellman fixed point stays bounded action-wise."""
        phi_shared = self.encode_shared(obs)
        phi_sa = self.action_feature_head(phi_shared)
        return F.normalize(phi_sa, p=2, dim=-1)

    def successor_features(self, obs: torch.Tensor) -> torch.Tensor:
        phi_sa = self.action_features(obs)
        psi = self.sr_head(phi_sa)
        return psi

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        psi = self.successor_features(obs)
        return torch.einsum("bad,d->ba", psi, self.reward_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs)


@dataclass
class SRACBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


def select_action(
    model: SRACNet,
    obs: torch.Tensor,
    epsilon: float,
    device: str = "cpu",
) -> int:
    if random.random() < epsilon:
        return random.randrange(model.n_actions)
    with torch.no_grad():
        obs = obs.unsqueeze(0).to(device)
        q = model.q_values(obs)
        return int(q.argmax(dim=1).item())


def compute_sr_ac_loss(
    model: SRACNet,
    target_model: SRACNet,
    batch: SRACBatch,
    gamma: float = 0.99,
    q_margin_weight: float = 0.0,
    q_margin: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """SR-Bellman + reward + optional Q-margin loss for action-conditioned SF."""
    obs = batch.obs
    actions = batch.actions.long()
    rewards = batch.rewards
    next_obs = batch.next_obs
    dones = batch.dones.float()
    idx = torch.arange(obs.shape[0], device=obs.device)

    phi_sa_all = model.action_features(obs)            # [B, A, d]
    psi_sa_all = model.sr_head(phi_sa_all)             # [B, A, d]
    phi_sa = phi_sa_all[idx, actions]                  # [B, d]
    psi_sa = psi_sa_all[idx, actions]                  # [B, d]

    with torch.no_grad():
        phi_sa_target_all = target_model.action_features(obs)
        phi_sa_target = phi_sa_target_all[idx, actions]                # [B, d]

        next_q = target_model.q_values(next_obs)                       # [B, A]
        next_actions = next_q.argmax(dim=1)
        next_psi_all = target_model.successor_features(next_obs)
        next_psi = next_psi_all[idx, next_actions]                     # [B, d]

        sr_target = phi_sa_target + gamma * (1.0 - dones.unsqueeze(1)) * next_psi

    sr_loss = F.mse_loss(psi_sa, sr_target)

    # Reward model: predict r from the action-conditioned feature for the
    # action actually taken. Mirrors Barreto 2017 r(s,a) = <phi(s,a), w>.
    pred_reward = torch.einsum("bd,d->b", phi_sa, model.reward_weights)
    reward_loss = F.mse_loss(pred_reward, rewards)

    REWARD_LOSS_WEIGHT = 5.0
    total_loss = sr_loss + REWARD_LOSS_WEIGHT * reward_loss

    margin_loss_val = 0.0
    mean_q_gap_val = 0.0
    if q_margin_weight > 0.0:
        q_all = torch.einsum("bad,d->ba", psi_sa_all, model.reward_weights)
        top2_vals, _ = q_all.topk(2, dim=1)
        q_gap = top2_vals[:, 0] - top2_vals[:, 1]
        margin_loss = F.relu(q_margin - q_gap).mean()
        total_loss = total_loss + q_margin_weight * margin_loss
        margin_loss_val = float(margin_loss.item())
        mean_q_gap_val = float(q_gap.mean().item())

    metrics = {
        "sr_loss": float(sr_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "total_loss": float(total_loss.item()),
        "mean_q": float(model.q_values(obs).mean().item()),
        "margin_loss": margin_loss_val,
        "mean_q_gap": mean_q_gap_val,
    }
    return total_loss, metrics


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.02) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - tau)
            t.data.add_(tau * s.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(copy.deepcopy(source.state_dict()))


def freeze_dynamics(model: SRACNet) -> None:
    """Freeze encoder, action-feature head, and per-action SR head; leave
    reward_weights trainable. Used for the Momennejad-style w-only adaptation
    variant on reward_change."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.action_feature_head.parameters():
        p.requires_grad = False
    for p in model.sr_head.parameters():
        p.requires_grad = False
    model.reward_weights.requires_grad_(True)


def unfreeze_all(model: SRACNet) -> None:
    for p in model.parameters():
        p.requires_grad = True
