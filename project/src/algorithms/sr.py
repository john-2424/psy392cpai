from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import GridCNNEncoder
from src.models.sr_heads import SuccessorFeatureHead


DEBUG_SR = False


class SRNet(nn.Module):
    """
    Deep successor feature network:
      obs -> phi(s)
      phi(s) -> psi(s,a)
      Q(s,a) = <psi(s,a), w>
    """

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
        self.sr_head = SuccessorFeatureHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_actions=n_actions,
        )

        # Reward weights w for r(s) ≈ phi(s)^T w
        self.reward_weights = nn.Parameter(torch.zeros(feature_dim))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        phi = self.encoder(obs)
        # L2-normalise so ||phi||=1, bounding the SR fixed-point to
        # ||psi*|| <= 1 / (1-gamma).  Without this, phi can grow
        # unboundedly during encoder training and cause loss explosion.
        return F.normalize(phi, p=2, dim=-1)

    def successor_features(self, obs: torch.Tensor) -> torch.Tensor:
        phi = self.encode(obs)                     # [B, d]
        psi = self.sr_head(phi)                   # [B, A, d]
        return psi

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        psi = self.successor_features(obs)        # [B, A, d]
        q = torch.einsum("bad,d->ba", psi, self.reward_weights)
        return q

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_values(obs)


@dataclass
class SRBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


def select_action(
    model: SRNet,
    obs: torch.Tensor,
    epsilon: float,
    device: str = "cpu",
) -> int:
    if random.random() < epsilon:
        return random.randrange(model.n_actions)

    with torch.no_grad():
        obs = obs.unsqueeze(0).to(device)   # [1, C, H, W]
        q = model.q_values(obs)             # [1, A]
        action = int(q.argmax(dim=1).item())
    return action


def compute_sr_loss(
    model: SRNet,
    target_model: SRNet,
    batch: SRBatch,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, dict]:
    """
    SR Bellman target:
      psi(s,a) ≈ phi(s) + gamma * psi(s', a*)
    where a* = argmax_a' Q_target(s', a')
    """
    obs = batch.obs
    actions = batch.actions.long()
    rewards = batch.rewards
    next_obs = batch.next_obs
    dones = batch.dones.float()

    phi = model.encode(obs)                                 # [B, d]
    psi_all = model.sr_head(phi)                            # [B, A, d]
    psi_sa = psi_all[torch.arange(obs.shape[0]), actions]  # [B, d]

    with torch.no_grad():
        # Use target network's phi for a stable, non-moving SR target
        phi_target = target_model.encode(obs)               # [B, d]
        next_q = target_model.q_values(next_obs)            # [B, A]
        next_actions = next_q.argmax(dim=1)                 # [B]
        next_psi_all = target_model.successor_features(next_obs)
        next_psi = next_psi_all[torch.arange(obs.shape[0]), next_actions]  # [B, d]

        sr_target = phi_target + gamma * (1.0 - dones.unsqueeze(1)) * next_psi

    sr_loss = F.mse_loss(psi_sa, sr_target)

    # Reward model: immediate reward predicted from phi(s)
    pred_reward = torch.einsum("bd,d->b", phi, model.reward_weights)
    reward_loss = F.mse_loss(pred_reward, rewards)

    # Emphasize reward prediction so that reward weights w learn fast.
    total_loss = sr_loss + 20.0 * reward_loss

    if DEBUG_SR and obs.shape[0] == 1:
        print("phi shape:", phi.shape)
        print("psi_all shape:", psi_all.shape)
        print("actions shape:", actions.shape)
        print("psi_sa shape:", psi_sa.shape)
        print("next_psi shape:", next_psi.shape)
        print("sr_target shape:", sr_target.shape)
        print("dones:", dones.shape, dones)
        print("sr_loss:", sr_loss.item(), "reward_loss:", reward_loss.item())

    metrics = {
        "sr_loss": float(sr_loss.item()),
        "reward_loss": float(reward_loss.item()),
        "total_loss": float(total_loss.item()),
        "mean_q": float(model.q_values(obs).mean().item()),
    }
    return total_loss, metrics


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.02) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(copy.deepcopy(source.state_dict()))


def freeze_encoder_and_sr_head(model: SRNet) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.sr_head.parameters():
        param.requires_grad = False
    model.reward_weights.requires_grad_(True)


def unfreeze_all(model: SRNet) -> None:
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.sr_head.parameters():
        param.requires_grad = True
    model.reward_weights.requires_grad_(True)