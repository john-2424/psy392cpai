from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_encoder import GridCNNEncoder
from src.models.q_heads import QValueHead


class ReplayQNet(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        n_actions: int = 4,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.encoder = GridCNNEncoder(feature_dim=feature_dim)
        self.q_head = QValueHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_actions=n_actions,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values


@dataclass
class ReplayTransition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition: ReplayTransition) -> None:
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, device: str = "cpu"):
        batch = random.sample(self.buffer, batch_size)

        obs = torch.stack([t.obs for t in batch]).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_obs = torch.stack([t.next_obs for t in batch]).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

        return obs, actions, rewards, next_obs, dones


def select_action(model: ReplayQNet, obs: torch.Tensor, epsilon: float, device: str = "cpu") -> int:
    if random.random() < epsilon:
        return random.randrange(model.n_actions)

    with torch.no_grad():
        q_values = model(obs.unsqueeze(0).to(device))
        return int(q_values.argmax(dim=1).item())


def compute_q_loss(
    model: ReplayQNet,
    target_model: ReplayQNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
):
    q_values = model(obs)                                    # [B, A]
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_obs)                      # [B, A]
        next_q_max = next_q.max(dim=1).values
        td_target = rewards + gamma * (1.0 - dones) * next_q_max

    loss = F.mse_loss(q_sa, td_target)

    metrics = {
        "q_loss": float(loss.item()),
        "mean_q": float(q_values.mean().item()),
        "mean_td_target": float(td_target.mean().item()),
    }
    return loss, metrics


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.01) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(copy.deepcopy(source.state_dict()))