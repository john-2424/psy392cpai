"""Ablation: SR without phi normalization.

Runs a short SR training (100 eps, seed 0) with `SRNet.encode` monkey-patched
to skip L2 normalization. Documents the deep-SR pixel-feature collapse
described by Lehnert et al. 2024 (arXiv 2410.22133). Produces
`sr_no_norm_seed0_train.csv` used as an ablation figure in the report.
"""
from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import torch
from torch.optim import Adam

from src.algorithms.sr import (
    SRNet,
    compute_sr_loss,
    hard_update,
    select_action,
    soft_update,
    unfreeze_all,
)
from src.common.adaptation import STABLE, append_csv_row, make_env
from scripts.train_sr import SRReplayBuffer, SRReplayTransition, obs_from_td


NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 50
GAMMA = 0.95
LR = 3e-4
TAU = 0.05
BUFFER_CAPACITY = 5000
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 100
REPLAY_UPDATES_PER_STEP = 2
EPS_START, EPS_END, EPS_DECAY_EPS = 1.0, 0.05, 80
LOSS_EARLY_STOP = 1e8  # bail out if loss explodes past this


def encode_no_norm(self, obs: torch.Tensor) -> torch.Tensor:
    """Replacement for SRNet.encode that skips L2 normalization."""
    return self.encoder(obs)


def main():
    seed = 0
    device = "cpu"
    torch.manual_seed(seed)
    random.seed(seed)

    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "sr_no_norm_seed0_train.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Monkey-patch SRNet.encode on the class before instantiating.
    SRNet.encode = encode_no_norm  # type: ignore[assignment]

    train_env = make_env(STABLE, seed=seed)
    model = SRNet(feature_dim=64, hidden_dim=64, n_actions=4).to(device)
    target_model = SRNet(feature_dim=64, hidden_dim=64, n_actions=4).to(device)
    hard_update(target_model, model)
    unfreeze_all(model)
    unfreeze_all(target_model)

    optimizer = Adam(model.parameters(), lr=LR)
    buffer = SRReplayBuffer(capacity=BUFFER_CAPACITY)

    fieldnames = [
        "seed", "episode", "global_step", "epsilon",
        "episode_return", "phi_norm", "reward_weights_norm",
        "sr_loss", "reward_loss", "total_loss",
    ]

    global_step = 0
    for episode in range(1, NUM_EPISODES + 1):
        epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPS),
        )
        td = train_env.reset()
        obs = obs_from_td(td, device=device)
        ep_return = 0.0
        last_metrics = {"sr_loss": 0.0, "reward_loss": 0.0, "total_loss": 0.0}
        phi_norm_val = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            global_step += 1
            action = select_action(model, obs, epsilon=epsilon, device=device)
            action_td = td.clone()
            action_td.set("action", torch.tensor(action, dtype=torch.long))
            step_td = train_env.step(action_td)
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            next_obs = next_td["observation"].to(device)
            reward = float(next_td["reward"].item())
            done = float(bool(next_td["done"].item()))

            buffer.add(SRReplayTransition(
                obs=obs.detach().cpu(),
                action=action,
                reward=reward,
                next_obs=next_obs.detach().cpu(),
                done=done,
            ))

            if len(buffer) >= MIN_BUFFER_SIZE:
                for _ in range(REPLAY_UPDATES_PER_STEP):
                    batch = buffer.sample(batch_size=BATCH_SIZE, device=device)
                    loss, metrics = compute_sr_loss(model, target_model, batch, gamma=GAMMA)
                    optimizer.zero_grad()
                    loss.backward()
                    # NOTE: no gradient clipping here either -- this is the ablation.
                    optimizer.step()
                    soft_update(target_model, model, tau=TAU)
                    last_metrics = metrics
                with torch.no_grad():
                    phi_norm_val = float(model.encode(obs.unsqueeze(0)).norm().item())

            ep_return += reward
            obs = next_obs
            td = next_td
            if bool(done):
                break

        append_csv_row(
            csv_path,
            {
                "seed": seed,
                "episode": episode,
                "global_step": global_step,
                "epsilon": epsilon,
                "episode_return": ep_return,
                "phi_norm": phi_norm_val,
                "reward_weights_norm": float(model.reward_weights.norm().item()),
                **last_metrics,
            },
            fieldnames,
        )

        if episode % 10 == 0 or episode <= 5:
            print(
                f"[no-norm ep={episode}] "
                f"total_loss={last_metrics['total_loss']:.3e} "
                f"phi_norm={phi_norm_val:.3e}"
            )

        if last_metrics["total_loss"] > LOSS_EARLY_STOP:
            print(f"Early stop: loss exceeded {LOSS_EARLY_STOP:.0e} at ep {episode}")
            break


if __name__ == "__main__":
    main()
