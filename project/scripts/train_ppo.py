from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector

from src.envs.gridworld import GridWorldEnv
from src.algorithms.ppo_torchrl import build_ppo_components


def make_env(mode: str = "stable", seed: int = 0) -> GridWorldEnv:
    return GridWorldEnv(
        grid_size=8,
        max_steps=50,
        change_mode=mode,
        seed=seed,
    )


def append_csv_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(policy, value_model, optimizer, step_idx: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"ppo_step_{step_idx}.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step_idx": step_idx,
        },
        ckpt_path,
    )


@torch.no_grad()
def evaluate_policy(policy, env, n_episodes: int = 20, device: str = "cpu"):
    policy.eval()

    successes = 0
    returns = []
    steps_list = []

    for _ in range(n_episodes):
        td = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done:
            td = td.to(device)
            td = policy(td)
            if "logits" in td.keys():
                td["action"] = td["logits"].argmax(dim=-1)
            td = env.step(td)

            step_td = td["next"] if "next" in td.keys() else td

            reward = float(step_td["reward"].item())
            done = bool(step_td["done"].item())
            ep_return += reward
            ep_steps += 1

            td = step_td.select("observation")

        success = bool(env.agent_pos == env.goal_pos)
        successes += int(success)
        returns.append(ep_return)
        steps_list.append(ep_steps)

    policy.train()

    return {
        "success_rate": successes / n_episodes,
        "avg_return": sum(returns) / len(returns),
        "avg_steps": sum(steps_list) / len(steps_list),
    }


def train():
    device = "cpu"
    seed = 0

    frames_per_batch = 512
    total_frames = 50000
    ppo_epochs = 4
    minibatch_size = 64
    lr = 3e-4
    eval_interval = 10
    save_interval = 20
    max_grad_norm = 1.0

    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    model_dir = results_dir / "models"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env("stable", seed=seed)
    eval_env_stable = make_env("stable", seed=seed + 1)
    eval_env_reward = make_env("reward_change", seed=seed + 2)
    eval_env_transition = make_env("transition_change", seed=seed + 3)

    components = build_ppo_components(train_env)
    policy = components["policy"].to(device)
    value_model = components["value_model"].to(device)
    advantage_module = components["advantage_module"]
    loss_module = components["loss_module"]

    optimizer = Adam(loss_module.parameters(), lr=lr)

    collector = SyncDataCollector(
        create_env_fn=lambda: make_env("stable", seed=seed),
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    train_csv = csv_dir / f"ppo_seed{seed}_train.csv"

    collected_frames = 0
    batch_idx = 0

    for tensordict_data in collector:
        batch_idx += 1
        collected_frames += tensordict_data.numel()

        with torch.no_grad():
            advantage_module(tensordict_data)

        flat_data = tensordict_data.reshape(-1)

        last_loss_vals = None
        grad_norm_value = 0.0

        for _ in range(ppo_epochs):
            perm = torch.randperm(flat_data.shape[0])
            for start in range(0, flat_data.shape[0], minibatch_size):
                idx = perm[start:start + minibatch_size]
                minibatch = flat_data[idx]

                loss_vals = loss_module(minibatch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optimizer.step()

                grad_norm_value = float(grad_norm)
                last_loss_vals = loss_vals

        step_td = tensordict_data["next"] if "next" in tensordict_data.keys() else tensordict_data
        mean_reward = step_td["reward"].float().mean().item()
        done_rate = step_td["done"].float().mean().item()

        row = {
            "seed": seed,
            "batch_idx": batch_idx,
            "collected_frames": collected_frames,
            "mean_batch_reward": mean_reward,
            "done_rate": done_rate,
            "loss_objective": float(last_loss_vals["loss_objective"].item()) if last_loss_vals is not None else 0.0,
            "loss_critic": float(last_loss_vals["loss_critic"].item()) if last_loss_vals is not None else 0.0,
            "loss_entropy": float(last_loss_vals["loss_entropy"].item()) if last_loss_vals is not None else 0.0,
            "total_loss": float(
                (
                    last_loss_vals["loss_objective"]
                    + last_loss_vals["loss_critic"]
                    + last_loss_vals["loss_entropy"]
                ).item()
            ) if last_loss_vals is not None else 0.0,
            "grad_norm": grad_norm_value,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_csv_row(train_csv, row)

        print(
            f"[Train Batch {batch_idx}] "
            f"frames={collected_frames} "
            f"mean_batch_reward={mean_reward:.4f} "
            f"done_rate={done_rate:.4f}"
        )

        if batch_idx % eval_interval == 0:
            stable_metrics = evaluate_policy(policy, eval_env_stable, n_episodes=20, device=device)
            reward_metrics = evaluate_policy(policy, eval_env_reward, n_episodes=20, device=device)
            transition_metrics = evaluate_policy(policy, eval_env_transition, n_episodes=20, device=device)

            print(
                f"[Batch {batch_idx}] "
                f"Stable={stable_metrics['success_rate']:.3f} | "
                f"Reward={reward_metrics['success_rate']:.3f} | "
                f"Transition={transition_metrics['success_rate']:.3f}"
            )

            append_csv_row(
                csv_dir / f"ppo_seed{seed}_eval_stable.csv",
                {
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "collected_frames": collected_frames,
                    "condition": "stable",
                    "n_eval_episodes": 20,
                    **stable_metrics,
                },
            )
            append_csv_row(
                csv_dir / f"ppo_seed{seed}_eval_reward_change.csv",
                {
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "collected_frames": collected_frames,
                    "condition": "reward_change",
                    "n_eval_episodes": 20,
                    **reward_metrics,
                },
            )
            append_csv_row(
                csv_dir / f"ppo_seed{seed}_eval_transition_change.csv",
                {
                    "seed": seed,
                    "batch_idx": batch_idx,
                    "collected_frames": collected_frames,
                    "condition": "transition_change",
                    "n_eval_episodes": 20,
                    **transition_metrics,
                },
            )

        if batch_idx % save_interval == 0:
            save_checkpoint(policy, value_model, optimizer, batch_idx, model_dir)

    collector.shutdown()


if __name__ == "__main__":
    train()