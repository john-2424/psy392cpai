from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.optim import Adam

from src.envs.gridworld import GridWorldEnv
from src.algorithms.sr import (
    SRNet,
    SRBatch,
    select_action,
    compute_sr_loss,
    hard_update,
    soft_update,
    freeze_encoder_and_sr_head,
    unfreeze_all,
)


def append_csv_row(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def obs_from_td(td, device="cpu"):
    obs = td["observation"]
    return obs.to(device)


@torch.no_grad()
def evaluate_sr(model, env, n_episodes: int = 20, device: str = "cpu"):
    model.eval()

    successes = 0
    returns = []
    steps_list = []

    for _ in range(n_episodes):
        td = env.reset()
        obs = obs_from_td(td, device=device)

        done = False
        ep_return = 0.0
        ep_steps = 0

        while not done:
            action = select_action(model, obs, epsilon=0.0, device=device)

            step_td = env.step(td.clone().set("action", torch.tensor(action)))
            next_td = step_td["next"] if "next" in step_td.keys() else step_td

            reward = float(next_td["reward"].item())
            done = bool(next_td["done"].item())

            obs = next_td["observation"].to(device)
            td = next_td

            ep_return += reward
            ep_steps += 1

        success = bool(env.agent_pos == env.goal_pos)
        successes += int(success)
        returns.append(ep_return)
        steps_list.append(ep_steps)

    model.train()
    return {
        "success_rate": successes / n_episodes,
        "avg_return": sum(returns) / len(returns),
        "avg_steps": sum(steps_list) / len(steps_list),
    }


def train():
    device = "cpu"
    seed = 0
    torch.manual_seed(seed)

    num_episodes = 500
    max_steps_per_episode = 50
    gamma = 0.99
    lr = 1e-3
    tau = 0.01

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 300

    eval_interval = 25
    save_interval = 100

    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    model_dir = results_dir / "models"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="stable", seed=seed)
    eval_env_stable = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="stable", seed=seed + 1)
    eval_env_reward = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="reward_change", seed=seed + 2)
    eval_env_transition = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="transition_change", seed=seed + 3)

    model = SRNet(feature_dim=128, hidden_dim=128, n_actions=4).to(device)
    target_model = SRNet(feature_dim=128, hidden_dim=128, n_actions=4).to(device)
    hard_update(target_model, model)

    optimizer = Adam(model.parameters(), lr=lr)

    train_csv = csv_dir / f"sr_seed{seed}_train.csv"

    global_step = 0

    action_counts = [0, 0, 0, 0]

    for episode in range(1, num_episodes + 1):
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_episodes),
        )

        td = train_env.reset()
        obs = obs_from_td(td, device=device)

        ep_return = 0.0
        ep_steps = 0
        action_counts[action] += 1

        for _ in range(max_steps_per_episode):
            global_step += 1

            action = select_action(model, obs, epsilon=epsilon, device=device)

            action_td = td.clone()
            action_td.set("action", torch.tensor(action))
            step_td = train_env.step(action_td)
            next_td = step_td["next"] if "next" in step_td.keys() else step_td

            next_obs = next_td["observation"].to(device)
            reward = torch.tensor([float(next_td["reward"].item())], dtype=torch.float32, device=device)
            done = torch.tensor([bool(next_td["done"].item())], dtype=torch.float32, device=device)

            batch = SRBatch(
                obs=obs.unsqueeze(0),
                actions=torch.tensor([action], dtype=torch.long, device=device),
                rewards=reward,
                next_obs=next_obs.unsqueeze(0),
                dones=done,
            )

            loss, metrics = compute_sr_loss(model, target_model, batch, gamma=gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update(target_model, model, tau=tau)

            ep_return += float(reward.item())
            ep_steps += 1

            obs = next_obs
            td = next_td

            if bool(done.item()):
                break

        train_row = {
            "seed": seed,
            "episode": episode,
            "global_step": global_step,
            "epsilon": epsilon,
            "episode_return": ep_return,
            "episode_steps": ep_steps,
            "reward_weights_norm": float(model.reward_weights.norm().item()),
            **metrics,
        }
        append_csv_row(train_csv, train_row)

        if episode == 1 and ep_steps < 3:
            with torch.no_grad():
                phi = model.encode(obs.unsqueeze(0))
                print("obs shape:", obs.unsqueeze(0).shape)
                print("phi shape:", phi.shape)
                print("phi mean/std:", phi.mean().item(), phi.std().item())
                q = model.q_values(obs.unsqueeze(0))
                print("q_values:", q.cpu().numpy())
        
        if episode % 25 == 0:
            with torch.no_grad():
                w_norm = model.reward_weights.norm().item()
                print(f"[Episode {episode}] reward_weights norm = {w_norm:.4f}")
        
        if episode % 25 == 0:
            print(f"[Episode {episode}] action counts: {action_counts}")
            action_counts = [0, 0, 0, 0]
            
        if episode % eval_interval == 0:
            stable_metrics = evaluate_sr(model, eval_env_stable, n_episodes=20, device=device)
            reward_metrics = evaluate_sr(model, eval_env_reward, n_episodes=20, device=device)
            transition_metrics = evaluate_sr(model, eval_env_transition, n_episodes=20, device=device)

            print(
                f"[Episode {episode}] "
                f"stable={stable_metrics['success_rate']:.3f}, "
                f"reward={reward_metrics['success_rate']:.3f}, "
                f"transition={transition_metrics['success_rate']:.3f}"
            )

            append_csv_row(
                csv_dir / f"sr_seed{seed}_eval_stable.csv",
                {"seed": seed, "episode": episode, "condition": "stable", **stable_metrics},
            )
            append_csv_row(
                csv_dir / f"sr_seed{seed}_eval_reward_change.csv",
                {"seed": seed, "episode": episode, "condition": "reward_change", **reward_metrics},
            )
            append_csv_row(
                csv_dir / f"sr_seed{seed}_eval_transition_change.csv",
                {"seed": seed, "episode": episode, "condition": "transition_change", **transition_metrics},
            )

        if episode % save_interval == 0:
            ckpt_path = model_dir / f"sr_seed{seed}_episode{episode}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode": episode,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    train()