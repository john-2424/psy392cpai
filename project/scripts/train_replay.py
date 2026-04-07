from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.optim import Adam

from src.envs.gridworld import GridWorldEnv
from src.algorithms.replay_planning import (
    ReplayQNet,
    ReplayBuffer,
    ReplayTransition,
    select_action,
    compute_q_loss,
    hard_update,
    soft_update,
)


def append_csv_row(csv_path: Path, row: dict, fieldnames: list[str]):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def obs_from_td(td, device="cpu"):
    return td["observation"].to(device)


@torch.no_grad()
def evaluate_replay(model, env, n_episodes: int = 20, device: str = "cpu"):
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

            action_td = td.clone()
            action_td.set("action", torch.tensor(action, dtype=torch.long))
            step_td = env.step(action_td)
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

    num_episodes = 300
    max_steps_per_episode = 50
    gamma = 0.99
    lr = 1e-3
    tau = 0.01

    buffer_capacity = 10000
    batch_size = 32
    replay_updates_per_step = 2
    min_buffer_size = 100

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 300

    eval_interval = 25
    save_interval = 50

    train_fieldnames = [
        "seed",
        "episode",
        "global_step",
        "epsilon",
        "episode_return",
        "episode_steps",
        "buffer_size",
        "replay_updates_per_step",
        "q_loss",
        "mean_q",
        "mean_td_target",
    ]

    eval_fieldnames = [
        "seed",
        "episode",
        "condition",
        "success_rate",
        "avg_return",
        "avg_steps",
    ]

    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    model_dir = results_dir / "models"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="stable", seed=seed)
    eval_env_stable = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="stable", seed=seed + 1)
    eval_env_reward = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="reward_change", seed=seed + 2)
    eval_env_transition = GridWorldEnv(grid_size=8, max_steps=max_steps_per_episode, change_mode="transition_change", seed=seed + 3)

    model = ReplayQNet(feature_dim=128, hidden_dim=64, n_actions=4).to(device)
    target_model = ReplayQNet(feature_dim=128, hidden_dim=64, n_actions=4).to(device)
    hard_update(target_model, model)

    optimizer = Adam(model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    train_csv = csv_dir / "replay_seed0_train.csv"
    action_counts = [0, 0, 0, 0]

    global_step = 0
    best_stable_success = 0.0

    for episode in range(1, num_episodes + 1):
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_episodes),
        )

        td = train_env.reset()
        obs = obs_from_td(td, device=device)

        ep_return = 0.0
        ep_steps = 0
        last_metrics = {"q_loss": 0.0, "mean_q": 0.0, "mean_td_target": 0.0}

        for _ in range(max_steps_per_episode):
            global_step += 1

            if episode == 1 and ep_steps < 3:
                with torch.no_grad():
                    q = model(obs.unsqueeze(0))
                    print("obs shape:", obs.unsqueeze(0).shape)
                    print("q shape:", q.shape)
                    print("q values:", q.cpu().numpy())

            action = select_action(model, obs, epsilon=epsilon, device=device)
            action_counts[action] += 1

            action_td = td.clone()
            action_td.set("action", torch.tensor(action, dtype=torch.long))
            step_td = train_env.step(action_td)
            next_td = step_td["next"] if "next" in step_td.keys() else step_td

            next_obs = next_td["observation"].to(device)
            reward = float(next_td["reward"].item())
            done = float(bool(next_td["done"].item()))

            replay_buffer.add(
                ReplayTransition(
                    obs=obs.detach().cpu(),
                    action=action,
                    reward=reward,
                    next_obs=next_obs.detach().cpu(),
                    done=done,
                )
            )

            if len(replay_buffer) >= min_buffer_size:
                if episode == 1:
                    sample_obs, sample_actions, sample_rewards, sample_next_obs, sample_dones = replay_buffer.sample(
                        batch_size=batch_size,
                        device=device,
                    )
                    print("sample_obs shape:", sample_obs.shape)
                    print("sample_actions shape:", sample_actions.shape)
                    print("sample_rewards mean:", sample_rewards.mean().item())
                    print("sample_dones mean:", sample_dones.mean().item())

                for _ in range(replay_updates_per_step):
                    batch = replay_buffer.sample(batch_size=batch_size, device=device)
                    loss, metrics = compute_q_loss(
                        model,
                        target_model,
                        *batch,
                        gamma=gamma,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    soft_update(target_model, model, tau=tau)
                    last_metrics = metrics

            ep_return += reward
            ep_steps += 1

            obs = next_obs
            td = next_td

            if done:
                break

        row = {
            "seed": seed,
            "episode": episode,
            "global_step": global_step,
            "epsilon": epsilon,
            "episode_return": ep_return,
            "episode_steps": ep_steps,
            "buffer_size": len(replay_buffer),
            "replay_updates_per_step": replay_updates_per_step,
            **last_metrics,
        }
        append_csv_row(train_csv, row, train_fieldnames)

        if episode % eval_interval == 0:
            stable_metrics = evaluate_replay(model, eval_env_stable, n_episodes=20, device=device)
            reward_metrics = evaluate_replay(model, eval_env_reward, n_episodes=20, device=device)
            transition_metrics = evaluate_replay(model, eval_env_transition, n_episodes=20, device=device)

            print(
                f"[Episode {episode}] "
                f"stable={stable_metrics['success_rate']:.3f}, "
                f"reward={reward_metrics['success_rate']:.3f}, "
                f"transition={transition_metrics['success_rate']:.3f}"
            )
            print(f"[Episode {episode}] action counts: {action_counts}")
            print(f"[Episode {episode}] replay buffer size: {len(replay_buffer)}")
            action_counts = [0, 0, 0, 0]

            if stable_metrics["success_rate"] > best_stable_success:
                best_stable_success = stable_metrics["success_rate"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "target_model_state_dict": target_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "episode": episode,
                        "stable_success_rate": stable_metrics["success_rate"],
                    },
                    model_dir / "replay_seed0_best.pt",
                )

            append_csv_row(
                csv_dir / "replay_seed0_eval_stable.csv",
                {"seed": seed, "episode": episode, "condition": "stable", **stable_metrics},
                eval_fieldnames,
            )
            append_csv_row(
                csv_dir / "replay_seed0_eval_reward_change.csv",
                {"seed": seed, "episode": episode, "condition": "reward_change", **reward_metrics},
                eval_fieldnames,
            )
            append_csv_row(
                csv_dir / "replay_seed0_eval_transition_change.csv",
                {"seed": seed, "episode": episode, "condition": "transition_change", **transition_metrics},
                eval_fieldnames,
            )

        if episode % save_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode": episode,
                },
                model_dir / f"replay_seed0_episode{episode}.pt",
            )


if __name__ == "__main__":
    train()