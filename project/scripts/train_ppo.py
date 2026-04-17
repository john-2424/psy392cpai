from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector

from src.algorithms.ppo_torchrl import build_ppo_components
from src.common.adaptation import (
    ADAPT_FIELDNAMES,
    CHANGED_CONDITIONS,
    append_csv_row,
    make_adaptation_envs,
    make_all_eval_envs,
    make_env,
    STABLE,
)


SEEDS = [0, 1, 2]
FRAMES_PER_BATCH = 512
TOTAL_FRAMES = 50000
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
LR = 3e-4
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 10  # batches
SAVE_INTERVAL = 20
NUM_ADAPT_BATCHES = 20
ADAPT_EVAL_INTERVAL = 2
N_EVAL_EPISODES = 20


def append_ppo_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate_policy(policy, env, n_episodes: int = 20, device: str = "cpu"):
    policy.eval()
    successes, returns, steps_list = 0, [], []
    for _ in range(n_episodes):
        td = env.reset()
        done, ep_return, ep_steps = False, 0.0, 0
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
        successes += int(env.agent_pos == env.goal_pos)
        returns.append(ep_return)
        steps_list.append(ep_steps)
    policy.train()
    return {
        "success_rate": successes / n_episodes,
        "avg_return": sum(returns) / len(returns),
        "avg_steps": sum(steps_list) / len(steps_list),
    }


def run_stable_training(seed: int, csv_dir: Path, model_dir: Path, device: str = "cpu"):
    torch.manual_seed(seed)

    train_env = make_env(STABLE, seed=seed)
    eval_envs = make_all_eval_envs(seed)

    components = build_ppo_components(train_env)
    policy = components["policy"].to(device)
    value_model = components["value_model"].to(device)
    advantage_module = components["advantage_module"]
    loss_module = components["loss_module"]

    optimizer = Adam(loss_module.parameters(), lr=LR)

    collector = SyncDataCollector(
        create_env_fn=lambda: make_env(STABLE, seed=seed),
        policy=policy,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        device=device,
    )

    train_csv = csv_dir / f"ppo_seed{seed}_train.csv"
    collected_frames, batch_idx = 0, 0

    for tensordict_data in collector:
        batch_idx += 1
        collected_frames += tensordict_data.numel()
        with torch.no_grad():
            advantage_module(tensordict_data)
        flat_data = tensordict_data.reshape(-1)
        last_loss_vals = None
        grad_norm_value = 0.0
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(flat_data.shape[0])
            for start in range(0, flat_data.shape[0], MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]
                minibatch = flat_data[idx]
                loss_vals = loss_module(minibatch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                grad_norm_value = float(grad_norm)
                last_loss_vals = loss_vals

        step_td = tensordict_data["next"] if "next" in tensordict_data.keys() else tensordict_data
        mean_reward = step_td["reward"].float().mean().item()
        done_rate = step_td["done"].float().mean().item()
        append_ppo_csv(
            train_csv,
            {
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
            },
        )

        if batch_idx % EVAL_INTERVAL == 0:
            metrics_by_cond = {
                name: evaluate_policy(policy, env, n_episodes=N_EVAL_EPISODES, device=device)
                for name, env in eval_envs.items()
            }
            print(
                f"[PPO seed={seed} batch={batch_idx}] "
                + " ".join(f"{n}={m['success_rate']:.2f}" for n, m in metrics_by_cond.items())
            )
            for name, m in metrics_by_cond.items():
                append_ppo_csv(
                    csv_dir / f"ppo_seed{seed}_eval_{name}.csv",
                    {
                        "seed": seed,
                        "batch_idx": batch_idx,
                        "collected_frames": collected_frames,
                        "condition": name,
                        "n_eval_episodes": N_EVAL_EPISODES,
                        **m,
                    },
                )

        if batch_idx % SAVE_INTERVAL == 0:
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "batch_idx": batch_idx,
                },
                model_dir / f"ppo_seed{seed}_batch{batch_idx}.pt",
            )

    collector.shutdown()

    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_idx": batch_idx,
        },
        model_dir / f"ppo_seed{seed}_stable_pretrain.pt",
    )
    return policy, value_model, advantage_module, loss_module, eval_envs


def run_ppo_adaptation(
    seed: int,
    policy,
    value_model,
    advantage_module,
    loss_module,
    condition,
    adapt_env_fn,
    eval_env,
    csv_dir: Path,
    device: str = "cpu",
) -> None:
    optimizer = Adam(loss_module.parameters(), lr=LR)
    collector = SyncDataCollector(
        create_env_fn=adapt_env_fn,
        policy=policy,
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=FRAMES_PER_BATCH * NUM_ADAPT_BATCHES,
        device=device,
    )
    adapt_csv = csv_dir / f"ppo_seed{seed}_adapt_{condition.name}_full.csv"

    batch_idx = 0
    for tensordict_data in collector:
        batch_idx += 1
        with torch.no_grad():
            advantage_module(tensordict_data)
        flat_data = tensordict_data.reshape(-1)
        last_loss = 0.0
        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(flat_data.shape[0])
            for start in range(0, flat_data.shape[0], MINIBATCH_SIZE):
                idx = perm[start:start + MINIBATCH_SIZE]
                minibatch = flat_data[idx]
                loss_vals = loss_module(minibatch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                last_loss = float(loss.item())

        step_td = tensordict_data["next"] if "next" in tensordict_data.keys() else tensordict_data
        ep_return = step_td["reward"].float().sum().item() / max(1, int(step_td["done"].float().sum().item()))
        ep_steps = int(tensordict_data.numel()) / max(1, int(step_td["done"].float().sum().item()))

        eval_m = {"success_rate": "", "avg_return": "", "avg_steps": ""}
        if batch_idx % ADAPT_EVAL_INTERVAL == 0:
            m = evaluate_policy(policy, eval_env, n_episodes=N_EVAL_EPISODES, device=device)
            eval_m = {"success_rate": m["success_rate"], "avg_return": m["avg_return"], "avg_steps": m["avg_steps"]}
            print(
                f"[PPO adapt seed={seed} cond={condition.name} batch={batch_idx}] "
                f"success={m['success_rate']:.2f}"
            )

        append_csv_row(
            adapt_csv,
            {
                "seed": seed,
                "agent": "ppo",
                "condition": condition.name,
                "variant": "full",
                "step": batch_idx,
                "episode_return": ep_return,
                "episode_steps": ep_steps,
                "loss": last_loss,
                "eval_success_rate": eval_m["success_rate"],
                "eval_avg_return": eval_m["avg_return"],
                "eval_avg_steps": eval_m["avg_steps"],
            },
            ADAPT_FIELDNAMES,
        )

    collector.shutdown()


def train():
    device = "cpu"
    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    model_dir = results_dir / "models"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f"\n========== PPO seed={seed} ==========")
        policy, value_model, advantage_module, loss_module, eval_envs = run_stable_training(
            seed, csv_dir, model_dir, device,
        )

        stable_policy = {k: v.clone() for k, v in policy.state_dict().items()}
        stable_value = {k: v.clone() for k, v in value_model.state_dict().items()}
        adapt_envs = make_adaptation_envs(seed)

        for cond in CHANGED_CONDITIONS:
            policy.load_state_dict(stable_policy)
            value_model.load_state_dict(stable_value)

            def make_this_env(c=cond, s=seed):
                return make_env(c, seed=s + 30)

            run_ppo_adaptation(
                seed, policy, value_model, advantage_module, loss_module,
                cond, make_this_env, eval_envs[cond.name], csv_dir, device=device,
            )


if __name__ == "__main__":
    train()
