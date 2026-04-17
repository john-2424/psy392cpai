from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.algorithms.sr import (
    SRNet,
    SRBatch,
    compute_sr_loss,
    freeze_encoder_and_sr_head,
    hard_update,
    select_action,
    soft_update,
    unfreeze_all,
)
from src.common.adaptation import (
    ADAPT_FIELDNAMES,
    CHANGED_CONDITIONS,
    REWARD_CHANGE,
    append_csv_row,
    make_adaptation_envs,
    make_all_eval_envs,
    make_env,
    STABLE,
)
from src.envs.gridworld import GridWorldEnv


SEEDS = [0, 1, 2]
NUM_EPISODES = 300
NUM_ADAPT_EPISODES = 60
MAX_STEPS_PER_EPISODE = 50
GAMMA = 0.95
LR = 3e-4
TAU = 0.05
BUFFER_CAPACITY = 5000
BATCH_SIZE = 32
MIN_BUFFER_SIZE = 100
REPLAY_UPDATES_PER_STEP = 2
EPS_START, EPS_END, EPS_DECAY_EPS = 1.0, 0.05, 200
EVAL_INTERVAL = 25
ADAPT_EVAL_INTERVAL = 5
N_EVAL_EPISODES = 20


@dataclass
class SRReplayTransition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    done: float


class SRReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, transition: SRReplayTransition) -> None:
        self.buffer.append(transition)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, device: str = "cpu") -> SRBatch:
        batch = random.sample(self.buffer, batch_size)
        obs = torch.stack([t.obs for t in batch]).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_obs = torch.stack([t.next_obs for t in batch]).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return SRBatch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)


def obs_from_td(td, device: str = "cpu") -> torch.Tensor:
    return td["observation"].to(device)


def save_sr_checkpoint(model, target_model, optimizer, tag: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tag": tag,
        },
        out_dir / f"{tag}.pt",
    )


def compute_reward_only_loss(model: SRNet, obs: torch.Tensor, rewards: torch.Tensor):
    with torch.no_grad():
        phi = model.encode(obs)
    pred_reward = torch.einsum("bd,d->b", phi, model.reward_weights)
    return F.mse_loss(pred_reward, rewards)


@torch.no_grad()
def evaluate_sr(model, env, n_episodes: int = 20, device: str = "cpu"):
    model.eval()
    successes, returns, steps_list = 0, [], []
    for _ in range(n_episodes):
        td = env.reset()
        obs = obs_from_td(td, device=device)
        done, ep_return, ep_steps = False, 0.0, 0
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
        successes += int(env.agent_pos == env.goal_pos)
        returns.append(ep_return)
        steps_list.append(ep_steps)
    model.train()
    return {
        "success_rate": successes / n_episodes,
        "avg_return": sum(returns) / len(returns),
        "avg_steps": sum(steps_list) / len(steps_list),
    }


def run_stable_training(seed: int, csv_dir: Path, model_dir: Path, device: str = "cpu"):
    torch.manual_seed(seed)
    random.seed(seed)

    train_env = make_env(STABLE, seed=seed)
    eval_envs = make_all_eval_envs(seed)

    model = SRNet(feature_dim=64, hidden_dim=64, n_actions=4).to(device)
    target_model = SRNet(feature_dim=64, hidden_dim=64, n_actions=4).to(device)
    hard_update(target_model, model)
    unfreeze_all(model)
    unfreeze_all(target_model)

    optimizer = Adam(model.parameters(), lr=LR)
    replay_buffer = SRReplayBuffer(capacity=BUFFER_CAPACITY)

    train_csv = csv_dir / f"sr_seed{seed}_train.csv"
    train_fieldnames = [
        "seed", "episode", "global_step", "epsilon",
        "episode_return", "episode_steps", "buffer_size",
        "reward_weights_norm", "sr_loss", "reward_loss", "total_loss", "mean_q",
    ]
    eval_fieldnames = ["seed", "episode", "condition", "success_rate", "avg_return", "avg_steps"]

    global_step = 0
    for episode in range(1, NUM_EPISODES + 1):
        epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (episode / EPS_DECAY_EPS),
        )
        td = train_env.reset()
        obs = obs_from_td(td, device=device)
        ep_return, ep_steps = 0.0, 0
        last_metrics = {"sr_loss": 0.0, "reward_loss": 0.0, "total_loss": 0.0, "mean_q": 0.0}

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

            replay_buffer.add(SRReplayTransition(
                obs=obs.detach().cpu(),
                action=action,
                reward=reward,
                next_obs=next_obs.detach().cpu(),
                done=done,
            ))

            if len(replay_buffer) >= MIN_BUFFER_SIZE:
                for _ in range(REPLAY_UPDATES_PER_STEP):
                    batch = replay_buffer.sample(batch_size=BATCH_SIZE, device=device)
                    loss, metrics = compute_sr_loss(model, target_model, batch, gamma=GAMMA)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    soft_update(target_model, model, tau=TAU)
                    last_metrics = metrics

            ep_return += reward
            ep_steps += 1
            obs = next_obs
            td = next_td
            if bool(done):
                break

        append_csv_row(
            train_csv,
            {
                "seed": seed,
                "episode": episode,
                "global_step": global_step,
                "epsilon": epsilon,
                "episode_return": ep_return,
                "episode_steps": ep_steps,
                "buffer_size": len(replay_buffer),
                "reward_weights_norm": float(model.reward_weights.norm().item()),
                **last_metrics,
            },
            train_fieldnames,
        )

        if episode % EVAL_INTERVAL == 0:
            metrics_by_cond = {
                name: evaluate_sr(model, env, n_episodes=N_EVAL_EPISODES, device=device)
                for name, env in eval_envs.items()
            }
            print(
                f"[SR seed={seed} ep={episode}] "
                + " ".join(f"{n}={m['success_rate']:.2f}" for n, m in metrics_by_cond.items())
            )
            for name, m in metrics_by_cond.items():
                append_csv_row(
                    csv_dir / f"sr_seed{seed}_eval_{name}.csv",
                    {"seed": seed, "episode": episode, "condition": name, **m},
                    eval_fieldnames,
                )

    save_sr_checkpoint(model, target_model, optimizer, tag=f"sr_seed{seed}_stable_pretrain", out_dir=model_dir)
    return model, target_model, eval_envs


def run_sr_adaptation(
    seed: int,
    model: SRNet,
    target_model: SRNet,
    condition,
    adapt_env,
    eval_env,
    csv_dir: Path,
    variant: str = "full",
    device: str = "cpu",
) -> None:
    """Continue training from the stable pretrain for NUM_ADAPT_EPISODES on the changed env.

    variant = "full": all params trainable (standard fine-tune).
    variant = "wonly": only reward_weights trainable (Momennejad-style
        revaluation; SR bellman equation assumes dynamics unchanged).
    """
    if variant == "full":
        unfreeze_all(model)
        unfreeze_all(target_model)
        optimizer = Adam(model.parameters(), lr=LR)
    elif variant == "wonly":
        freeze_encoder_and_sr_head(model)
        freeze_encoder_and_sr_head(target_model)
        optimizer = Adam([model.reward_weights], lr=LR)
    else:
        raise ValueError(variant)

    replay_buffer = SRReplayBuffer(capacity=BUFFER_CAPACITY)
    adapt_csv = csv_dir / f"sr_seed{seed}_adapt_{condition.name}_{variant}.csv"

    for adapt_ep in range(1, NUM_ADAPT_EPISODES + 1):
        epsilon = 0.1
        td = adapt_env.reset()
        obs = obs_from_td(td, device=device)
        ep_return, ep_steps, last_loss = 0.0, 0, 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = select_action(model, obs, epsilon=epsilon, device=device)
            action_td = td.clone()
            action_td.set("action", torch.tensor(action, dtype=torch.long))
            step_td = adapt_env.step(action_td)
            next_td = step_td["next"] if "next" in step_td.keys() else step_td
            next_obs = next_td["observation"].to(device)
            reward = float(next_td["reward"].item())
            done = float(bool(next_td["done"].item()))

            replay_buffer.add(SRReplayTransition(
                obs=obs.detach().cpu(),
                action=action,
                reward=reward,
                next_obs=next_obs.detach().cpu(),
                done=done,
            ))

            if len(replay_buffer) >= MIN_BUFFER_SIZE:
                batch = replay_buffer.sample(batch_size=BATCH_SIZE, device=device)
                if variant == "full":
                    loss, _ = compute_sr_loss(model, target_model, batch, gamma=GAMMA)
                else:
                    loss = compute_reward_only_loss(model, batch.obs, batch.rewards)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if variant == "full":
                    soft_update(target_model, model, tau=TAU)
                last_loss = float(loss.item())

            ep_return += reward
            ep_steps += 1
            obs = next_obs
            td = next_td
            if bool(done):
                break

        eval_m = {"success_rate": "", "avg_return": "", "avg_steps": ""}
        if adapt_ep % ADAPT_EVAL_INTERVAL == 0:
            m = evaluate_sr(model, eval_env, n_episodes=N_EVAL_EPISODES, device=device)
            eval_m = {
                "success_rate": m["success_rate"],
                "avg_return": m["avg_return"],
                "avg_steps": m["avg_steps"],
            }
            print(
                f"[SR adapt seed={seed} cond={condition.name}/{variant} ep={adapt_ep}] "
                f"success={m['success_rate']:.2f}"
            )

        append_csv_row(
            adapt_csv,
            {
                "seed": seed,
                "agent": "sr",
                "condition": condition.name,
                "variant": variant,
                "step": adapt_ep,
                "episode_return": ep_return,
                "episode_steps": ep_steps,
                "loss": last_loss,
                "eval_success_rate": eval_m["success_rate"],
                "eval_avg_return": eval_m["avg_return"],
                "eval_avg_steps": eval_m["avg_steps"],
            },
            ADAPT_FIELDNAMES,
        )


def train():
    device = "cpu"
    results_dir = Path("results")
    csv_dir = results_dir / "csv"
    model_dir = results_dir / "models"
    csv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print(f"\n========== SR seed={seed} ==========")
        model, target_model, eval_envs = run_stable_training(seed, csv_dir, model_dir, device)

        # Snapshot stable pretrain state; reload before each adaptation so
        # adaptation phases do not contaminate each other.
        stable_state = {
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "target": {k: v.clone() for k, v in target_model.state_dict().items()},
        }

        adapt_envs = make_adaptation_envs(seed)

        for cond in CHANGED_CONDITIONS:
            model.load_state_dict(stable_state["model"])
            target_model.load_state_dict(stable_state["target"])
            run_sr_adaptation(
                seed, model, target_model, cond, adapt_envs[cond.name],
                eval_envs[cond.name], csv_dir, variant="full", device=device,
            )

        # Extra Momennejad-style reward-only revaluation on reward_change.
        model.load_state_dict(stable_state["model"])
        target_model.load_state_dict(stable_state["target"])
        run_sr_adaptation(
            seed, model, target_model, REWARD_CHANGE, adapt_envs["reward_change"],
            eval_envs["reward_change"], csv_dir, variant="wonly", device=device,
        )


if __name__ == "__main__":
    train()
