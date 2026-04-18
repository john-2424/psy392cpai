"""Linear-probe the CNN encoders of PPO, SR, and Replay for spatial content.

Addresses proposal M2 representation-probing deliverable. Cross-agent
comparison: if PPO/Replay encoders linearly predict position but SR does
not, that sharpens the deep-SF policy-extraction discussion.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from src.envs.gridworld import GridWorldEnv
from src.models.cnn_encoder import GridCNNEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "results" / "models"
OUT_CSV = PROJECT_ROOT / "results" / "csv" / "probe_results.csv"

SEEDS = [0, 1, 2]
AGENTS = ["ppo", "sr", "replay"]

AGENT_FEATURE_DIM = {"ppo": 128, "sr": 64, "replay": 128}
PPO_ENCODER_PREFIX = "module.0.module."  # policy encoder inside the TensorDictModule stack


def load_encoder(agent: str, seed: int) -> GridCNNEncoder:
    feature_dim = AGENT_FEATURE_DIM[agent]
    encoder = GridCNNEncoder(feature_dim=feature_dim)

    if agent == "ppo":
        ckpt = torch.load(MODEL_DIR / f"ppo_seed{seed}_stable_pretrain.pt",
                          map_location="cpu", weights_only=False)
        full_sd = ckpt["policy_state_dict"]
        enc_sd = {
            k[len(PPO_ENCODER_PREFIX):]: v
            for k, v in full_sd.items()
            if k.startswith(PPO_ENCODER_PREFIX)
        }
    elif agent == "sr":
        ckpt = torch.load(MODEL_DIR / f"sr_seed{seed}_best.pt",
                          map_location="cpu", weights_only=False)
        full_sd = ckpt["model_state_dict"]
        enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
    elif agent == "replay":
        ckpt = torch.load(MODEL_DIR / f"replay_seed{seed}_best.pt",
                          map_location="cpu", weights_only=False)
        full_sd = ckpt["model_state_dict"]
        enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}
    else:
        raise ValueError(agent)

    encoder.load_state_dict(enc_sd)
    encoder.eval()
    return encoder


def build_dataset() -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Enumerate (agent_pos, goal_condition) pairs and return features + targets.

    For each condition in {stable, reward_change} and each walkable cell as
    agent position, build the 3x8x8 observation directly via the env and
    record: agent row/col, goal col, Manhattan distance to goal.
    """
    obs_list: list[torch.Tensor] = []
    agent_rows: list[int] = []
    agent_cols: list[int] = []
    goal_cols: list[int] = []
    distances: list[int] = []

    for change_mode in ("stable", "reward_change"):
        env = GridWorldEnv(change_mode=change_mode, observation_mode="normal")
        env.reset()
        gr, gc = env.goal_pos
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if (r, c) in env.walls:
                    continue
                env.agent_pos = (r, c)
                obs = env._build_observation()
                obs_list.append(obs.clone())
                agent_rows.append(r)
                agent_cols.append(c)
                goal_cols.append(gc)
                distances.append(abs(r - gr) + abs(c - gc))

    X_obs = torch.stack(obs_list, dim=0)  # [N, 3, 8, 8]
    targets = {
        "agent_row": np.array(agent_rows),
        "agent_col": np.array(agent_cols),
        "goal_col": np.array(goal_cols),
        "manhattan": np.array(distances),
    }
    return X_obs, targets


def cv_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for train_idx, test_idx in kf.split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores)), float(np.std(scores))


def cv_accuracy(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = []
    for train_idx, test_idx in kf.split(X):
        clf = LogisticRegression(max_iter=1000).fit(X[train_idx], y[train_idx])
        scores.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    X_obs, targets = build_dataset()
    print(f"Dataset: {X_obs.shape[0]} states")

    rows = []
    for agent in AGENTS:
        for seed in SEEDS:
            encoder = load_encoder(agent, seed)
            with torch.no_grad():
                feats = encoder(X_obs).cpu().numpy()

            for name, y in targets.items():
                if name == "goal_col":
                    mean, std = cv_accuracy(feats, y)
                    metric = "accuracy"
                else:
                    mean, std = cv_r2(feats, y)
                    metric = "r2"
                rows.append(
                    {
                        "agent": agent,
                        "seed": seed,
                        "probe_target": name,
                        "metric": metric,
                        "score_mean": mean,
                        "score_std": std,
                        "feature_dim": AGENT_FEATURE_DIM[agent],
                        "n_samples": X_obs.shape[0],
                    }
                )
                print(f"{agent} seed={seed} {name:12s} {metric}={mean:+.3f} ± {std:.3f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
