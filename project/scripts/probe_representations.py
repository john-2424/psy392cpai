"""Linear-probe the CNN encoders of PPO, SR, Replay, and SR-AC for spatial content.

Addresses proposal M2 representation-probing deliverable. Cross-agent
comparison: if PPO/Replay encoders linearly predict position but SR does
not, that sharpens the deep-SF policy-extraction discussion.

For SR-AC (slide-15 future-work fix #3) the same shared encoder is probed,
but additionally the per-action features phi(s, a) -- concatenated across
actions -- are probed as a separate ``feature_kind="action_conditioned"``
row. The action-conditioned probe tests whether per-action features
recover column discrimination that the shared encoder loses.
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
# sr_ac is opt-in: only included when its checkpoints exist.
AGENTS = ["ppo", "sr", "replay", "sr_ac"]

AGENT_FEATURE_DIM = {"ppo": 128, "sr": 64, "replay": 128, "sr_ac": 64}
PPO_ENCODER_PREFIX = "module.0.module."  # policy encoder inside the TensorDictModule stack


def _checkpoint_path(agent: str, seed: int) -> Path:
    if agent == "ppo":
        return MODEL_DIR / f"ppo_seed{seed}_stable_pretrain.pt"
    if agent == "sr":
        return MODEL_DIR / f"sr_seed{seed}_best.pt"
    if agent == "replay":
        return MODEL_DIR / f"replay_seed{seed}_best.pt"
    if agent == "sr_ac":
        return MODEL_DIR / f"sr_ac_seed{seed}_best.pt"
    raise ValueError(agent)


def load_encoder(agent: str, seed: int) -> GridCNNEncoder:
    """Load the shared CNN encoder for any agent. SR-AC's shared encoder is
    inside the SRACNet (key prefix "encoder."), same as SR/Replay."""
    feature_dim = AGENT_FEATURE_DIM[agent]
    encoder = GridCNNEncoder(feature_dim=feature_dim)

    ckpt = torch.load(_checkpoint_path(agent, seed), map_location="cpu", weights_only=False)
    if agent == "ppo":
        full_sd = ckpt["policy_state_dict"]
        enc_sd = {
            k[len(PPO_ENCODER_PREFIX):]: v
            for k, v in full_sd.items()
            if k.startswith(PPO_ENCODER_PREFIX)
        }
    else:
        full_sd = ckpt["model_state_dict"]
        enc_sd = {k[len("encoder."):]: v for k, v in full_sd.items() if k.startswith("encoder.")}

    encoder.load_state_dict(enc_sd)
    encoder.eval()
    return encoder


def load_sr_ac_model(seed: int):
    """Return a fully-loaded SRACNet for action-conditioned probing."""
    from src.algorithms.sr_ac import SRACNet
    model = SRACNet(feature_dim=AGENT_FEATURE_DIM["sr_ac"], hidden_dim=64, n_actions=4)
    ckpt = torch.load(_checkpoint_path("sr_ac", seed), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


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
        # Skip silently if checkpoints aren't there (e.g. extension stage not run yet)
        missing = [s for s in SEEDS if not _checkpoint_path(agent, s).exists()]
        if missing:
            print(f"  [skip] {agent}: missing checkpoints for seeds {missing}")
            continue

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
                rows.append({
                    "agent": agent,
                    "seed": seed,
                    "feature_kind": "shared",
                    "probe_target": name,
                    "metric": metric,
                    "score_mean": mean,
                    "score_std": std,
                    "feature_dim": AGENT_FEATURE_DIM[agent],
                    "n_samples": X_obs.shape[0],
                })
                print(f"{agent}/shared    seed={seed} {name:12s} {metric}={mean:+.3f} ± {std:.3f}")

            # Action-conditioned probe (SR-AC only): probe the concatenated
            # phi(s, a) across actions. If action-conditioning recovers
            # column-axis discrimination, agent_col R^2 should rise here even
            # when it stays low on the shared encoder.
            if agent == "sr_ac":
                model = load_sr_ac_model(seed)
                with torch.no_grad():
                    phi_sa = model.action_features(X_obs)         # [N, A, d]
                    phi_ac = phi_sa.flatten(start_dim=1).cpu().numpy()
                for name, y in targets.items():
                    if name == "goal_col":
                        mean, std = cv_accuracy(phi_ac, y)
                        metric = "accuracy"
                    else:
                        mean, std = cv_r2(phi_ac, y)
                        metric = "r2"
                    rows.append({
                        "agent": agent,
                        "seed": seed,
                        "feature_kind": "action_conditioned",
                        "probe_target": name,
                        "metric": metric,
                        "score_mean": mean,
                        "score_std": std,
                        "feature_dim": phi_ac.shape[1],
                        "n_samples": X_obs.shape[0],
                    })
                    print(f"{agent}/per-action seed={seed} {name:12s} {metric}={mean:+.3f} ± {std:.3f}")

    if not rows:
        print("No probe rows produced; check that model checkpoints exist.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
