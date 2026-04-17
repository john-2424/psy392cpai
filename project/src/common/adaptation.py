"""Shared utilities for few-shot adaptation experiments.

Defines the four post-change conditions (reward_change, transition_change,
obs_visual, obs_remap) and provides env factories + CSV helpers used by all
three agent training scripts.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

from src.envs.gridworld import GridWorldEnv


class Condition(NamedTuple):
    name: str
    change_mode: str
    observation_mode: str


# Five evaluation conditions: stable baseline + four perturbations.
# The first three correspond to standard revaluation paradigms
# (Momennejad et al. 2017). The last two correspond to rate remapping
# and global remapping (Sanders, Wilson & Gershman 2020).
STABLE = Condition("stable", "stable", "normal")
REWARD_CHANGE = Condition("reward_change", "reward_change", "normal")
TRANSITION_CHANGE = Condition("transition_change", "transition_change", "normal")
OBS_VISUAL = Condition("obs_visual", "stable", "visual_perturb")
OBS_REMAP = Condition("obs_remap", "stable", "obs_remap")

ALL_CONDITIONS = [STABLE, REWARD_CHANGE, TRANSITION_CHANGE, OBS_VISUAL, OBS_REMAP]
CHANGED_CONDITIONS = [REWARD_CHANGE, TRANSITION_CHANGE, OBS_VISUAL, OBS_REMAP]


def make_env(condition: Condition, seed: int, grid_size: int = 8, max_steps: int = 50) -> GridWorldEnv:
    return GridWorldEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        change_mode=condition.change_mode,
        observation_mode=condition.observation_mode,
        seed=seed,
    )


def make_all_eval_envs(seed: int) -> dict[str, GridWorldEnv]:
    """Five eval envs for a given base seed. Each condition gets a distinct
    derived seed so distractor masks don't collide."""
    offsets = {
        "stable": 1,
        "reward_change": 2,
        "transition_change": 3,
        "obs_visual": 4,
        "obs_remap": 5,
    }
    return {c.name: make_env(c, seed=seed + offsets[c.name]) for c in ALL_CONDITIONS}


def make_adaptation_envs(seed: int) -> dict[str, GridWorldEnv]:
    """Four adaptation envs -- separate from eval envs so adaptation learning
    does not touch the eval env state."""
    offsets = {
        "reward_change": 20,
        "transition_change": 21,
        "obs_visual": 22,
        "obs_remap": 23,
    }
    return {c.name: make_env(c, seed=seed + offsets[c.name]) for c in CHANGED_CONDITIONS}


def append_csv_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# Schema for adaptation CSVs. One row per adaptation step (episode for
# SR/Replay; batch for PPO). `eval_success_rate` etc. are filled in at eval
# checkpoints and left blank / NaN in between.
ADAPT_FIELDNAMES = [
    "seed",
    "agent",
    "condition",
    "variant",
    "step",
    "episode_return",
    "episode_steps",
    "loss",
    "eval_success_rate",
    "eval_avg_return",
    "eval_avg_steps",
]
