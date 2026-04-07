from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _save_line_plot(
    dfs: dict[str, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    x_col: str = "collected_frames",
) -> None:
    plt.figure(figsize=(8, 5))
    for label, df in dfs.items():
        if df.empty:
            continue
        plt.plot(df[x_col], df[y_col], label=label)

    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_training_plot(
    train_df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
    x_col: str = "collected_frames",
) -> None:
    if y_col not in train_df.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_df[x_col], train_df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_ppo_plots(seed: int = 0, results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    csv_dir = results_path / "csv"
    fig_dir = results_path / "figures"

    train_df = _read_csv(csv_dir / f"ppo_seed{seed}_train.csv")
    stable_df = _read_csv(csv_dir / f"ppo_seed{seed}_eval_stable.csv")
    reward_df = _read_csv(csv_dir / f"ppo_seed{seed}_eval_reward_change.csv")
    transition_df = _read_csv(csv_dir / f"ppo_seed{seed}_eval_transition_change.csv")

    eval_dfs = {
        "stable": stable_df,
        "reward_change": reward_df,
        "transition_change": transition_df,
    }

    _save_line_plot(
        eval_dfs,
        "success_rate",
        f"PPO Seed {seed}: Success Rate Across Conditions",
        "success_rate",
        fig_dir / f"ppo_seed{seed}_condition_success_comparison.png",
    )
    _save_line_plot(
        eval_dfs,
        "avg_return",
        f"PPO Seed {seed}: Average Return Across Conditions",
        "avg_return",
        fig_dir / f"ppo_seed{seed}_condition_return_comparison.png",
    )
    _save_line_plot(
        eval_dfs,
        "avg_steps",
        f"PPO Seed {seed}: Average Steps Across Conditions",
        "avg_steps",
        fig_dir / f"ppo_seed{seed}_condition_steps_comparison.png",
    )
    _save_line_plot(
        {"stable": stable_df},
        "success_rate",
        f"PPO Seed {seed}: Stable Success Rate",
        "success_rate",
        fig_dir / f"ppo_seed{seed}_stable_success.png",
    )
    _save_line_plot(
        {"stable": stable_df},
        "avg_return",
        f"PPO Seed {seed}: Stable Average Return",
        "avg_return",
        fig_dir / f"ppo_seed{seed}_stable_return.png",
    )
    _save_training_plot(
        train_df,
        "mean_batch_reward",
        f"PPO Seed {seed}: Mean Batch Reward",
        "mean_batch_reward",
        fig_dir / f"ppo_seed{seed}_train_mean_batch_reward.png",
    )
    _save_training_plot(
        train_df,
        "total_loss",
        f"PPO Seed {seed}: Total Loss",
        "total_loss",
        fig_dir / f"ppo_seed{seed}_train_total_loss.png",
    )

    print(f"Saved plots to: {fig_dir}")


if __name__ == "__main__":
    make_ppo_plots(seed=0, results_dir="results")