from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")


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
        if df.empty or y_col not in df.columns or x_col not in df.columns:
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
    if y_col not in train_df.columns or x_col not in train_df.columns:
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


def make_agent_plots(agent_name: str = "ppo", seed: int = 0, results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    csv_dir = results_path / "csv"
    fig_dir = results_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_csv = csv_dir / f"{agent_name}_seed{seed}_train.csv"
    stable_csv = csv_dir / f"{agent_name}_seed{seed}_eval_stable.csv"
    reward_csv = csv_dir / f"{agent_name}_seed{seed}_eval_reward_change.csv"
    transition_csv = csv_dir / f"{agent_name}_seed{seed}_eval_transition_change.csv"

    train_df = _read_csv(train_csv)
    stable_df = _read_csv(stable_csv)
    reward_df = _read_csv(reward_csv)
    transition_df = _read_csv(transition_csv)

    eval_dfs = {
        "stable": stable_df,
        "reward_change": reward_df,
        "transition_change": transition_df,
    }

    x_col = "collected_frames" if "collected_frames" in stable_df.columns else "episode"
    train_x_col = "collected_frames" if "collected_frames" in train_df.columns else "episode"

    _save_line_plot(
        dfs=eval_dfs,
        y_col="success_rate",
        title=f"{agent_name.upper()} Seed {seed}: Success Rate Across Conditions",
        ylabel="success_rate",
        out_path=fig_dir / f"{agent_name}_seed{seed}_condition_success_comparison.png",
        x_col=x_col,
    )

    _save_line_plot(
        dfs=eval_dfs,
        y_col="avg_return",
        title=f"{agent_name.upper()} Seed {seed}: Average Return Across Conditions",
        ylabel="avg_return",
        out_path=fig_dir / f"{agent_name}_seed{seed}_condition_return_comparison.png",
        x_col=x_col,
    )

    _save_line_plot(
        dfs=eval_dfs,
        y_col="avg_steps",
        title=f"{agent_name.upper()} Seed {seed}: Average Steps Across Conditions",
        ylabel="avg_steps",
        out_path=fig_dir / f"{agent_name}_seed{seed}_condition_steps_comparison.png",
        x_col=x_col,
    )

    _save_line_plot(
        dfs={"stable": stable_df},
        y_col="success_rate",
        title=f"{agent_name.upper()} Seed {seed}: Stable Success Rate",
        ylabel="success_rate",
        out_path=fig_dir / f"{agent_name}_seed{seed}_stable_success.png",
        x_col=x_col,
    )

    _save_line_plot(
        dfs={"stable": stable_df},
        y_col="avg_return",
        title=f"{agent_name.upper()} Seed {seed}: Stable Average Return",
        ylabel="avg_return",
        out_path=fig_dir / f"{agent_name}_seed{seed}_stable_return.png",
        x_col=x_col,
    )

    if "mean_batch_reward" in train_df.columns:
        _save_training_plot(
            train_df=train_df,
            y_col="mean_batch_reward",
            title=f"{agent_name.upper()} Seed {seed}: Mean Batch Reward",
            ylabel="mean_batch_reward",
            out_path=fig_dir / f"{agent_name}_seed{seed}_train_mean_batch_reward.png",
            x_col=train_x_col,
        )

    if "total_loss" in train_df.columns:
        _save_training_plot(
            train_df=train_df,
            y_col="total_loss",
            title=f"{agent_name.upper()} Seed {seed}: Total Loss",
            ylabel="total_loss",
            out_path=fig_dir / f"{agent_name}_seed{seed}_train_total_loss.png",
            x_col=train_x_col,
        )

    if "episode_return" in train_df.columns:
        _save_training_plot(
            train_df=train_df,
            y_col="episode_return",
            title=f"{agent_name.upper()} Seed {seed}: Episode Return",
            ylabel="episode_return",
            out_path=fig_dir / f"{agent_name}_seed{seed}_train_episode_return.png",
            x_col=train_x_col,
        )

    if "reward_weights_norm" in train_df.columns:
        _save_training_plot(
            train_df=train_df,
            y_col="reward_weights_norm",
            title=f"{agent_name.upper()} Seed {seed}: Reward Weights Norm",
            ylabel="reward_weights_norm",
            out_path=fig_dir / f"{agent_name}_seed{seed}_train_reward_weights_norm.png",
            x_col=train_x_col,
        )

    if "buffer_size" in train_df.columns:
        _save_training_plot(
            train_df=train_df,
            y_col="buffer_size",
            title=f"{agent_name.upper()} Seed {seed}: Replay Buffer Size",
            ylabel="buffer_size",
            out_path=fig_dir / f"{agent_name}_seed{seed}_train_buffer_size.png",
            x_col=train_x_col,
        )

    print(f"Saved plots to: {fig_dir}")


def make_sr_reward_reval_plot(seed: int = 0, results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    csv_dir = results_path / "csv"
    fig_dir = results_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_csv = csv_dir / f"sr_seed{seed}_eval_reward_reval.csv"
    df = _read_csv(eval_csv)

    x_col = "reval_episode"

    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df["success_rate"])
    plt.xlabel("reval_episode")
    plt.ylabel("success_rate")
    plt.title(f"SR Seed {seed}: Reward Revaluation Success")
    plt.tight_layout()
    plt.savefig(fig_dir / f"sr_seed{seed}_reward_revaluation_success.png", dpi=200)
    plt.close()

    if "avg_return" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], df["avg_return"])
        plt.xlabel("reval_episode")
        plt.ylabel("avg_return")
        plt.title(f"SR Seed {seed}: Reward Revaluation Return")
        plt.tight_layout()
        plt.savefig(fig_dir / f"sr_seed{seed}_reward_revaluation_return.png", dpi=200)
        plt.close()


def make_all_smoke_test_plots(results_dir: str = "results") -> None:
    for agent_name in ["ppo", "sr", "replay"]:
        try:
            make_agent_plots(agent_name=agent_name, seed=0, results_dir=results_dir)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Skipping {agent_name}: {e}")

    try:
        make_sr_reward_reval_plot(seed=0, results_dir=results_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Skipping SR reward revaluation plot: {e}")


if __name__ == "__main__":
    make_all_smoke_test_plots(results_dir="results")