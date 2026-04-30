"""Compare slide-15 future-work extensions against the baseline runs.

Reads CSVs produced by:
  - baseline:      train_ppo / train_sr / train_replay
  - extension #1:  train_ppo.train_adapt_only(0.1, "full_lr10x")
  - extension #2:  train_sr.train_with_qmargin(...)
  - extension #3:  train_sr_ac.train(...)

Writes:
  - results/figures/extensions_compare.png
  - results/figures/extensions_auc_bars.png
  - results/extensions_summary.csv

Usage:
    cd project
    PYTHONPATH=. python scripts/analyze_extensions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT / "results" / "csv"
FIG_DIR = PROJECT / "results" / "figures"
OUT_CSV = PROJECT / "results" / "extensions_summary.csv"

SEEDS = [0, 1, 2]
CHANGED = ["reward_change", "transition_change", "obs_visual", "obs_remap"]

# (label, glob pattern). Each label becomes a row in the summary.
RUNS = [
    ("PPO baseline",          "ppo_seed{seed}_adapt_{cond}_full.csv"),
    ("PPO LR/10 (fix #1)",    "ppo_seed{seed}_adapt_{cond}_full_lr10x.csv"),
    ("SR-full baseline",      "sr_seed{seed}_adapt_{cond}_full.csv"),
    ("SR w-only baseline",    "sr_seed{seed}_adapt_{cond}_wonly.csv"),
    ("SR-full Q-margin (#2)", "sr_seed{seed}_adapt_{cond}_full_qmargin.csv"),
    ("SR w-only Q-margin",    "sr_seed{seed}_adapt_{cond}_wonly_qmargin.csv"),
    ("SR-AC full (fix #3)",   "sr_ac_seed{seed}_adapt_{cond}_full.csv"),
    ("SR-AC w-only (#3)",     "sr_ac_seed{seed}_adapt_{cond}_wonly.csv"),
    ("Replay baseline",       "replay_seed{seed}_adapt_{cond}_full.csv"),
]


def load(label, pattern):
    frames = []
    for s in SEEDS:
        for cond in CHANGED:
            p = CSV_DIR / pattern.format(seed=s, cond=cond)
            if p.exists():
                df = pd.read_csv(p)
                df["seed"] = s
                df["run_label"] = label
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def normalized_auc(steps, ys):
    if len(steps) < 2:
        return np.nan
    span = steps[-1] - steps[0]
    if span <= 0:
        return np.nan
    return float(np.trapz(ys, steps) / span)


def t_thr(steps, ys, thr=0.5):
    above = np.where(ys >= thr)[0]
    return float(steps[above[0]]) if above.size else np.nan


def metrics_per_run(df):
    """Compute AUC, t_0.5, and end-of-phase success per (cond, seed),
    then aggregate per condition."""
    out = []
    for (cond, seed), g in df.groupby(["condition", "seed"]):
        g = g[g["eval_success_rate"].notna()].sort_values("step")
        if g.empty:
            continue
        steps = g["step"].to_numpy(dtype=float)
        ys = g["eval_success_rate"].to_numpy(dtype=float)
        last_n = ys[-3:].mean() if len(ys) >= 3 else float(ys.mean())
        out.append(dict(
            condition=cond,
            seed=seed,
            auc=normalized_auc(steps, ys),
            t_thr=t_thr(steps, ys, 0.5),
            end_success=last_n,
            peak=float(ys.max()),
        ))
    return pd.DataFrame(out)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_run = {}
    for label, pat in RUNS:
        df = load(label, pat)
        if df is None:
            print(f"  [skip] {label}: no CSVs found ({pat})")
            continue
        m = metrics_per_run(df)
        per_run[label] = m
        agg = m.groupby("condition").agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            t_thr_med=("t_thr", "median"),
            end_mean=("end_success", "mean"),
            end_std=("end_success", "std"),
            peak_mean=("peak", "mean"),
        ).reset_index()
        agg["run"] = label
        summary_rows.append(agg)
        print(f"  [ok]   {label}: {len(m)} (cond, seed) rows")

    if not summary_rows:
        print("No extension CSVs found yet. Run training first, then re-run this script.")
        return

    summary = pd.concat(summary_rows, ignore_index=True)
    summary = summary[["run", "condition", "auc_mean", "auc_std",
                       "t_thr_med", "end_mean", "end_std", "peak_mean"]]
    summary.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(summary.to_string(index=False))

    # Plotting (only if matplotlib is available, which it is in this env)
    import matplotlib.pyplot as plt

    # ----- 1. Adaptation curves: PPO baseline vs PPO LR/10 on each cond -----
    if all(lbl in per_run for lbl in ("PPO baseline", "PPO LR/10 (fix #1)")):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
        for ax, cond in zip(axes, CHANGED):
            for label, color in [
                ("PPO baseline", "#E15759"),
                ("PPO LR/10 (fix #1)", "#2C3E91"),
            ]:
                df = load(label, dict(RUNS)[label])
                sub = df[(df["condition"] == cond) & df["eval_success_rate"].notna()]
                if sub.empty:
                    continue
                agg = sub.groupby("step")["eval_success_rate"].agg(["mean", "std"]).reset_index()
                ax.plot(agg["step"], agg["mean"], color=color, linewidth=1.8, label=label)
                ax.fill_between(agg["step"], agg["mean"] - agg["std"].fillna(0),
                                agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.18)
            ax.set_title(cond, fontsize=10)
            ax.set_xlabel("Adapt batch")
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("Greedy success rate")
                ax.legend(fontsize=8, loc="lower right")
        fig.suptitle("Fix #1: PPO adaptation with adapt-LR scaled by 0.1 (mean ± std, 3 seeds)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "extensions_ppo_lr10x.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'extensions_ppo_lr10x.png'}")

    # ----- 2. SR family AUC on reward_change (the key Momennejad signal) -----
    rc = summary[summary["condition"] == "reward_change"].copy()
    if not rc.empty:
        order = [
            "PPO baseline", "PPO LR/10 (fix #1)",
            "SR-full baseline", "SR w-only baseline",
            "SR-full Q-margin (#2)", "SR w-only Q-margin",
            "SR-AC full (fix #3)", "SR-AC w-only (#3)",
            "Replay baseline",
        ]
        rc = rc.set_index("run").reindex([r for r in order if r in rc["run"].values]).reset_index()

        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(len(rc))
        bars = ax.bar(x, rc["auc_mean"], yerr=rc["auc_std"].fillna(0),
                      color=["#E15759", "#2C3E91",
                             "#4E79A7", "#76B7B2",
                             "#F28E2B", "#FFB74D",
                             "#9C755F", "#BAB0AC",
                             "#59A14F"][:len(rc)],
                      capsize=4, edgecolor="white")
        for b, v in zip(bars, rc["auc_mean"]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(rc["run"], rotation=18, ha="right", fontsize=8)
        ax.set_ylabel("AUC on reward_change (norm., higher=better)")
        ax.set_ylim(0, 1.1)
        ax.set_title("Fix #2 + #3: SR-family AUC on reward_change vs baseline",
                     fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "extensions_auc_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {FIG_DIR / 'extensions_auc_bars.png'}")

    # ----- 3. SR-AC vs SR-full vs Replay across all four conds -----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    palette = {
        "SR-full baseline":    "#4E79A7",
        "SR-AC full (fix #3)": "#9C755F",
        "Replay baseline":     "#59A14F",
    }
    for ax, cond in zip(axes, CHANGED):
        for label, color in palette.items():
            if label not in dict(RUNS):
                continue
            df = load(label, dict(RUNS)[label])
            if df is None:
                continue
            sub = df[(df["condition"] == cond) & df["eval_success_rate"].notna()]
            if sub.empty:
                continue
            agg = sub.groupby("step")["eval_success_rate"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["step"], agg["mean"], color=color, linewidth=1.8, label=label)
            ax.fill_between(agg["step"], agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0), color=color, alpha=0.18)
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel("Adapt episode")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Greedy success rate")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Fix #3: SR-AC (action-conditioned phi) vs SR-full and Replay",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "extensions_sr_ac_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'extensions_sr_ac_compare.png'}")


if __name__ == "__main__":
    main()
