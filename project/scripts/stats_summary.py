"""Per-claim statistical analyses for the paper (n=3 seeds).

Honest framing for small-n: every test reported here is reported alongside
its raw per-seed paired differences, an effect size, and the lowest
achievable p-value at this n. With n=3 seeds the smallest possible
two-sided p from a sign test is 0.25 (3-of-3 in one direction); the
smallest from a paired Wilcoxon signed-rank is 0.25 as well. We therefore
report direction + effect size + lower-bound p, and let the reader judge.

Bootstrap CIs on cross-seed means are computed by resampling the 3 seeds
with replacement (BCa not used; basic percentile interval). With n=3 the
CI is wide and approximate, but it gives the rubric what it asks for and
makes the n=3 limitation visible to the reader.

Outputs:
    results/stats_summary.csv         -- one row per claim
    stdout                            -- formatted table the paper can quote
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats

PROJECT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT / "results" / "csv"
OUT_CSV = PROJECT / "results" / "stats_summary.csv"

SEEDS = [0, 1, 2]
N_BOOT = 5000
RNG = np.random.default_rng(seed=42)


def per_seed_metric(file_pattern: str, metric: str = "auc") -> dict[int, float]:
    """For a given (agent, variant, condition), compute per-seed AUC or
    end-of-phase greedy success from the adaptation CSVs.
    metric in {"auc", "end3"}.
    """
    out = {}
    for s in SEEDS:
        p = CSV_DIR / file_pattern.format(seed=s)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        ev = df[df["eval_success_rate"].notna()].sort_values("step")
        if ev.empty:
            out[s] = 0.0
            continue
        steps = ev["step"].to_numpy(dtype=float)
        ys = ev["eval_success_rate"].to_numpy(dtype=float)
        if metric == "auc":
            span = steps[-1] - steps[0]
            out[s] = float(np.trapz(ys, steps) / span) if span > 0 else float(ys.mean())
        elif metric == "end3":
            out[s] = float(ys[-3:].mean()) if len(ys) >= 3 else float(ys.mean())
        else:
            raise ValueError(metric)
    return out


def bootstrap_mean_ci(values: list[float], n_boot: int = N_BOOT, alpha: float = 0.05):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, len(arr), size=len(arr))
        boot_means[b] = arr[idx].mean()
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def cohens_d_paired(a: list[float], b: list[float]) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    diffs = np.asarray(a) - np.asarray(b)
    sd = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    if sd == 0:
        return float("inf") if diffs.mean() != 0 else 0.0
    return float(diffs.mean() / sd)


def sign_test(a: list[float], b: list[float]) -> tuple[int, int, float]:
    """Two-sided sign test: count of (a > b), (a < b), and binomial p."""
    diffs = np.asarray(a) - np.asarray(b)
    pos = int(np.sum(diffs > 0))
    neg = int(np.sum(diffs < 0))
    n = pos + neg  # ties dropped
    if n == 0:
        return pos, neg, 1.0
    # two-sided binomial
    p = scistats.binomtest(min(pos, neg), n, p=0.5, alternative="two-sided").pvalue
    return pos, neg, float(p)


def wilcoxon(a: list[float], b: list[float]):
    diffs = np.asarray(a) - np.asarray(b)
    diffs = diffs[diffs != 0]
    if len(diffs) < 1:
        return None
    try:
        # n=3 with ties triggers a warning in some scipy versions; suppress
        res = scistats.wilcoxon(diffs, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------
# Claim definitions: (label, A pattern, B pattern, metric, claim direction)
# A is the variant claimed to be better than B.
# ---------------------------------------------------------------------------

CLAIMS = [
    # H1 (original): SR-wonly > PPO baseline on reward_change (AUC)
    ("H1: SR-wonly AUC > PPO baseline AUC on reward_change",
     "sr_seed{seed}_adapt_reward_change_wonly.csv",
     "ppo_seed{seed}_adapt_reward_change_full.csv",
     "auc"),
    # H2: Replay > SR-full on transition_change (AUC)
    ("H2: Replay AUC > SR-full AUC on transition_change",
     "replay_seed{seed}_adapt_transition_change_full.csv",
     "sr_seed{seed}_adapt_transition_change_full.csv",
     "auc"),
    # Fix #1: PPO LR/10 > PPO baseline on obs_visual (end-of-phase)
    ("Fix #1: PPO LR/10 end > PPO baseline end on obs_visual",
     "ppo_seed{seed}_adapt_obs_visual_full_lr10x.csv",
     "ppo_seed{seed}_adapt_obs_visual_full.csv",
     "end3"),
    # Fix #1 also via AUC
    ("Fix #1: PPO LR/10 AUC > PPO baseline AUC on obs_visual",
     "ppo_seed{seed}_adapt_obs_visual_full_lr10x.csv",
     "ppo_seed{seed}_adapt_obs_visual_full.csv",
     "auc"),
    # Fix #3: SR-AC > SR-full on reward_change (end-of-phase)
    ("Fix #3: SR-AC end > SR-full end on reward_change",
     "sr_ac_seed{seed}_adapt_reward_change_full.csv",
     "sr_seed{seed}_adapt_reward_change_full.csv",
     "end3"),
    # Fix #3: SR-AC > SR-full on reward_change (AUC)
    ("Fix #3: SR-AC AUC > SR-full AUC on reward_change",
     "sr_ac_seed{seed}_adapt_reward_change_full.csv",
     "sr_seed{seed}_adapt_reward_change_full.csv",
     "auc"),
    # Fix #3: SR-AC > PPO baseline on reward_change (AUC, harder bar)
    ("Fix #3: SR-AC AUC > PPO baseline AUC on reward_change",
     "sr_ac_seed{seed}_adapt_reward_change_full.csv",
     "ppo_seed{seed}_adapt_reward_change_full.csv",
     "auc"),
    # Fix #3: SR-AC > SR-wonly on reward_change (AUC) -- AC beats Momennejad ceiling?
    ("Fix #3: SR-AC AUC > SR-wonly AUC on reward_change",
     "sr_ac_seed{seed}_adapt_reward_change_full.csv",
     "sr_seed{seed}_adapt_reward_change_wonly.csv",
     "auc"),
    # Fix #3 on obs_remap
    ("Fix #3: SR-AC end > SR-full end on obs_remap",
     "sr_ac_seed{seed}_adapt_obs_remap_full.csv",
     "sr_seed{seed}_adapt_obs_remap_full.csv",
     "end3"),
]


def evaluate_claim(label, pat_a, pat_b, metric):
    a_dict = per_seed_metric(pat_a, metric)
    b_dict = per_seed_metric(pat_b, metric)
    common = sorted(set(a_dict).intersection(b_dict))
    if not common:
        return None
    a = [a_dict[s] for s in common]
    b = [b_dict[s] for s in common]
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_a, lo_a, hi_a = bootstrap_mean_ci(a)
    mean_b, lo_b, hi_b = bootstrap_mean_ci(b)
    mean_d, lo_d, hi_d = bootstrap_mean_ci(diffs)
    pos, neg, p_sign = sign_test(a, b)
    wil = wilcoxon(a, b)
    p_wil = wil[1] if wil is not None else None
    d = cohens_d_paired(a, b)
    return dict(
        label=label,
        metric=metric,
        n=len(common),
        seeds=common,
        a_per_seed=a,
        b_per_seed=b,
        diff_per_seed=diffs,
        a_mean=mean_a, a_ci=(lo_a, hi_a),
        b_mean=mean_b, b_ci=(lo_b, hi_b),
        diff_mean=mean_d, diff_ci=(lo_d, hi_d),
        sign_pos=pos, sign_neg=neg, p_sign=p_sign,
        p_wilcoxon=p_wil,
        cohens_d=d,
    )


def main():
    rows = []
    print()
    print("=" * 100)
    print("Statistical analyses for the headline claims (n=3 seeds)")
    print("=" * 100)
    for label, pa, pb, metric in CLAIMS:
        result = evaluate_claim(label, pa, pb, metric)
        if result is None:
            print(f"\n[SKIP] {label}: missing CSVs")
            continue
        print(f"\n{label}")
        print(f"  metric           {result['metric']}")
        print(f"  per-seed A       {[fmt(x) for x in result['a_per_seed']]}")
        print(f"  per-seed B       {[fmt(x) for x in result['b_per_seed']]}")
        print(f"  per-seed diff    {[fmt(x) for x in result['diff_per_seed']]}")
        print(f"  mean A (95% CI)  {fmt(result['a_mean'])}  [{fmt(result['a_ci'][0])}, {fmt(result['a_ci'][1])}]")
        print(f"  mean B (95% CI)  {fmt(result['b_mean'])}  [{fmt(result['b_ci'][0])}, {fmt(result['b_ci'][1])}]")
        print(f"  mean diff        {fmt(result['diff_mean'])}  [{fmt(result['diff_ci'][0])}, {fmt(result['diff_ci'][1])}]")
        print(f"  sign test        +{result['sign_pos']} / -{result['sign_neg']}, p={fmt(result['p_sign'])}")
        wp = result["p_wilcoxon"]
        print(f"  Wilcoxon p       {fmt(wp)}" + (" (n too small for reliable p)" if (wp is not None and result['n'] < 5) else ""))
        print(f"  Cohen's d        {fmt(result['cohens_d'])}")
        rows.append({
            "claim": result["label"],
            "metric": result["metric"],
            "n_seeds": result["n"],
            "a_seed_values": ";".join(f"{x:.3f}" for x in result["a_per_seed"]),
            "b_seed_values": ";".join(f"{x:.3f}" for x in result["b_per_seed"]),
            "diff_seed_values": ";".join(f"{x:+.3f}" for x in result["diff_per_seed"]),
            "a_mean": result["a_mean"],
            "a_ci_lo": result["a_ci"][0], "a_ci_hi": result["a_ci"][1],
            "b_mean": result["b_mean"],
            "b_ci_lo": result["b_ci"][0], "b_ci_hi": result["b_ci"][1],
            "diff_mean": result["diff_mean"],
            "diff_ci_lo": result["diff_ci"][0], "diff_ci_hi": result["diff_ci"][1],
            "sign_pos": result["sign_pos"], "sign_neg": result["sign_neg"],
            "p_sign": result["p_sign"],
            "p_wilcoxon": result["p_wilcoxon"],
            "cohens_d": result["cohens_d"],
        })

    if rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f"\nWrote {OUT_CSV}")

    # Concise interpretation note
    print()
    print("-" * 100)
    print("Power note: with n=3 the sign test floor is p=0.25 (3-of-3 same direction)")
    print("and Wilcoxon signed-rank likewise bottoms at 0.25. Effect size + CI bound")
    print("are therefore the more informative columns at this sample size.")
    print("-" * 100)


if __name__ == "__main__":
    main()
