"""Single entry point for the full PSY392CPAI pipeline.

Runs, in order: PPO training, SR training, Replay training, SR no-norm ablation,
and the analysis notebook. Each stage writes to results/csv/, results/models/,
and results/figures/ as documented in README.md.

Three slide-15 future-work extensions are exposed as separate stages:
    ppo_lr10x   -- PPO adapt-only with adapt-LR scaled by 0.1 (fix #1)
    sr_qmargin  -- SR retrained with the Q-margin hinge enabled (fix #2)
    sr_ac       -- Action-conditioned phi(s, a) SR variant (fix #3)
None of these are part of the default --only/all set; opt in via --only.

Usage:
    cd project
    export PYTHONPATH="."
    python run.py                                    # baseline pipeline
    python run.py --only ppo                         # just PPO baseline
    python run.py --only ppo_lr10x sr_qmargin sr_ac  # just the three extensions
    python run.py --skip ablation analysis
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

STAGES = {
    "ppo":         ("scripts.train_ppo",        "train"),
    "sr":          ("scripts.train_sr",         "train"),
    "replay":      ("scripts.train_replay",     "train"),
    "ablation":    ("scripts.train_sr_no_norm", "main"),
    # Slide-15 extensions (not in the default pipeline)
    "ppo_lr10x":   ("scripts.train_ppo",        "train_adapt_only"),
    "sr_qmargin":  ("scripts.train_sr",         "train_with_qmargin"),
    "sr_ac":       ("scripts.train_sr_ac",      "train"),
}
DEFAULT_STAGES = ["ppo", "sr", "replay", "ablation"]
EXTENSION_STAGES = ["ppo_lr10x", "sr_qmargin", "sr_ac"]
NOTEBOOK = Path("notebooks/analysis.ipynb")


def run_stage(name: str) -> None:
    module, fn = STAGES[name]
    print(f"\n===== [{name}] starting ({module}.{fn}) =====", flush=True)
    t0 = time.time()
    mod = __import__(module, fromlist=[fn])
    getattr(mod, fn)()
    dt = time.time() - t0
    print(f"===== [{name}] done in {dt/60:.1f} min =====", flush=True)


def run_notebook() -> None:
    print("\n===== [analysis] executing notebook =====", flush=True)
    t0 = time.time()
    subprocess.check_call([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", str(NOTEBOOK), "--inplace",
    ])
    dt = time.time() - t0
    print(f"===== [analysis] done in {dt/60:.1f} min =====", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(STAGES) + ["analysis"],
                        help="Run only these stages.")
    parser.add_argument("--skip", nargs="+", choices=list(STAGES) + ["analysis"],
                        default=[], help="Skip these stages.")
    args = parser.parse_args()

    # Default stage set excludes the three slide-15 extensions; user must opt in.
    default_set = DEFAULT_STAGES + ["analysis"]
    if args.only:
        # If the user opted into extension stages explicitly, include them.
        all_stages = list(STAGES) + ["analysis"]
        stages = [s for s in all_stages if s in args.only]
    else:
        stages = [s for s in default_set if s not in args.skip]

    print(f"Running stages: {stages}", flush=True)
    t0 = time.time()
    for s in stages:
        if s == "analysis":
            run_notebook()
        else:
            run_stage(s)
    dt = time.time() - t0
    print(f"\nAll done in {dt/60:.1f} min total.", flush=True)


if __name__ == "__main__":
    main()
