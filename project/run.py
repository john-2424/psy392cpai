"""Single entry point for the full PSY392CPAI pipeline.

Runs, in order: PPO training, SR training, Replay training, SR no-norm ablation,
and the analysis notebook. Each stage writes to results/csv/, results/models/,
and results/figures/ as documented in README.md.

Usage:
    cd project
    export PYTHONPATH="."
    python run.py                         # everything
    python run.py --only ppo              # just PPO
    python run.py --skip ablation analysis
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

STAGES = {
    "ppo":      ("scripts.train_ppo",         "train"),
    "sr":       ("scripts.train_sr",          "train"),
    "replay":   ("scripts.train_replay",      "train"),
    "ablation": ("scripts.train_sr_no_norm",  "main"),
}
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

    all_stages = list(STAGES) + ["analysis"]
    if args.only:
        stages = [s for s in all_stages if s in args.only]
    else:
        stages = [s for s in all_stages if s not in args.skip]

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
