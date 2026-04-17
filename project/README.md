# PSY392CPAI — Vision-Based RL for Reward, Transition, and Observation Perturbations

Three RL agents (PPO, deep Successor Representation, Replay/Dyna DQN) compared on an 8×8 vision-based gridworld across five conditions: `stable`, `reward_change`, `transition_change`, `obs_visual`, `obs_remap`. See `report.md` for the full write-up and hypotheses.

This README covers reproducing the experiments end-to-end on a Linux host.

---

## 1. Environment setup (Linux, Miniconda)

Requires a working Miniconda/Anaconda install. CPU-only is sufficient; GPU not required.

```bash
# from the repo root
cd psy392cpai

conda create -n psy392_project python=3.10 -y
conda activate psy392_project

cd project
pip install -r requirements.txt
```

The dependencies are: `torch`, `torchrl`, `tensordict`, `gymnasium`, `numpy`, `matplotlib`, `pandas`, `pyyaml`, plus `jupyter` for the notebook:

```bash
pip install jupyter
```

Sanity check:

```bash
python -c "import torch, torchrl, tensordict, gymnasium; print(torch.__version__, torchrl.__version__)"
```

## 2. Repository layout

```
project/
├── src/
│   ├── envs/gridworld.py          # 8x8 GridWorldEnv, 5 conditions
│   ├── algorithms/
│   │   ├── ppo_torchrl.py         # PPO loss + policy/value nets
│   │   ├── sr.py                  # SRNet, SR Bellman loss, reward-only revaluation helpers
│   │   └── replay_planning.py     # ReplayQNet, Q-loss, replay buffer
│   ├── common/
│   │   ├── adaptation.py          # Condition tuple, make_env, CSV helpers
│   │   └── evaluation.py          # Shared greedy eval functions
│   └── models/                    # Shared CNN encoder and heads
├── scripts/
│   ├── train_ppo.py               # 3 seeds × (stable + 4 adaptation conditions)
│   ├── train_sr.py                # 3 seeds × (stable + 4 adaptation + wonly variant on reward_change)
│   ├── train_replay.py            # 3 seeds × (stable + 4 adaptation conditions)
│   └── train_sr_no_norm.py        # Ablation: SR without phi normalization
├── notebooks/
│   └── analysis.ipynb             # Figures + summary table
├── results/                       # CSVs, figures, model checkpoints (generated)
├── report.md                      # Full write-up
└── requirements.txt
```

## 3. Run all training (from `project/`)

Each script writes to `results/csv/` (CSVs) and `results/models/` (checkpoints). All three agents are CPU-bound and independent, so the three terminals approach parallelizes cleanly. If you only have one shell, run them sequentially.

```bash
cd project
export PYTHONPATH="."

# Three parallel terminals, one per agent:
python scripts/train_ppo.py     2>&1 | tee results/ppo_run.log      # ~20 min
python scripts/train_sr.py      2>&1 | tee results/sr_run.log       # ~45 min
python scripts/train_replay.py  2>&1 | tee results/replay_run.log   # ~60 min

# Ablation (after SR finishes, or any time):
python scripts/train_sr_no_norm.py 2>&1 | tee results/sr_no_norm_run.log  # ~5 min
```

Alternative background launch (single terminal, all four at once):

```bash
cd project
export PYTHONPATH="."
mkdir -p results
nohup python scripts/train_ppo.py         > results/ppo_run.log         2>&1 &
nohup python scripts/train_sr.py          > results/sr_run.log          2>&1 &
nohup python scripts/train_replay.py      > results/replay_run.log      2>&1 &
nohup python scripts/train_sr_no_norm.py  > results/sr_no_norm_run.log  2>&1 &
wait
```

Monitor progress while running:

```bash
tail -f results/ppo_run.log
wc -l results/csv/ppo_seed0_train.csv        # should grow to ~100 rows at 50k frames
ls results/csv/ | wc -l                      # should hit ~60+ CSVs once everything runs
```

## 4. Expected outputs per agent × seed

| CSV pattern | Purpose | Count per agent × seed |
|---|---|---|
| `<agent>_seed<s>_train.csv` | Stable-phase training log | 1 |
| `<agent>_seed<s>_eval_<cond>.csv` (5 conditions) | Zero-shot eval at periodic checkpoints | 5 |
| `<agent>_seed<s>_adapt_<cond>_full.csv` (4 conditions) | Few-shot adaptation (all params unfrozen) | 4 |
| `sr_seed<s>_adapt_reward_change_wonly.csv` | SR-only: Momennejad-style reward-weights-only variant | 1 (SR only) |

So for 3 seeds × 3 agents + SR `wonly` + ablation:

- PPO:    3 × (1 + 5 + 4) = 30
- SR:     3 × (1 + 5 + 4 + 1) = 33
- Replay: 3 × (1 + 5 + 4) = 30
- Ablation: 1

**Total ≈ 94 CSVs in `results/csv/`.**

Plus model checkpoints in `results/models/`: one stable-pretrain `.pt` per agent × seed.

## 5. Generate figures and summary table

After all training finishes:

```bash
cd project
export PYTHONPATH="."
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --inplace
```

This produces, in `results/figures/`:

- `env_conditions.png` — 5-panel observation visualization
- `training_curves.png` — stable-phase return, mean ± std over 3 seeds
- `zero_shot_eval.png` — 3 subplots (one per agent), success across all 5 conditions
- `adaptation_grid.png` — 3 × 4 grid of adaptation curves per (agent, condition)
- `cross_agent_adaptation.png` — bar chart at Early / Mid / Late checkpoints per condition
- `ablation_sr_no_norm.png` — SR with vs without φ-normalization

…and the summary table at `results/summary_table.csv`.

To inspect the notebook interactively instead:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 6. Expected runtime (single-socket modern laptop CPU)

| Step | Wall-time |
|---|---|
| PPO (3 seeds × all conditions) | ~20 min |
| SR (3 seeds × all conditions + `wonly`) | ~45 min |
| Replay (3 seeds × all conditions) | ~60 min |
| SR no-norm ablation | ~5 min |
| Notebook execution | ~30 s |
| **Full pipeline (serial)** | **~2 h 10 min** |
| **Full pipeline (3 parallel terminals)** | **~1 h** |

## 7. Troubleshooting

- `ModuleNotFoundError: No module named 'src'` — you forgot `export PYTHONPATH="."` in the current shell, or you're not in the `project/` directory.
- `DeprecationWarning: SyncDataCollector has been deprecated` — harmless; it's a torchrl API warning. If it clutters your logs, add `export RL_WARNINGS=False`.
- SR run ends early with `Early stop: loss exceeded 1e8` — only happens for `train_sr_no_norm.py`. That is the expected ablation behavior.
- CSV files appear partial — runs are likely still in progress. Check with `tail -f results/<agent>_run.log` and `wc -l results/csv/*.csv`.
- Re-running a script is idempotent for eval/adapt CSVs (they append) but destructive for `<agent>_seed<s>_train.csv` only in the ablation (`train_sr_no_norm.py` deletes its own CSV at start). If you want a clean slate, remove the specific CSVs first: `rm results/csv/ppo_seed*`, etc.

## 8. Report

The full write-up — motivation, hypotheses, methods, results, and discussion — is in `report.md` at the `project/` level. The notebook's figures are referenced by filename from the report.
