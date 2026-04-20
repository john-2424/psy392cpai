# PSY 392 CPAI — Vision-Based RL as a Probe of Reward, Transition, and Observation Change

Three canonical deep-RL architectures, three kinds of environmental change, one 8×8 vision-based gridworld. This repo tests whether the architectural priors of **PPO**, a deep **Successor Representation (SR)**, and a **Replay / Dyna DQN** produce the adaptation signatures that cognitive-neuroscience frameworks predict — [Momennejad et al., 2017](https://www.nature.com/articles/s41562-017-0180-8) for reward vs. transition revaluation, [Sanders, Wilson & Gershman, 2020](https://elifesciences.org/articles/56644) for hippocampal remapping, [Lehnert et al., 2024](https://arxiv.org/abs/2402.03900) for the deep-SF φ-normalization story.

Full write-up: **[`project/paper/main.tex`](project/paper/main.tex)** (NeurIPS 2024 format). Slide deck: **[`project/slides/slides.tex`](project/slides/slides.tex)** + **[transcripts](project/slides/slides_notes.md)** + **[Q&A prep](project/slides/qa_prep.md)**. Detailed reproduction guide: **[`project/README.md`](project/README.md)**.

---

## The project in one screen

- **Environment.** A deterministic 8×8 gridworld with 3-channel image observations (agent / goal / walls). Five conditions along two orthogonal axes:
  - *Layout* axis: `stable`, `reward_change` (goal moves), `transition_change` (wall moves).
  - *Observation* axis: `obs_visual` (distractor pixels — rate-remapping analog), `obs_remap` (channel permutation — global-remapping analog).
- **Agents.** Same CNN encoder for all three; only the head differs.
  - **PPO** — on-policy actor-critic (Schulman 2017).
  - **SR** — deep successor features, Q(s,a) = ⟨ψ(s,a), w⟩, φ ℓ₂-normalized (Dayan 1993; Momennejad 2017; Lehnert 2024).
  - **Replay** — DQN with a large buffer and Dyna-style amortization (Sutton 1990; Mnih 2015).
- **Protocol.** For each (agent, seed ∈ {0,1,2}): train on `stable` → zero-shot eval on all five conditions → short few-shot adaptation on each changed condition (20 PPO batches or 60 SR / Replay episodes). Plus an SR-only `wonly` variant on `reward_change` (freeze encoder + ψ, update only **w**).
- **Metrics.** Greedy success rate, adaptation-window AUC, median *t*₀.₅ (first step crossing 0.5), and a 5-fold-CV linear probe on CNN features (agent row, agent col, goal col, Manhattan-to-goal).

## Hypotheses and headline outcomes

| ID | Prediction | Outcome |
|---|---|---|
| **H1** | SR adapts fastest on `reward_change` | **Partially supported** — SR-`wonly` AUC **0.38** > PPO 0.15 > SR-full 0.12, within reach of Replay 0.44. |
| **H2** | Replay fastest on `transition_change` | **Directionally supported** — Replay and PPO both hit 1.00 (tie near ceiling). |
| **H3** | H1 ∧ H2 ⇒ cross-agent dissociation | **Not supported** at the population level; present *within* SR (reward AUC 0.38 / transition AUC 0.14 = 2.7×). |
| **H4** | All agents drop zero-shot on `obs_visual`, recover few-shot | **Clean for Replay** (0.67→1.00); **partial for PPO** (1.00→0.56 stable-LR fine-tune regression). |
| **H5** | `obs_remap` strictly harder than `obs_visual` | **Clean for Replay** (0.50 vs 1.00); PPO ties numerically; SR near floor on both. |

The most informative empirical result is the **representation probe**: SR's encoder is the only one that linearly decodes goal identity (acc **0.94**) and Manhattan distance (R² **+0.50**), and pays for it on agent-column (R² **−0.15**). SR's failure to hold a greedy policy is therefore a **deep-SF policy-extraction** problem, not a representation-collapse problem — honest to diagnose, concrete to fix (action-conditioned φ(s,a) + Q-margin regularizer).

---

## Repo map

```
psy392cpai/
├── README.md                       # (this file)
└── project/
    ├── README.md                   # Detailed reproduction guide — start here to re-run
    ├── report.md                   # Original project report (markdown, pre-paper)
    ├── requirements.txt
    ├── src/
    │   ├── envs/gridworld.py              # 8×8 GridWorldEnv + five conditions
    │   ├── algorithms/
    │   │   ├── ppo_torchrl.py             # PPO loss + policy/value networks
    │   │   ├── sr.py                      # SRNet, SR-Bellman loss, wonly helpers
    │   │   └── replay_planning.py         # ReplayQNet, Q-loss, replay buffer
    │   ├── models/                        # Shared CNN encoder + heads
    │   └── common/
    │       ├── adaptation.py              # Condition tuple, make_env, CSV helpers
    │       └── evaluation.py              # Shared greedy-eval functions
    ├── scripts/
    │   ├── train_ppo.py                   # 3 seeds × (stable + 4 adaptation)
    │   ├── train_sr.py                    # 3 seeds × (stable + 4 adaptation + wonly)
    │   ├── train_replay.py                # 3 seeds × (stable + 4 adaptation)
    │   ├── train_sr_no_norm.py            # Ablation: SR without φ-normalization
    │   └── probe_representations.py       # Linear-probe encoder features
    ├── notebooks/
    │   └── analysis.ipynb                 # Figures + summary tables
    ├── results/                           # CSVs, figures, checkpoints (generated)
    ├── paper/
    │   ├── main.tex + appendix.tex        # NeurIPS 2024 write-up
    │   └── figs/ + neurips_2024.sty
    └── slides/
        ├── slides.tex                     # 15-frame Beamer deck (story-driven)
        ├── slides_notes.md                # Per-slide spoken transcripts + timing
        └── qa_prep.md                     # 27 anticipated Q&As across 7 categories
```

---

## Quick start

Targets: Linux or WSL, Miniconda, CPU-only (no GPU needed). All commands below run from the repo root.

### 1. Setup

```bash
cd psy392cpai

conda create -n psy392_project python=3.10 -y
conda activate psy392_project

cd project
pip install -r requirements.txt
pip install jupyter
```

Sanity check:

```bash
python -c "import torch, torchrl, tensordict, gymnasium; print(torch.__version__, torchrl.__version__)"
```

### 2. Train all three agents (from `project/`)

Each script handles its own 3-seed sweep and writes CSVs + checkpoints under `results/`. Runs are independent and CPU-bound, so three terminals parallelizes cleanly:

```bash
cd project
export PYTHONPATH="."

python scripts/train_ppo.py     2>&1 | tee results/ppo_run.log      # ~20 min
python scripts/train_sr.py      2>&1 | tee results/sr_run.log       # ~60 min
python scripts/train_replay.py  2>&1 | tee results/replay_run.log   # ~60 min

# Ablation (SR with φ-normalization stripped — reproduces Lehnert 2024's prediction):
python scripts/train_sr_no_norm.py 2>&1 | tee results/sr_no_norm_run.log   # ~5 min
```

Each training script **already evaluates** zero-shot (periodic checkpoints across all 5 conditions) and runs the few-shot adaptation phase inline. There is no separate `eval.py` to call — everything downstream reads from the CSVs these scripts produce. Look at `results/csv/` to verify the outputs:

```bash
ls results/csv/ | wc -l          # ~94 CSVs once everything is done
```

For a per-seed monitoring cheat-sheet:

```bash
tail -f results/sr_run.log                             # live log
wc -l results/csv/sr_seed0_train.csv                   # grows to ~100 rows
```

### 3. Representation probe

Depends on the best-stable checkpoints written by the training scripts:

```bash
python scripts/probe_representations.py                # ~30 s
```

Writes `results/csv/probe_results.csv`.

### 4. Plot + generate summary tables

```bash
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --inplace
```

Produces, in `results/figures/`:

| File | Contents |
|---|---|
| `env_conditions.png` | 5-panel rendering of the five conditions |
| `training_curves.png` | Stable-phase return, mean ± std over 3 seeds |
| `zero_shot_eval.png` | Per-agent success across all 5 conditions |
| `adaptation_grid.png` | 3×4 grid of adaptation curves per (agent, condition) |
| `cross_agent_adaptation.png` | Early / Mid / Late checkpoint comparison |
| `representation_probe.png` | Linear-probe R² / accuracy per (agent, target) |
| `ablation_sr_no_norm.png` | SR with vs. without φ-normalization |

Plus `results/summary_table.csv` (Table 2 in the paper) and `results/adaptation_metrics.csv` (Table 3).

To inspect interactively instead of running headless:

```bash
jupyter notebook notebooks/analysis.ipynb
```

### 5. Build the paper and slides

```bash
cd project/paper  && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd ../slides      && pdflatex slides.tex && pdflatex slides.tex
```

Both are Overleaf-compatible if you prefer a hosted build (`project/paper/neurips_2024.sty` is bundled for that reason).

---

## Expected runtime (single-socket modern laptop CPU)

| Step | Wall-time |
|---|---|
| PPO (3 seeds × all conditions) | ~20 min |
| SR (3 seeds × 500 eps × all conditions + `wonly`) | ~60 min |
| Replay (3 seeds × all conditions) | ~60 min |
| SR no-norm ablation | ~5 min |
| Probe script | ~30 s |
| Notebook execution | ~30 s |
| **Full pipeline (serial)** | **~2 h 30 min** |
| **Full pipeline (3 parallel terminals)** | **~1 h 5 min** |

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'src'` — run from `project/` with `export PYTHONPATH="."` in the current shell.
- `DeprecationWarning: SyncDataCollector has been deprecated` — harmless torchrl API warning; `export RL_WARNINGS=False` silences it.
- `train_sr_no_norm.py` ends with `Early stop: loss exceeded 1e8` — that is the ablation behavior (Lehnert 2024's prediction reproduced).
- Notebook figures look stale after a rerun — clear `results/figures/` and re-execute the notebook.

See **[`project/README.md`](project/README.md)** for the long-form version (CSV count per agent × seed, parallel-launch recipes, more detailed troubleshooting).

---

## Citation

If you use this code or build on the findings, please cite the report in `project/paper/main.tex`:

```
Shrikrishna Rajule. "Vision-Based Reinforcement Learning as a Computational Probe of
Reward Revaluation, Transition Revaluation, and Hippocampal Remapping."
PSY 392 CPAI final project, Purdue University, Spring 2026.
```
