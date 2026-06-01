# psy392cpai

Repository for MS Autonomy PSY392CPAI Computational Psychology and AI

Project Archives: https://drive.google.com/drive/folders/1sG5Yqnz0p08hTLwBtSAaymHquQYzb1f8?usp=sharing

---

## Main Work: Final Project

**Vision-Based Reinforcement Learning as a Computational Probe of Reward Revaluation, Transition Revaluation, and Hippocampal Remapping**

The centerpiece of this repository is the final PSY392CPAI project in [`project/`](project/). It builds a small, controlled vision-based reinforcement learning testbed for studying how different RL architectures respond when the world changes in different ways.

At a high level, the project asks:

> Can agents with different computational priors distinguish reward changes, transition changes, and observation-level remapping?

To test that, the project implements an 8x8 gridworld where observations are 3-channel image tensors, then compares three agents that share the same CNN encoder but differ in their learning heads and adaptation mechanisms:

- **PPO**: an on-policy actor-critic baseline using TorchRL.
- **Successor Representation (SR)**: a deep successor-feature agent that factorizes values into successor features and reward weights.
- **Replay/Dyna DQN**: a Q-learning agent with a large replay buffer and multiple updates per environment step.

The experimental design is motivated by computational psychology and neuroscience work on successor representations, reward revaluation, transition revaluation, replay-based planning, and hippocampal remapping.

### Conditions Studied

The project separates five task conditions:

| Condition | What Changes | Interpretation |
|---|---|---|
| `stable` | Nothing | Baseline navigation task |
| `reward_change` | Goal/reward location moves | Reward revaluation |
| `transition_change` | Wall layout changes | Transition revaluation |
| `obs_visual` | Low-intensity distractor pixels are added | Rate-remapping-style observation shift |
| `obs_remap` | Observation channels are permuted | Global-remapping-style state observation shift |

Each agent is trained and evaluated across seeds with two evaluation phases:

- **Zero-shot evaluation**: train on the stable environment, then test directly on all five conditions.
- **Few-shot adaptation**: continue training on each changed condition and measure how quickly the agent recovers.

### Main Takeaways

The final report and presentation highlight several useful findings:

- Replay/Dyna is the strongest overall adaptation baseline, especially on transition changes and visual observation shifts.
- PPO learns the stable task quickly and generalizes surprisingly well to some perturbations, but can become unstable during adaptation unless the adaptation learning rate is tuned.
- The SR agent shows the expected reward-revaluation signal most clearly when only its reward weights are updated, matching the Momennejad-style successor representation prediction better than the end-of-phase snapshot alone suggests.
- The project does not recover a clean population-level crossover where SR dominates reward changes and Replay dominates transition changes. Instead, Replay remains strong on both.
- The representation probe is one of the most interesting diagnostics: SR learns goal identity and distance-to-goal more clearly than PPO or Replay, but struggles with action-discriminative position information, especially agent column. This reframes the SR result as a policy-extraction fragility rather than simple representational collapse.
- The SR no-normalization ablation reproduces a deep successor-feature failure mode: without feature normalization, feature norms and Bellman errors diverge quickly.
- The extension experiments include PPO adaptation with a lower learning rate, Q-margin SR, and an action-conditioned SR variant.

### Final Project Layout

```text
project/
|-- README.md                  # Detailed reproduction guide for the final project
|-- report.md                  # In-repo project write-up
|-- run.py                     # Single entry point for baseline and extension stages
|-- requirements.txt           # Python dependencies
|-- configs/                   # Placeholder config files
|-- notebooks/
|   `-- analysis.ipynb         # Figure generation and summary analysis
|-- scripts/
|   |-- train_ppo.py           # PPO baseline and adaptation runs
|   |-- train_sr.py            # Successor Representation runs
|   |-- train_replay.py        # Replay/Dyna DQN runs
|   |-- train_sr_no_norm.py    # SR feature-normalization ablation
|   |-- train_sr_ac.py         # Action-conditioned SR extension
|   |-- probe_representations.py
|   |-- analyze_extensions.py
|   |-- stats_summary.py
|   `-- check_env.py
|-- src/
|   |-- envs/                  # 8x8 vision gridworld
|   |-- algorithms/            # PPO, SR, Replay, SR-AC implementations
|   |-- models/                # Shared CNN encoder and agent heads
|   |-- common/                # Adaptation helpers and shared utilities
|   |-- evaluation/            # Plotting and evaluation utilities
|   |-- collectors/            # Scaffolding for rollout collection
|   `-- utils/                 # Utility scaffolding
`-- results/
    |-- csv/                   # Training, zero-shot, adaptation, and probe CSVs
    |-- figures/               # Generated report and extension figures
    |-- models/                # Saved model checkpoints
    |-- summary_table.csv      # Zero-shot/adapted success summary
    |-- adaptation_metrics.csv # AUC and time-to-threshold metrics
    `-- extensions_summary.csv # Extension experiment summaries
```

### Reproducing the Final Project

The project is designed to run from the `project/` directory. CPU-only execution is sufficient.

```bash
cd project
pip install -r requirements.txt
```

Run the baseline pipeline:

```bash
python run.py
```

Run selected stages:

```bash
python run.py --only ppo
python run.py --only sr replay ablation analysis
python run.py --only ppo_lr10x sr_qmargin sr_ac
```

The project-level README has the most detailed reproduction notes, expected runtimes, expected outputs, and troubleshooting guidance.

### Key Final Project Outputs

The checked-in `project/results/` directory includes:

- **177 CSV files** for training curves, zero-shot evaluation, few-shot adaptation, probes, and extension summaries.
- **10 figure files** covering environment conditions, training curves, zero-shot evaluation, adaptation behavior, representation probing, ablations, and extension comparisons.
- **39 PyTorch checkpoints** for trained and adapted agents.

Important summary artifacts include:

- [`project/results/summary_table.csv`](project/results/summary_table.csv)
- [`project/results/adaptation_metrics.csv`](project/results/adaptation_metrics.csv)
- [`project/results/extensions_summary.csv`](project/results/extensions_summary.csv)
- [`project/results/figures/representation_probe.png`](project/results/figures/representation_probe.png)
- [`project/results/figures/adaptation_grid.png`](project/results/figures/adaptation_grid.png)
- [`project/results/figures/extensions_sr_ac_compare.png`](project/results/figures/extensions_sr_ac_compare.png)

---

## Course Assignment Archive

The [`assignments/`](assignments/) directory contains the PSY392CPAI tutorial notebooks from the semester. Together they document the course progression from basic computational modeling through modern neural and probabilistic models:

| Tutorial | Topic |
|---|---|
| Tutorial 01 | Model types |
| Tutorial 02 | Python workshops 1 and 2 |
| Tutorial 03 | Probability and statistics |
| Tutorial 04 | Linear algebra and calculus |
| Tutorial 05 | Introduction to reinforcement learning |
| Tutorial 06 | Reinforcement learning and the brain |
| Tutorial 07 | Introduction to artificial neural networks |
| Tutorial 08 | Convolutional neural networks |
| Tutorial 09 | Hopfield networks |
| Tutorial 11 | Boltzmann machines |
| Tutorial 12 | Helmholtz machines and variational autoencoders |
| Tutorial 13 | Transformers and natural language processing, parts 1 and 2 |
| Tutorial 14 | The Free Energy Principle in psychology and neuroscience |

These notebooks provide the course context for the final project: MDPs and TD learning, neuroscience framing, neural-network function approximation, CNN encoders, representation learning, and probabilistic generative models.

---

## Repository Contents

At the repository root:

- [`README.md`](README.md): this high-level guide.
- [`LICENSE.txt`](LICENSE.txt): project license.
- [`assignments/`](assignments/): course tutorial notebooks.
- [`project/`](project/): final project code, analysis, report, outputs, and reproduction guide.

For day-to-day use, start with `project/README.md` if you want to rerun experiments, and start with `project/report.md` if you want the scientific motivation, methods, results, and discussion.
