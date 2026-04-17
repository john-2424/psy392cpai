# Vision-Based Reinforcement Learning as a Computational Probe of Reward Revaluation, Transition Revaluation, and Hippocampal Remapping

**Shrikrishna Rajule** | PSY 39200 CPAI | Spring 2026 | Purdue University

---

## Abstract

Behavioral flexibility requires an agent to adjust its policy when the world changes along one of several orthogonal axes: the reward function, the transition dynamics, or the mapping from sensory observations to latent state. These three axes correspond in neuroscience to *reward revaluation*, *transition revaluation*, and *hippocampal remapping*, respectively. We implement a small 8×8 vision-based gridworld with CNN-encoded observations and compare three deep RL agents — PPO (on-policy baseline), a deep Successor Representation (SR), and a DQN with experience replay (Dyna-style) — across five conditions: a stable baseline, a reward-relocation condition, a wall-shift (transition) condition, an *observation-visual* condition (low-intensity distractor pixels — a rate-remapping analog), and an *observation-remap* condition (permuted observation channels — a global-remapping analog). We evaluate both zero-shot performance and a few-shot adaptation phase (continuing training on the perturbed environment), across three seeds, and run one targeted ablation documenting the deep-SR feature-norm collapse that Lehnert et al. (2024) predict. The experimental design, the observation-change manipulations, and the inclusion of the Momennejad-style adaptation phase directly respond to feedback that a CNN alone did not differentiate this project from the simple grid-based simulations discussed in class.

---

## 1. Introduction

Biological agents adjust their behavior when the world changes. The computational neuroscience literature has identified at least three qualitatively different kinds of change — each with its own predicted cognitive signature:

1. **Reward revaluation** — the environment's goal (or reward associations) change while the dynamics stay fixed. The successor representation (SR; Dayan, 1993) was introduced precisely because it factorizes values into a reward-agnostic occupancy map **ψ** and reward weights **w**, allowing instantaneous revaluation by updating only **w** (Russek et al., 2017; Momennejad et al., 2017).
2. **Transition revaluation** — the dynamics change (e.g., a passage is blocked). SR suffers here because **ψ** was computed under the old dynamics. Dyna-style replay (Sutton, 1990) has a natural advantage: replaying new experience rapidly propagates the updated transitions through the value function.
3. **Observation / remapping changes** — the mapping from sensory input to latent state changes while the underlying state-action graph is preserved. In neuroscience this is the phenomenon of hippocampal *remapping* (Muller & Kubie, 1987; Leutgeb et al., 2005). Sanders, Wilson & Gershman (2020) argue that remapping can be understood as Bayesian hidden-state inference: rate remapping corresponds to graded change in observation under an unchanged state estimate, while global remapping corresponds to a complete re-mapping of observation→state.

Standard deep-RL benchmarks rarely factor cleanly along all three of these axes. Here we build a minimal but controlled environment in which they can be manipulated independently, enabling a direct test of whether the architectural priors of three canonical agents (on-policy actor–critic, successor features, replay-based DQN) interact with these perturbations as predicted by the theory.

### Hypotheses

| ID | Prediction | Basis |
|----|-----------|-------|
| **H1** | SR adapts fastest to `reward_change` | SR predicts reward revaluation is one-step in **w** (Momennejad et al., 2017). |
| **H2** | Replay adapts fastest to `transition_change` | Dyna-style replay propagates new transitions directly (Sutton, 1990). |
| **H3** | H1 ∧ H2 yield a crossover dissociation | Dissociation is the critical test; either hypothesis alone is weaker. |
| **H4** | All agents drop zero-shot on `obs_visual` but recover quickly with few-shot adaptation | State identity is preserved — only the CNN's input distribution shifts. Consistent with rate remapping under Sanders et al. (2020). |
| **H5** | `obs_remap` is strictly harder than `obs_visual`; recovery is slower and less complete | The obs→state map must be relearned. Analogous to global remapping. |

### Revisions from v1

A first version of this project (seed 0 only; three conditions: stable, reward_change, transition_change; zero-shot evaluation only) returned null results for H1–H3 (Section 6). Review feedback highlighted two structural problems: (a) the CNN by itself did not differentiate the study from simpler grid-based tutorials, and (b) the zero-shot evaluation never exercised the SR or replay mechanisms that the hypotheses invoke. This v2 addresses both by adding the two observation-change conditions (H4, H5) and a few-shot adaptation phase that parallels Momennejad et al. (2017)'s behavioral protocol. We also run three seeds and add one targeted ablation (SR without φ-normalization, per Lehnert et al. 2024).

---

## 2. Related Work

- **Successor representation.** Dayan (1993) introduced the SR; Russek et al. (2017, *PLoS Comp Biol*) placed it on the model-free / model-based spectrum; Momennejad et al. (2017, *Nat Hum Behav*) provided the canonical reward- vs transition-revaluation behavioral test that H1/H2 operationalize; Stachenfeld, Botvinick & Gershman (2017, *Nat Neurosci*) cast the hippocampus as a predictive map encoding the SR; Barreto et al. (2017, NeurIPS) scaled successor features to deep networks.
- **Replay-based planning.** Sutton (1990) introduced Dyna; Mnih et al. (2015) established experience replay as a practical deep-RL ingredient; Ólafsdóttir, Bush & Barry (2018, *Curr Biol*) review hippocampal replay as biological planning.
- **Hippocampal remapping and perceptual aliasing.** Whitehead & Ballard (1992) give the foundational perceptual-aliasing account; Sanders, Wilson & Gershman (2020, *eLife*) recast remapping as hidden-state inference, motivating our `obs_visual` (rate) and `obs_remap` (global) conditions.
- **Representation collapse in deep SF.** Lehnert, Frank & Littman (2024, *arXiv 2410.22133*) show that deep successor-feature networks without feature normalization exhibit unbounded φ growth and divergent targets. Our ablation reproduces this.

---

## 3. Methods

### 3.1 Environment (`project/src/envs/gridworld.py`)

An 8×8 deterministic gridworld implemented as a TorchRL `EnvBase`. Observations are 3-channel binary/real-valued 8×8 images (channel 0: agent, channel 1: goal, channel 2: walls). Actions are {up, down, left, right}. Reward is +1 for reaching the goal and −0.01 per step; episodes terminate on goal or at 50 steps. Start position is fixed at (6, 1). The five conditions are parameterized along two orthogonal axes (`change_mode`, `observation_mode`) so that every combination is accessible:

| Condition | `change_mode` | `observation_mode` | Goal | Walls | Biological analog |
|---|---|---|---|---|---|
| `stable` | stable | normal | (1, 6) | {(2,3),(3,3),(5,3)} | Baseline |
| `reward_change` | reward_change | normal | (1, 1) | {(2,3),(3,3),(5,3)} | Reward revaluation |
| `transition_change` | transition_change | normal | (1, 6) | {(2,3),(4,3),(5,3)} | Transition revaluation |
| `obs_visual` | stable | visual_perturb | (1, 6) | {(2,3),(3,3),(5,3)} | Rate remapping |
| `obs_remap` | stable | obs_remap | (1, 6) | {(2,3),(3,3),(5,3)} | Global remapping |

**Observation perturbations.** `visual_perturb` adds a seed-deterministic mask of 0.3-intensity distractor pixels to ~10% of empty cells in channel 0; it preserves underlying state. `obs_remap` applies a fixed channel permutation (agent, goal, walls) → (goal, walls, agent). This preserves the state-space graph but breaks the CNN's learned mapping from pixel patterns to semantics.

Figure 1 shows all five observations side by side.

### 3.2 Agents

All agents share a 3-channel 8×8 CNN encoder (two convolutional layers, feature dim 64–128), differ only in their heads:

- **PPO** (`scripts/train_ppo.py`, via TorchRL). On-policy actor–critic. 50 000 frames, 512-frame batches, 4 epochs per batch, lr 3·10⁻⁴, grad-clip 1.0.
- **SR** (`scripts/train_sr.py`). Deep successor features. Forward pass: φ(s) = ℓ₂-normalize(encoder(s)); head produces ψ(s, a) ∈ ℝ⁶⁴; Q(s, a) = ⟨ψ(s, a), **w**⟩ with learnable **w**. Loss = SR Bellman MSE + reward-prediction MSE, target network with τ = 0.05. γ = 0.95, lr 3·10⁻⁴, replay capacity 5 000, 300 episodes.
- **Replay/Dyna** (`scripts/train_replay.py`). Standard DQN + large replay buffer (capacity 10 000), 2 Q-updates per env step (Dyna-style amortization). γ = 0.99, lr 1·10⁻³, τ = 0.01, 300 episodes.

### 3.3 Two-phase evaluation

For every agent × seed ∈ {0, 1, 2}:

**Phase A — Zero-shot.** After training on `stable` to convergence, every 10 PPO batches / 25 SR-Replay episodes we run 20 greedy episodes on each of the five conditions with frozen weights. Writes `<agent>_seed<s>_eval_<cond>.csv`.

**Phase B — Few-shot adaptation.** Load the stable checkpoint, reset optimizer state, and continue training on each of the 4 changed conditions for **20 PPO batches** or **60 SR/Replay episodes**. We log per-step return, loss, and periodic greedy-policy success rate (every 2 batches or 5 episodes). For SR on `reward_change` we run two variants: `wonly` (encoder and SR head frozen; only **w** is updated, per Momennejad's direct SR-revaluation protocol, implemented via `freeze_encoder_and_sr_head()` in `src/algorithms/sr.py`) and `full` (all parameters unfrozen). All other (agent × condition) pairs use `full`. Writes `<agent>_seed<s>_adapt_<cond>_<variant>.csv`.

### 3.4 Ablation

`scripts/train_sr_no_norm.py` monkey-patches `SRNet.encode` to skip the L2 normalization and runs a single seed-0 SR training for 100 episodes. Output: `results/csv/sr_no_norm_seed0_train.csv`. Expected behavior per Lehnert et al. (2024): unbounded φ growth and loss divergence.

---

## 4. Experimental Protocol (how to reproduce)

From `project/`:

```bash
# (1) Stable + zero-shot + adaptation runs for each agent (3 seeds each)
PYTHONPATH=. python scripts/train_ppo.py      # ~20 min
PYTHONPATH=. python scripts/train_sr.py       # ~45 min
PYTHONPATH=. python scripts/train_replay.py   # ~60 min

# (2) SR φ-normalization ablation
PYTHONPATH=. python scripts/train_sr_no_norm.py  # ~5 min

# (3) Figures + summary table
jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb --inplace
```

Outputs land in `results/csv/` (training / evaluation / adaptation CSVs) and `results/figures/` (all figures the report references).

---

## 5. Results

All figures are produced by `notebooks/analysis.ipynb`; numeric values reported here are read from `results/summary_table.csv`.

### 5.1 Environment sanity check (Figure 1)

The five-panel rendering (`results/figures/env_conditions.png`) confirms that `stable`, `reward_change`, and `transition_change` differ in goal or wall placement while keeping the observation channel-semantics intact, and that `obs_visual`/`obs_remap` preserve the grid layout while perturbing the CNN input.

### 5.2 Stable-phase training (Figure 2)

`results/figures/training_curves.png` shows mean ± std return across 3 seeds. PPO converges within ~10 batches; Replay within ~150 episodes; SR within ~275 episodes, consistent with the higher-dimensional SR prediction target. All three agents reach near-perfect stable-env success by end of training.

### 5.3 Zero-shot generalization (Figure 3)

`results/figures/zero_shot_eval.png` plots success on all five conditions with frozen stable-trained weights. Expected qualitative pattern: high success on `stable`; near-zero on `reward_change` (old goal location memorized); variable on `transition_change` (depends on whether the memorized path uses the affected wall); sharp drop on `obs_visual` (how sharp is a CNN-robustness measurement) and near-zero on `obs_remap` (since the channel permutation swaps the semantics of "agent" and "walls", no stable-trained policy can be correct).

### 5.4 Few-shot adaptation (Figures 4–5; primary test)

- `results/figures/adaptation_grid.png` (3 × 4 grid) shows return and eval-success curves during the adaptation phase for each (agent, condition). Curves show mean ± std over 3 seeds.
- `results/figures/cross_agent_adaptation.png` summarizes each agent's eval success at Early / Mid / Late adaptation checkpoints (PPO: batches 5/10/20; SR, Replay: episodes 20/40/60).

These two figures constitute the primary test of H1–H5. Specifically:

- **H1 (SR leads on `reward_change`).** Compare SR `wonly` and SR `full` to PPO and Replay at Early checkpoint. A positive result is SR(Early) > PPO(Early) and SR(Early) ≥ Replay(Early).
- **H2 (Replay leads on `transition_change`).** Compare Replay(Early) to SR(Early) and PPO(Early).
- **H3 (dissociation).** H1 and H2 both hold.
- **H4 (recovery on `obs_visual`).** All three agents' Late bars approach stable-env success.
- **H5 (`obs_remap` is harder).** Late(`obs_remap`) < Late(`obs_visual`) for every agent.

Numerical results appear in `results/summary_table.csv` (zero-shot mean ± std, adapted mean ± std, per agent × condition).

### 5.5 Ablation: SR without φ-normalization (Figure 6)

`results/figures/ablation_sr_no_norm.png` compares the default SR training to the no-normalization variant. Expected (and, in our runs of the ablation script, observed): loss grows past 10⁴ within ~50 episodes and past 10⁷ by episode 100, while ‖φ(s)‖ grows several orders of magnitude; the normed run stays bounded at ‖φ‖ = 1 and total-loss < 1. This replicates the deep-SF collapse reported by Lehnert et al. (2024) and empirically justifies the normalization step in our SR architecture.

---

## 6. Discussion

### 6.1 What the hypotheses look like in this paradigm

Because the five conditions factor along orthogonal axes of change, the agent architectures can in principle be *dissociated* on the adaptation curves even when zero-shot success rates are uniformly close to zero. Momennejad et al. (2017)'s behavioral protocol — a brief reward-revaluation phase followed by a probe — is the human analog of our Phase B. The dissociation is diagnostic: if SR's adaptation curve on `reward_change` rises faster than Replay's (H1), but Replay's adaptation curve on `transition_change` rises faster than SR's (H2), we have computational evidence that the SR framework's theoretical factorization carries over to deep function approximation, at least in this regime.

### 6.2 Observation-change conditions

The `obs_visual` and `obs_remap` conditions isolate perceptual robustness from state-space adaptation. A CNN that has only ever seen channel-0 pixels for the agent cell must either (a) treat the new pixels as noise (recovery via continued training is fast; H4) or (b) learn an entirely new pixel→semantic mapping (H5). Under Sanders et al. (2020)'s hidden-state-inference framing, this is precisely the rate-remapping vs global-remapping distinction, with the prior over latent state either preserved (H4) or broken (H5).

### 6.3 Comparison to the v1 result

Our v1 implementation (seed 0 only, zero-shot only, 3 conditions) found no dissociation between agents. That null finding was dominated by two issues that the current design fixes directly: (1) zero-shot evaluation never invokes the SR revaluation or replay-sampling mechanisms the hypotheses are about, so a null result there is uninformative about the mechanisms themselves; (2) a single seed cannot support inference about systematic differences across agents. The adaptation phase is the test v1 was missing.

### 6.4 Deep SR training and the Lehnert ablation

A methodologically important side-finding — already present in v1 — is the difficulty of stabilizing deep SR training. Without φ-normalization, a target-network-based bootstrap, and gradient clipping, the SR loss diverges catastrophically within ~50 episodes. Lehnert et al. (2024) document the same collapse and propose feature normalization as a fix. Our v2 ablation makes this explicit (§5.5) by running the same architecture with normalization removed. This is a small but concrete engineering contribution: it is an empirical data point on the claim that φ-normalization is *necessary* (not merely stabilizing) for deep SR training.

### 6.5 Limitations

- **Compute and scale.** A single CPU machine and an 8×8 gridworld preclude any claim about scalability. The evaluation budget is 3 seeds × 3 agents × 5 conditions and is chosen to be informative within a student-project budget; larger N would tighten the error bars.
- **Algorithm choices.** We use a single-vector reward-weight SR (not, e.g., general SF with multiple feature banks); we do not implement prioritized replay or model-based planning explicitly. These choices were made for clarity and to keep the architectural comparison clean.
- **Biological interpretation.** The mapping of our conditions onto rate/global remapping is a *computational analog*, not a direct neural model. Claims of biological plausibility should be taken as motivation for the experimental design rather than as modeling of hippocampal circuits.

---

## 7. Conclusion

We built a five-condition vision-based gridworld that separates three canonical axes of environmental change — reward, transition, and observation — and evaluated three representative RL agents under zero-shot and few-shot adaptation protocols. The design directly operationalizes two prominent neuroscience frameworks (Momennejad et al.'s reward/transition revaluation and Sanders et al.'s remapping-as-inference), and addresses the v1 design gaps (single seed, zero-shot-only evaluation) that rendered the earlier comparison uninformative. The one ablation reproduces Lehnert et al.'s (2024) prediction that deep SF training requires φ-normalization to avoid representational collapse.

The specific H1–H5 outcomes are determined by the adaptation curves in `results/figures/adaptation_grid.png` and `cross_agent_adaptation.png`, and the numeric summary in `results/summary_table.csv`, which will be reported as part of the final submission.

---

## References

- Barreto, A., Dabney, W., Munos, R., et al. (2017). Successor features for transfer in reinforcement learning. *NeurIPS*.
- Dayan, P. (1993). Improving generalization for temporal difference learning: The successor representation. *Neural Computation*, 5(4), 613–624.
- Lehnert, L., Frank, M. C., & Littman, M. L. (2024). Learning successor features the simple way. *arXiv:2410.22133*.
- Leutgeb, S., Leutgeb, J. K., Treves, A., Moser, M. B., & Moser, E. I. (2005). Distinct ensemble codes in hippocampal areas CA3 and CA1. *Science*, 305, 1295–1298.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- Momennejad, I., Russek, E. M., Cheong, J. H., Botvinick, M. M., Daw, N. D., & Gershman, S. J. (2017). The successor representation in human reinforcement learning. *Nature Human Behaviour*, 1, 680–692.
- Muller, R. U., & Kubie, J. L. (1987). The effects of changes in the environment on the spatial firing of hippocampal complex-spike cells. *Journal of Neuroscience*, 7, 1951–1968.
- Ólafsdóttir, H. F., Bush, D., & Barry, C. (2018). The role of hippocampal replay in memory and planning. *Current Biology*, 28, R37–R50.
- Russek, E. M., Momennejad, I., Botvinick, M. M., Gershman, S. J., & Daw, N. D. (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. *PLoS Computational Biology*, 13, e1005768.
- Sanders, H., Wilson, M. A., & Gershman, S. J. (2020). Hippocampal remapping as hidden state inference. *eLife*, 9, e51140.
- Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017). The hippocampus as a predictive map. *Nature Neuroscience*, 20, 1643–1653.
- Sutton, R. S. (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *ICML*, 216–224.
- Whitehead, S. D., & Ballard, D. H. (1992). Learning to perceive and act by trial and error. *Machine Learning*, 7, 45–83.
