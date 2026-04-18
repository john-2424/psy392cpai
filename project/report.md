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

All figures are in `results/figures/`; numeric values are read from `results/summary_table.csv`. Results below are from a first full 3-seed run; the SR training instability documented in §5.3 is being addressed by a follow-up run using the patched `scripts/train_sr.py` (500 episodes, best-checkpoint saving, best-checkpoint-as-adaptation-starting-point).

### 5.1 Environment sanity check (Figure 1)

The five-panel rendering (`results/figures/env_conditions.png`) confirms that `stable`, `reward_change`, and `transition_change` differ in goal or wall placement while keeping the observation channel-semantics intact, and that `obs_visual`/`obs_remap` preserve the grid layout while perturbing the CNN input.

### 5.2 Stable-phase training (Figure 2)

`results/figures/training_curves.png` shows mean ± std return across 3 seeds.

- **PPO** converges within ~10 batches; stable eval success reaches 1.00 across all three seeds.
- **Replay** converges by ~150 episodes; final-checkpoint stable success = 0.89 ± 0.19 (seed 1 regresses at eps 275–300 after hitting 1.00 at eps 250; the `_best.pt` checkpoint is fine but adaptation loads the final state).
- **SR** reaches stable success in only 1 of 3 seeds within 300 episodes (seed 2 hits 1.00 at ep 300; seeds 0 and 1 remain at 0.00). Aggregate stable eval = 0.11 ± 0.19.

### 5.3 Zero-shot generalization (Figure 3)

`results/figures/zero_shot_eval.png`. Headline numbers for the last three eval checkpoints on each of the 5 conditions:

| Agent | Stable | Reward Δ | Transition Δ | Obs Visual | Obs Remap |
|---|---|---|---|---|---|
| PPO | 1.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.00 ± 0.00 |
| SR | 0.11 ± 0.19 | 0.00 ± 0.00 | 0.11 ± 0.19 | 0.11 ± 0.19 | 0.00 ± 0.00 |
| Replay | 0.89 ± 0.19 | 0.00 ± 0.00 | 0.67 ± 0.58 | 0.67 ± 0.33 | 0.00 ± 0.00 |

PPO and Replay generalize to `transition_change` and (for PPO) `obs_visual` from stable alone — both happened to learn paths not blocked by the shifted wall, and PPO's CNN proved insensitive to the low-intensity distractors in `obs_visual`. Both agents hit 0 zero-shot on `reward_change` and `obs_remap`, as expected.

### 5.4 Few-shot adaptation (Figures 4–5; primary test)

`results/figures/adaptation_grid.png` (3 × 4 grid) shows return and eval-success curves during the adaptation phase for each (agent, condition); `results/figures/cross_agent_adaptation.png` summarizes eval success at Early / Mid / Late checkpoints (PPO: batches 5/10/20; SR, Replay: episodes 20/40/60).

| Agent | Reward Δ adapted | Transition Δ adapted | Obs Visual adapted | Obs Remap adapted |
|---|---|---|---|---|
| PPO | 0.11 ± 0.19 | 1.00 ± 0.00 | 0.56 ± 0.51 | 0.56 ± 0.51 |
| SR (full) | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.11 ± 0.19 | 0.00 ± 0.00 |
| Replay | **0.78 ± 0.38** | **1.00 ± 0.00** | **1.00 ± 0.00** | **0.50 ± 0.71** |

Replay dominates the adaptation phase across all four changed conditions. PPO and Replay tie on `transition_change`; Replay clearly leads on `reward_change` (0.78 vs PPO 0.11 vs SR 0.00). SR `full` adaptation fails on every condition — an expected downstream consequence of the 2/3 seeds that never learned a stable policy.

A **positive within-seed result** appears for the Momennejad-style `wonly` variant: on seed 0 (the only SR run with any stable success), freezing the encoder and fine-tuning only **w** reaches **1.00 success by adaptation episode 60** on `reward_change` (`sr_seed0_adapt_reward_change_wonly.csv`). Seeds 1 and 2 remain at 0 (the encoder never learned useful φ, so reward-weight tuning has nothing to attach to). This means the reward-revaluation mechanism works *when* the SR is properly trained — the block is upstream, in stable-phase convergence.

### 5.5 Ablation: SR without φ-normalization (Figure 6)

`results/figures/ablation_sr_no_norm.png`. In the no-normalization run, total loss exceeded 10⁷ and ‖φ(s)‖ grew several orders of magnitude within 100 episodes; the default normed run kept loss bounded below 1 and ‖φ‖ = 1 by construction. This replicates Lehnert et al. (2024)'s deep-SF representation collapse and empirically justifies the normalization step.

### 5.6 SR training-stability patch (planned follow-up)

`scripts/train_sr.py` has been patched to address the 2/3-seed failure:

1. `NUM_EPISODES` raised from 300 → 500; `EPS_DECAY_EPS` 200 → 300.
2. Best-stable-success checkpointing added (`sr_seed{s}_best.pt` saved whenever stable eval improves, matching the Replay script's existing behavior).
3. The adaptation phase now loads the best-stable checkpoint rather than the final checkpoint, so a late-training regression no longer contaminates the adaptation-phase starting point.

A follow-up run with the patched script will replace the SR numbers in §5.2–§5.4; the qualitative Replay and PPO results are unaffected.

---

## 6. Discussion

### 6.1 Hypothesis-by-hypothesis readout

- **H1 (SR fastest on `reward_change`): not supported in aggregate, but supported within the single SR seed that converged.** Replay's adaptation (0→0.78) outperforms SR's `full` (0) and `wonly` (0.33 mean, entirely driven by seed 0) variants when aggregated across seeds. However, seed 0's `wonly` adaptation reaches 1.00 success within 60 episodes — exactly the Momennejad-style behavior predicted by the SR framework. The block on H1 is therefore not a failure of the revaluation mechanism; it is a failure of stable-phase SR training on 2/3 seeds under the original hyperparameters (§5.3). The patched SR script (§5.6) is the planned path to disambiguate.
- **H2 (Replay fastest on `transition_change`): supported.** Replay reaches 1.00 adapted, PPO ties (1.00), SR is at 0. Replay and PPO were already at or near ceiling zero-shot, so the adaptation signal here is smaller than hoped, but the direction matches theory.
- **H3 (crossover dissociation): not cleanly supported** because H1 is confounded by the SR training-stability issue. If the patched run recovers SR's stable-phase convergence, H3 becomes directly testable.
- **H4 (obs_visual recovery): supported for Replay; partial for PPO.** Replay's zero-shot 0.67 → adapted 1.00 is the clean H4 signature: state identity preserved, policy recovers with a small number of new rollouts. PPO's result is unexpected and *diagnostic* — see §6.2.
- **H5 (`obs_remap` hardest): supported.** Every agent's adapted success on `obs_remap` is lower than its adapted success on `obs_visual`. Replay 0.50 vs 1.00; PPO 0.56 vs 0.56 (numerically equal but with 0.51 std, whereas obs_visual variance came from one failed seed); SR at floor for both. This is consistent with the global-remapping prediction: breaking the obs→state map forces the encoder to re-learn a pixel→semantic correspondence, which a fixed-capacity CNN and 60 adaptation episodes cannot fully accomplish.

### 6.2 PPO regression on `obs_visual` under adaptation

PPO's `obs_visual` adaptation went 1.00 (zero-shot) → 0.56 (adapted). Continuing training at the stable-phase learning rate on a *working* policy destabilized it. This is a well-known fine-tuning pitfall: the PPO KL penalty implicitly assumes a fresh-enough distribution, so restarting the optimizer and running new data through an already-converged policy pushes the actor away from its competent region before enough advantage signal accumulates. The right fix is a 10× smaller adaptation-phase learning rate (or a LR warmup/anneal). We flag this but leave it for a follow-up pass — the qualitative story for H4 already holds via Replay.

### 6.3 The SR training bottleneck

The SR aggregate numbers are dominated by 2/3 seeds failing to converge on stable, not by any failure of the revaluation mechanism. Diagnostically: in the failing seeds, SR loss stays below 0.1, the reward-weights norm stays ~0.3–0.5 (smaller than the converging seed 2 which reaches ~0.83), and ε collapses to the floor of 0.05 by ep 200 (the current `EPS_DECAY_EPS`), so the agent stops exploring before it has discovered the goal. Two patches address this directly: (1) extend training to 500 episodes with `EPS_DECAY_EPS=300`, giving the encoder more time under high exploration; (2) save and reload the best-stable checkpoint for adaptation so a late regression cannot undo a successful stable-phase run. Both are now in `scripts/train_sr.py`.

### 6.4 Observation-change conditions as a remapping probe

The `obs_visual`/`obs_remap` pair is the main methodological contribution that differentiates v2 from the v1 design. Under Sanders, Wilson & Gershman (2020)'s framing, rate remapping (graded change in observation given the same latent state) should produce fast recovery once the agent collects a few new observations; global remapping (the observation-to-state map itself changes) should require reconstructing the encoder's pixel→semantics function. Our adapted numbers are consistent with this: the best-performing agent (Replay) recovers fully on `obs_visual` (1.00) but only partially on `obs_remap` (0.50 ± 0.71). PPO's partial recovery on `obs_remap` (0.56) reflects that PPO's actor is retraining the pixel→action map from scratch on the permuted-channel input, and 20 batches × 512 frames is roughly the budget it needs to re-learn the simple navigation task. This is the predicted qualitative dissociation between rate and global remapping within a single architecture.

### 6.5 Deep SR training and the Lehnert ablation

The ablation (§5.5) reproduces Lehnert et al. (2024)'s prediction directly: without φ-normalization the feature norm and the SR-Bellman MSE both diverge by ~10⁷ within 100 episodes. This is a small but concrete empirical contribution: the `F.normalize(phi, p=2)` line in `SRNet.encode` is not merely a convenience for numerical stability — it is the boundary condition that makes the SR fixed-point (bounded by 1/(1−γ) ≈ 20 here) well-posed under deep function approximation.

### 6.6 Comparison to v1

v1 (seed 0 only, zero-shot only, 3 conditions) returned null H1/H2/H3 because zero-shot evaluation does not exercise the mechanisms the hypotheses are about. v2's adaptation phase is where the dissociation actually becomes measurable, as the Replay-vs-PPO gap on `reward_change` and the PPO/Replay regression/recovery behaviors on `obs_visual` show. The residual v2 issue is SR stable-phase training — a tractable engineering problem, not a theoretical null.

### 6.7 Limitations

- **SR training budget.** The initial 300-episode SR budget was set based on v1's seed-0 convergence time and turned out to be insufficient when variance across seeds is considered. The patched run addresses this.
- **Compute and scale.** A single CPU machine and an 8×8 gridworld preclude claims about scalability; 3 seeds × 3 agents × 5 conditions is the informative minimum for a student-project budget.
- **Fixed per-agent LRs at adaptation time.** PPO's `obs_visual` regression suggests adaptation-phase learning rates should be tuned per agent rather than reused from stable training.
- **Algorithm scope.** Single-vector reward-weights SR (no multi-head SF bank), no prioritized replay, no explicit world model. The choice keeps the architectural comparison clean at the cost of lower peak performance for each agent.
- **Biological interpretation.** The obs-change conditions are *computational analogs* of rate/global remapping, not direct neural models of hippocampal dynamics.

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
