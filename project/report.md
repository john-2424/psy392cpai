# Vision-Based Reinforcement Learning for Flexible Navigation Under Reward and Transition Changes

**Shrikrishna Rajule** | PSY 39200 CPAI | Spring 2026 | Purdue University

---

## 1. Introduction

A hallmark of biological intelligence is behavioral flexibility: the ability to rapidly adjust actions when goals change or the environment shifts. In reinforcement learning (RL), this flexibility depends critically on the agent's internal representation and learning algorithm. Model-free agents cache state-action values that must be slowly relearned after any perturbation, while structured representations such as the Successor Representation (SR; Dayan, 1993) decouple environment dynamics from reward, potentially enabling faster adaptation.

This project compares three RL architectures on a simple vision-based gridworld navigation task and evaluates their ability to generalize when (a) the reward function changes (goal relocation) or (b) the transition dynamics change (wall configuration shifts).

### Hypotheses

| ID | Prediction |
|----|------------|
| **H1** | The SR agent adapts faster to reward changes than model-free baselines, because only reward weights **w** need updating while successor features psi remain valid. |
| **H2** | The Replay/Dyna agent adapts better to transition changes than the SR agent, because replayed experience from the new layout provides direct training signal, whereas the SR Bellman equation assumes fixed transitions. |
| **H3** | A crossover interaction: SR outperforms Replay on reward changes; Replay outperforms SR on transition changes. |

---

## 2. Methods

### 2.1 Environment

An 8x8 gridworld implemented as a TorchRL `EnvBase` subclass. The agent receives a 3-channel binary image observation (channels: agent position, goal position, walls). Four discrete actions (up, down, left, right). Reward is +1.0 for reaching the goal and -0.01 per time step. Episodes terminate on goal arrival or after 50 steps.

Three evaluation conditions share the same start position (6, 1):

- **Stable:** Goal at (1, 6), walls at {(2,3), (3,3), (5,3)}
- **Reward Change:** Goal relocated to (1, 1), same walls
- **Transition Change:** Goal at (1, 6), walls shifted to {(2,3), (4,3), (5,3)}

The transition change moves one wall from row 3 to row 4 in column 3, creating a different gap pattern. The reward change moves the goal from the top-right to the top-left of the grid.

### 2.2 Agents

**PPO (Proximal Policy Optimization).** On-policy actor-critic baseline using TorchRL's PPO loss module. CNN encoder (2 conv layers) feeds separate policy and value heads. Trained for 50,000 frames with batch size 512, 4 PPO epochs per batch, gradient clipping at 1.0.

**Successor Representation (SR).** The SR factorizes the Q-function as Q(s,a) = psi(s,a)^T w, where psi(s,a) are successor features satisfying the Bellman equation psi(s,a) = phi(s) + gamma * psi(s', pi(s')), and **w** are learned reward weights. Architecture: shared CNN encoder -> L2-normalized features phi(s) -> successor feature head -> psi(s,a). Feature normalization bounds the SR fixed-point and prevents loss explosion. Trained with experience replay (capacity 5000), epsilon-greedy exploration, soft target network updates (tau=0.05), gamma=0.95, and gradient clipping at 1.0.

**Replay/Dyna (DQN with experience replay).** Standard DQN with a large replay buffer (capacity 10,000) and 2 Q-learning updates per environment step, approximating Dyna-style planning. CNN encoder -> Q-value head. Trained for 300 episodes with epsilon decay and soft target updates (tau=0.01).

All agents are trained exclusively on the **stable** condition and evaluated (zero-shot, no further learning) on all three conditions.

### 2.3 SR Reward Revaluation Phase

After stable-environment training, the SR agent undergoes a 20-episode revaluation phase on the **reward_change** environment: the encoder and SR head are frozen, and only the reward weights **w** are fine-tuned. This tests whether the learned successor features support fast goal revaluation.

### 2.4 Evaluation Protocol

Every 25 episodes (SR/Replay) or 10 batches (PPO), the agent is evaluated for 20 greedy episodes (epsilon=0) on each condition. Success rate, average return, and average steps are recorded.

---

## 3. Results

### 3.1 Training Convergence

All three agents learn to navigate the stable environment:
- **PPO** converges within the first 5,000 frames (~10 batches), achieving 100% success rate from the earliest evaluation checkpoint.
- **Replay** reaches 100% success at episode 150, with a brief regression at episode 200 before recovering by episode 225.
- **SR** converges later, achieving 100% success at episodes 275-300. The slower convergence reflects the additional complexity of learning successor features (a 64-dimensional prediction per state-action pair) compared to scalar Q-values.

Training losses remain stable throughout for all agents. The SR loss (after phi normalization) stays in the 0.01-0.05 range; the Replay Q-loss decreases to ~0.001; PPO loss follows the typical non-monotonic PPO trajectory.

### 3.2 Zero-Shot Generalization

| Agent | Stable | Reward Change | Transition Change |
|-------|--------|---------------|-------------------|
| **PPO** | 1.00 | 0.00 | 1.00 |
| **SR** | 0.67 | 0.00 | 0.67 |
| **Replay** | 1.00 | 0.00 | 0.00 |

*(Mean success rate over the last 3 evaluation checkpoints)*

**Reward Change:** No agent achieves non-zero success. This is expected: all agents were trained with the goal at (1,6) and have never experienced the (1,1) goal. Without any adaptation mechanism active during evaluation, the learned policies direct the agent toward the old goal location.

**Transition Change:** PPO achieves 100% success. SR achieves 100% at the final two checkpoints (67% average over last 3). Replay achieves 0%. The key distinction: the policies learned by PPO and (late-stage) SR happen to route through the gap at row 4 in column 3, which exists in both wall configurations. The Replay agent learned a path that passes through the row-3 gap, which is blocked in the transition-change condition.

### 3.3 SR Reward Revaluation

The 20-episode revaluation phase (freezing encoder + SR head, fine-tuning only **w**) produced 0% success on the reward-change environment at both evaluation points (episodes 10 and 20). The successor features learned during stable-environment training did not capture sufficient state-transition structure for the reward weights alone to redirect the policy toward the relocated goal.

---

## 4. Discussion

### 4.1 Hypothesis Evaluation

**H1 (SR advantage on reward changes): Not supported.** The deep SR agent could not revalue rewards through weight-only fine-tuning. This contrasts with predictions from the tabular SR literature, where revaluation is immediate and exact (Russek et al., 2017). The deep SR's features are learned via function approximation and are biased toward the training trajectory, meaning psi(s,a) is only accurate for states and actions frequently visited during stable-environment training. States near the relocated goal at (1,1) were rarely visited, so the successor features provide no useful signal for directing the policy there.

**H2 (Replay advantage on transition changes): Not supported as predicted.** Replay showed 0% success on transition changes, while PPO and SR achieved high success. However, this reflects path-specific generalization rather than a mechanistic advantage: PPO and SR learned paths that happened to avoid the affected wall, while Replay's path went through it. The evaluation is zero-shot (no further training), so the replay buffer provides no advantage.

**H3 (Crossover interaction): Not observed** due to the limitations above.

### 4.2 The Deep SR Training Challenge

A significant finding of this project is the difficulty of training deep SR networks. The initial implementation suffered from catastrophic loss explosion (reaching 10^9 within 30 episodes), caused by:

1. **Unbounded encoder features:** Without L2 normalization, phi(s) values grew during training, causing the SR Bellman target to diverge. Adding feature normalization (||phi|| = 1) bounded the fixed-point at ||psi*|| <= 1/(1-gamma).

2. **Moving target instability:** Using the main network's phi in the SR target (rather than the target network's phi) created a non-stationary optimization objective. Using the target network for the entire target computation resolved this.

3. **Gradient scale mismatch:** The SR loss is an MSE over 64-dimensional vectors, producing much larger gradients than the scalar Q-loss used in standard DQN. Gradient clipping was essential.

These engineering challenges are underreported in the SR literature, which primarily works with tabular or simple linear function approximators.

### 4.3 Limitations

1. **Single seed:** All results use seed 0. Variance across seeds could change which agents generalize to which conditions.
2. **Zero-shot evaluation:** The experimental design tests generalization without adaptation, which does not directly test the revaluation/replay mechanisms. A proper test would involve a few-shot adaptation phase where agents continue learning in the changed environment.
3. **Small environment:** The 8x8 gridworld is simple enough that PPO memorizes an optimal policy almost instantly. A larger, more complex environment would better reveal representational advantages.
4. **Brief revaluation phase:** 20 episodes of reward weight fine-tuning may be insufficient. Longer revaluation with higher learning rate and more exploration could potentially show SR advantages.

### 4.4 Implications for Cognitive Models

Despite the negative results on the dissociation hypothesis, this project highlights an important gap between theoretical predictions of the SR framework and practical deep-learning implementations. The hippocampal SR theory (Stachenfeld et al., 2017) assumes that successor representations are learned over many episodes of free exploration, producing accurate psi estimates across all reachable states. In contrast, our deep SR agent (like a real organism with limited experience) only develops accurate psi along frequently-traveled paths.

This suggests that biological reward revaluation may depend not just on having an SR-like representation, but also on the quality of exploration during initial learning. Hippocampal replay during sleep (Olafsdottir et al., 2018) may serve precisely this function: broadening the coverage of successor representations to support future flexibility.

---

## 5. Conclusion

We implemented and compared three RL architectures (PPO, Successor Representation, and Replay/Dyna) on a vision-based gridworld with reward and transition perturbations. While all agents learned the training environment, the predicted dissociation between SR (advantage on reward changes) and Replay (advantage on transition changes) was not observed in the zero-shot evaluation setting. The most significant technical contribution is identifying and resolving the training instabilities inherent in deep SR networks (feature normalization, target network stabilization, gradient clipping). Future work should extend the experimental design to include few-shot adaptation phases and test on larger environments with multiple seeds.

---

## References

- Barreto, A., et al. (2017). Successor features for transfer in reinforcement learning. *NeurIPS*.
- Daw, N. D., et al. (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience*.
- Dayan, P. (1993). Improving generalization for temporal difference learning. *Neural Computation*.
- Momennejad, I., et al. (2017). The successor representation in human reinforcement learning. *Nature Human Behaviour*.
- Olafsdottir, H. F., et al. (2018). The role of hippocampal replay in memory and planning. *Current Biology*.
- Russek, E. M., et al. (2017). Predictive representations can link model-based reinforcement learning to model-free mechanisms. *PLoS Computational Biology*.
- Stachenfeld, K. L., et al. (2017). The hippocampus as a predictive map. *Nature Neuroscience*.
- Sutton, R. S. (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *ICML*.
