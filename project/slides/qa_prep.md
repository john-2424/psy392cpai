# Q&A Prep -- PSY 392 CPAI Final Project

Ready-to-say answers to likely class and professor questions. Each answer is 3-5 sentences, under 60 s spoken, cites a specific number / figure / paper, and ends with a backup-slide pointer or an honest limitation. Skim before the talk; keep a printed copy on the lectern.

Organized into seven categories:

1. [Problem framing & motivation](#1-problem-framing--motivation)
2. [Class connections](#2-class-connections)
3. [Method & design choices](#3-method--design-choices)
4. [Results interpretation](#4-results-interpretation)
5. [Implementation & reproducibility](#5-implementation--reproducibility)
6. [Neuroscience interpretation](#6-neuroscience-interpretation)
7. [Future work & critique](#7-future-work--critique)

---

## 1. Problem framing & motivation

### Q1.1 -- Why these three kinds of change? Why not more?
The three axes are the ones that come packaged with named cognitive-neuroscience phenomena: reward revaluation, transition revaluation, and hippocampal remapping. They're also orthogonal in the sense that each can be manipulated without perturbing the other two. I could have added action-space change, but the literature there is thinner and I wanted every axis to come with a testable prediction. Five conditions covered all three axes without exploding compute on a laptop.

### Q1.2 -- Is this machine-learning research or neuroscience?
It's a computational analog of two neuroscience results, not a model of the brain. The question is whether the architectural priors of three canonical deep-RL agents produce adaptation behavior that lines up with the predictions from Momennejad 2017 and Sanders 2020. The gridworld is small enough to be tractable, and the neuroscience framing is what gives me clean hypotheses rather than vague "agent X adapts faster" comparisons. Call it a sanity check on whether the computational story from those papers carries over to a deep function approximator.

### Q1.3 -- Why not just one agent on one task?
The whole point is the dissociation. Any single agent on any single task will adapt in some way, and you can't attribute that adaptation to the architecture without a comparison. By crossing three architectures against five conditions, a differential pattern has to come from the architecture-condition interaction -- which is exactly what H3 is designed to test. Without the cross, the result is a single number, not a finding.

### Q1.4 -- What's new here relative to the papers you cite?
The specific combination -- a five-condition testbed that separates all three axes cleanly, the SR-`wonly` variant as a direct Momennejad analog, and the linear probe as an architectural diagnostic -- I did not find packaged together in the literature. The probe turning an ambiguous SR negative into an architectural finding is the most novel piece, at least for me. Scale-wise, nothing here is new; this is a student-scale reproduction of ideas from Momennejad, Sanders, Barreto, and Lehnert.

### Q1.5 -- Is the question ambitious enough for a master's project?
It's deliberately scoped small. Five conditions, three agents, three seeds is the informative minimum a single person on a CPU laptop can run with confidence. The value is in the architectural comparison being clean, not in scaling to Atari or DM-Lab. If the question were "does SR scale," the honest answer would be "I don't have the compute to tell you" -- and I'd rather answer a smaller question well.

---

## 2. Class connections

### Q2.1 -- Which tutorials did this build on?
Four of them, explicitly. Tutorial 05 gave the MDP vocabulary -- state, action, reward, policy, value, Q. Tutorial 06 tied value learning to the brain, which is the bridge to the Momennejad and Sanders papers. Tutorial 07 was the MLP, which is what every head on every agent ends up being. Tutorial 08 was the CNN -- the pixel-to-feature pipeline. Slide 4 lays this out visually.

### Q2.2 -- How is PPO different from the Q-learning in Tutorial 05?
Q-learning learns Q(s, a) off-policy and picks actions by argmax. PPO is on-policy actor-critic: it learns a stochastic policy pi and a value function V separately, and updates pi with a clipped-surrogate gradient that keeps each step small. For my gridworld, PPO trains very smoothly but is not as fast at absorbing new transitions as Replay, which is closer in spirit to the TD updates we saw in 05.

### Q2.3 -- Why a CNN and not an MLP on the flat grid?
Two reasons. First, Tutorial 08 gave us the CNN, so using it keeps the project continuous with the class. Second, the observation perturbations -- `obs_visual` and `obs_remap` -- are only meaningful if the agent actually processes pixels. A flat-state MLP would bypass the observation channel entirely, and you'd lose the whole remapping-analog story.

### Q2.4 -- But we only used CNNs with supervised labels in Tutorial 08. How does it work here?
Same architecture, different training signal. Here the CNN is updated indirectly: the RL loss -- PPO's clipped policy gradient, SR's Bellman-plus-reward, or Replay's TD -- provides the gradient that shapes phi(s). No labels, just reward and bootstrapping. The probe on slide 14 is how I audit what the encoder ended up learning, since I can't just read the label loss.

---

## 3. Method & design choices

### Q3.1 -- Why only 3 seeds?
CPU budget. Each SR run takes about an hour, and 3 seeds x 3 agents x 5 conditions is already on the order of 40 training runs when you include adaptation. I report std bands, but I can't claim statistical significance with three seeds. The probe metrics have lower within-seed noise than greedy-success, so the probe result on slide 14 is more robust than the error bars on the adaptation bar plots would suggest.

### Q3.2 -- Why 8x8?
Small enough to train all three agents CPU-only to convergence in under an hour each, and small enough that the linear probe on every walkable state is tractable. But large enough that `obs_visual` and `obs_remap` actually matter -- a 4x4 grid you can basically solve from memory. Ideally I'd use 16x16 for scale-robustness, but that multiplies training time by a factor I couldn't afford.

### Q3.3 -- Why 500 episodes for SR but 300 for Replay?
SR is slower to stabilize. In earlier runs at 300 episodes, SR hadn't yet reached the mid-training 1.0 peaks I needed the best-stable checkpoint to catch. I pushed to 500 in the v2 patch. Replay reaches stable success by roughly episode 150, so 300 is more than sufficient. PPO uses a frame budget instead -- 50,000 frames, which at 512 frames per batch is around 100 PPO updates.

### Q3.4 -- What would falsify H1?
If SR-`wonly` AUC on `reward_change` were at or below PPO's AUC -- same metric, same adaptation window. That's the cleanest falsification because it tests the Momennejad mechanism at its most direct. It's not what I see: SR-`wonly` is 0.38 vs PPO's 0.15, a clean positive direction. If the `wonly` variant had been zero across all three seeds, I'd have concluded the Momennejad mechanism doesn't carry over into a deep function approximator.

### Q3.5 -- Why a linear probe and not a non-linear one?
Linearity is exactly the test. A non-linear probe can in principle recover almost any information from any representation, which makes it useless as a diagnostic. A linear probe measures whether the information is in a form the agent itself can use, since every agent's head is (nearly) a linear function of phi. So the linear-probe R^2 is a lower bound on how much the downstream policy could exploit the feature.

---

## 4. Results interpretation

### Q4.1 -- Isn't the SR result just a failure?
The end-of-phase snapshot says yes. But on `reward_change_wonly`, the AUC is 0.38 against PPO's 0.15 -- that's the Momennejad direction recovered. And the probe on slide 14 shows SR's encoder is the only one that linearly encodes goal identity at 0.94 and Manhattan distance at +0.50. So SR learned the right thing, it just can't stabilize a greedy policy from it. Calling it a "failure" conflates representation quality with policy extraction, which is exactly what this project is set up to separate.

### Q4.2 -- Why does PPO regress on `obs_visual` adaptation?
Because I continued fine-tuning a working policy at the same learning rate used in stable training. The stable checkpoint was already at 1.0 success on `obs_visual`, and adapting on something the policy handled at ceiling just destabilized it. That's a standard fine-tuning pitfall. A 10x smaller adaptation-phase learning rate or a short warmup would almost certainly remove the 1.00 -> 0.56 regression. I flag this explicitly in the paper's limitations section.

### Q4.3 -- Why is Replay so dominant? Is the comparison fair?
Replay runs 2 Q-updates per env step against a 10,000-capacity buffer -- that's Dyna-style amortization, baked in by design. If I gave PPO the same amortization, it wouldn't become Replay, because on-policy gradients don't benefit from stale off-policy data. So the comparison is fair in the sense that each agent runs its idiomatic recipe. Backup 1 shows the per-condition curves if anyone wants to see Replay's trajectories in detail.

### Q4.4 -- What does "SR encodes goal-distance" mean in practice?
It means the SR encoder output phi(s) is linearly related to Manhattan-to-goal -- a linear regression on phi predicts distance with R^2 = +0.50, well above PPO (-3.1) and Replay (-1.4). Practically, SR's Q-values ought to vary monotonically with distance from the goal, which is what you want. The problem is that phi also collapses along the column axis -- R^2 = -0.15 -- so Q-values tie on horizontal moves. Correct geometry, wrong discrimination.

### Q4.5 -- Why does the AUC view disagree with the snapshot view?
Because SR-`wonly`'s greedy success is transient. Two of three seeds hit 1.0 mid-window and then regress before the final checkpoint. The snapshot only sees the endpoint; the AUC integrates over the window and gives credit for transient correctness. The methodological lesson is: for architectures that pass through the solution instead of settling in it, AUC is the honest metric. The snapshot alone is misleading.

### Q4.6 -- Did the phi-normalization ablation surprise you?
Not really -- Lehnert 2024 predicted exactly what happens. What was striking was the speed: total loss above 10^7 and phi-norm up several orders of magnitude inside 100 episodes. Backup 2 has the curves. The ablation is a small reproducibility point rather than a result, but it confirms that the SR results elsewhere in the paper aren't an artifact of a weird normalization choice -- they're all under the regime Lehnert prescribes.

---

## 5. Implementation & reproducibility

### Q5.1 -- What tools did you use?
PyTorch for the networks, TorchRL for PPO's trajectory collection and GAE, plain Python classes for the SR and Replay loops, scikit-learn for the 5-fold CV linear probe. All CPU-only -- no GPU required. The environment is a custom TorchRL `EnvBase`. Every hyperparameter is listed in Appendix A of the paper.

### Q5.2 -- How long did training take end-to-end?
PPO: about 20 min per seed. SR: ~60 min per seed (500 episodes, v3 patch). Replay: ~60 min. The adaptation phase adds another 20-30 min per seed across the four conditions. Total, three seeds per agent, is under 8 wall-clock hours on a laptop. The probe itself runs in 30 seconds -- it's cheap analysis, not extra training.

### Q5.3 -- How would a classmate reproduce this?
Appendix D of the paper lists four commands: `train_ppo.py`, `train_sr.py`, `train_replay.py`, and `train_sr_no_norm.py` for the ablation. Then `probe_representations.py` for the linear probe. Finally, the analysis notebook runs end-to-end to produce every figure and CSV. All seeded, so reruns give the same numbers within floating-point noise.

### Q5.4 -- What was the most annoying bug you hit?
The SR training patches -- v2 and v3. SR's end-of-phase greedy policy kept regressing to zero even after 500 episodes and best-stable checkpointing. Rebalancing the reward-loss coefficient from 20x to 5x reshaped what the encoder represents but didn't fix policy stability. That's what drove me to run the linear probe, which ended up being the best result in the project. So the bug I couldn't fully fix became the finding.

---

## 6. Neuroscience interpretation

### Q6.1 -- Is `obs_remap` really global remapping?
It's a computational analog, not a claim about hippocampal physiology. Global remapping in Sanders 2020's framing is the obs-to-state map being replaced. I implement that by permuting the three channels, so the CNN's learned pixel-to-semantics correspondence is broken while the state graph stays the same. That maps cleanly onto Sanders' prediction at the computational level, but I'm not modelling place cells.

### Q6.2 -- Are you claiming the hippocampus runs a deep SR?
No. The claim is narrower: if the computational story from Momennejad 2017 is right, a deep network implementing the same factorization should show an analogous signature -- and the SR-`wonly` AUC result (0.38 vs 0.15) says it partly does. The project is about whether the architecture carries the neuroscience prediction, not about neural substrate. Stachenfeld 2017 is the paper that actually argues for the SR-in-hippocampus mapping.

### Q6.3 -- Momennejad's experiment was in humans. What does a deep-network result add?
Two things. First, it tests whether the factorization they used to explain the human data actually behaves as predicted in a deep function approximator -- that's a consistency check the human data alone can't give you. Second, it exposes architectural constraints: under a linear-in-features Q-head, policy extraction can break even when the reward revaluation mechanism works. That's something you can only see from the computational side.

### Q6.4 -- What would change if you used a tabular SR instead of a deep one?
Tabular SR would very likely pass H1 cleanly -- `wonly` should settle at 1.0 without the regression I observe. That would match Momennejad's human result directly. I stayed with deep SR precisely to test whether the prediction scales to function approximators, which is the harder case and the one relevant to modern deep RL. Stachenfeld 2017 does the tabular / place-cell version if you want that comparison.

---

## 7. Future work & critique

### Q7.1 -- What would you do with more compute?
Three things. A 16x16 grid, to confirm the architectural signatures aren't 8x8-specific. Ten seeds instead of three, for actual statistical significance. And a per-agent adaptation-phase LR schedule, which would almost certainly remove PPO's `obs_visual` regression. None of these change the direction of the findings; they'd let me make stronger claims.

### Q7.2 -- What's the single fix most likely to close the SR gap?
Action-conditioned features -- phi(s, a) branches rather than a shared phi(s) routed through an action-indexed SR head. Barreto 2017 and Lehnert 2024 both argue for this. The mechanism: action discrimination lives in the representation itself, which directly targets the column/row asymmetry my probe exposed. A Q-margin regularizer would be the second fix -- force a confident argmax even where the successor geometry is already correct.

### Q7.3 -- What's a good extension for a classmate next semester?
Implement phi(s, a) and rerun the same five-condition protocol. If SR's greedy policy stabilizes, the story becomes a clean cross-agent dissociation -- H3 recovered. If it doesn't, the explanation shifts from policy extraction to something else, maybe critic capacity. Either way, the project becomes a study of when deep SR actually delivers the Momennejad advantage, which is the unfinished question here.

---

## Printed day-of cheats

- **If a question starts with "why did you..."**: answer what you did, then why it was the minimum-viable choice given the CPU budget.
- **If a question is about scale / generalization**: acknowledge the 8x8, 3-seed limit openly, then redirect to the architectural question the project actually answers.
- **If a question is hostile or you don't know the answer**: "Honest answer: I don't know. My guess is X because Y, but that's a guess." That's far better than bluffing.
- **If time is running out in Q&A**: "Happy to follow up on that after class or over email -- my address is on slide 1."
