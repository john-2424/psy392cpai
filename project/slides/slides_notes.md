# Speaker notes -- 10 min talk + 2 min Q&A

Rough budget: 14 content slides at ~40s each. Land the bolded number on every slide; do not read bullets verbatim. Backup slides only on request.

---

## Slide 1 -- Title (15 s)

"Hi, I'm Shrikrishna. For my final project I asked: when the world changes in one of three qualitatively different ways, do three different deep-RL architectures show different adaptation signatures, in the way cognitive neuroscience predicts? I'll walk through the design, the main results, and what I think the interesting negative is."

## Slide 2 -- Motivation (45 s)

"Three kinds of change: the reward can move; the dynamics can change; or the observation can shift without the underlying state changing. These three have very different cognitive signatures. Humans revalue fast after reward change, slower after transition change. Hippocampal neurons remap when the observation changes but the environment is otherwise stable. Standard RL benchmarks don't factor cleanly along these three axes -- my project builds one that does."

**Transition:** "Let me anchor the two neuroscience results I'm responding to."

## Slide 3 -- Neuroscience anchor (50 s)

"Momennejad 2017 tested reward vs transition revaluation behaviorally in humans, and argued the result is explained by the Successor Representation: if Q factorizes as $\psi$ times reward-weights, reward revaluation is just updating the weights. Sanders Wilson Gershman 2020 take hippocampal remapping -- the observation-change axis -- and recast it as hidden-state inference: rate remapping is graded change under the same state; global remapping is the observation-to-state map itself changing. My five conditions operationalize exactly these two frameworks."

## Slide 4 -- RL recap (40 s)

"Quick recap from Tutorial 05: state, action, reward, policy, Q-value. Tutorial 08 gave us the CNN: pixels in, feature vector out. Everything I did uses this pipeline. The one number to remember: reward is $+1$ at goal, $-0.01$ per step, episodes cap at 50 steps."

## Slide 5 -- Three agents, one encoder (55 s)

"This is the conceptual scaffold for the whole project. Same CNN, three heads. PPO: on-policy actor-critic -- it just learns $\pi$ and $V$ directly. SR: successor features -- it learns $\psi$ of state and action, and reward-weights, and reconstructs Q as their dot product. Replay / Dyna: DQN with a large experience buffer and twice-per-step Q-updates -- it's the fastest at absorbing new transitions. Each one has a predicted adaptation signature: SR on reward change, Replay on transition change, and observation-visual recovery on anything that doesn't destroy state identity."

## Slide 6 -- SR intuition (55 s)

"The one equation: Q equals psi dot w. Psi is the discounted sum of future feature visits -- reward-agnostic. W is the reward weights. If you change the reward, you don't have to re-learn psi; you just update w. This is the Momennejad prediction. My SR agent has a `wonly` variant that implements exactly this: freeze the encoder and SR head, train only w. That's the clean analog of the human experiment."

## Slide 7 -- Five conditions (35 s)

"Five panels. First three: goal and walls visible in the image, so they look different. Last two: same layout as stable, but the agent channel has distractor pixels (`obs_visual`) or the three channels are permuted (`obs_remap`). Rate vs global remapping."

## Slide 8 -- Hypotheses (45 s)

"Here are the five predictions I'm testing. H1 is the SR reward story, H2 is the Replay transition story, H3 is the dissociation between them, H4 is the rate-remapping recovery, H5 is that global remapping is strictly harder. Every results slide maps back to one of these."

## Slide 9 -- Protocol and metrics (45 s)

"Two phases. Train on stable, do zero-shot evaluation on all five conditions, then reset the optimizer and do a short few-shot adaptation phase. 3 seeds, 3 agents, 5 conditions. Four metrics: greedy success rate, AUC over the adaptation window, median time-to-half-success, and a linear probe of what each agent's CNN features actually encode."

## Slide 10 -- Results I (50 s)

"Adapted greedy success. Replay dominates: transition change, obs visual, obs remap -- all three -- it's at or near ceiling. PPO ties on transition change. H2, H4, H5 are all supported cleanly from this one plot. The one surprise is PPO's regression on obs visual: it went $1.00$ zero-shot to $0.56$ adapted. That's the known PPO-fine-tuning-LR problem -- I flag it in the paper."

## Slide 11 -- Results II, H1 (55 s)

"The end-of-phase snapshot says Replay beats PPO beats SR, and SR is zero. If I stopped here, H1 is rejected. But the adaptation curve is transient -- the SR-wonly variant hits $1.00$ and then regresses. AUC captures this: SR-wonly $0.38$ beats PPO $0.15$ and approaches Replay $0.44$. The Momennejad direction is there, it just requires the right metric. H1 partially supported."

## Slide 12 -- Probe (55 s)

"I was worried the SR failure meant bad features. The linear probe says the opposite: SR's encoder is the only one that linearly encodes goal identity -- $0.94$ accuracy -- and distance to goal, $R^2$ of $+0.50$. PPO and Replay are random or worse on those. But SR's encoder is $-0.15$ on agent column -- below chance. That's the trade-off."

## Slide 13 -- Diagnosis (55 s)

"Putting it together: SR's features are \emph{correctly shaped} by the successor objective. But the optimal path needs column discrimination, which the SR encoder has given up. So Q-values tie along the column axis. Greedy argmax: you see repeated $1.0$ peaks, not a persistent plateau. That's a policy-extraction problem, not a representation problem. The fix is action-conditioned features -- $\phi(s, a)$ branches instead of one $\phi(s)$ -- or an explicit Q-margin loss. That's what Barreto 2017 and Lehnert 2024 both recommend, and it's out of scope here."

## Slide 14 -- Takeaways (40 s)

"Five-condition testbed cleanly separates three axes of change. H2, H4, H5 clean; H1 once you pick the right metric; H3 doesn't replicate but it's explained. The representation probe turned an ambiguous SR result into a diagnostic one. Future work: action-conditioned $\phi$, Q-margin term, and a proper adaptation-phase LR schedule. Thanks -- happy to take questions."

---

## Q&A prompts I'm prepared for

- "Why only 3 seeds?" $\rightarrow$ CPU budget; the probe is within-seed averaged so its signal-to-noise is higher than the greedy-success variance suggests.
- "Why not use a larger environment?" $\rightarrow$ 8x8 lets me run all three agents CPU-only and keep the architectural comparison clean. Scaling is follow-up work.
- "Did you try prioritized replay?" $\rightarrow$ No -- kept the algorithm scope tight so the per-axis signatures aren't confounded by implementation differences. See Limitations.
- "What would falsify the policy-extraction hypothesis?" $\rightarrow$ If a $\phi(s,a)$ variant still showed the same transient-peak pattern, the explanation would shift to inadequate critic capacity or a fundamentally different issue.
