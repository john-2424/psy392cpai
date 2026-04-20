# Speaker transcripts -- 10 min talk + 2 min Q&A

Format: per-slide word-for-word spoken script, 80-100 words each, ~1300 words total, ~10 min at 130 wpm. Each script opens with the slide's takeaway, lands a memorable number or phrase, carries a class- or paper-callout where the plan specifies, includes one `[pause]` marker, and ends with a hand-off to the next slide. Backup-slide notes are at the bottom, followed by a timing card and day-of cues.

---

## Slide 1 -- Title (15 s)

Hi, I'm Shrikrishna. For my final project I wanted to see whether three different deep-RL architectures actually show the adaptation signatures that cognitive neuroscience predicts when the world changes in three different ways. [pause] Let me walk you through the question, what I built, and what I found.

## Slide 2 -- The question: three ways the world can change (45 s)

When the world changes, an agent has to change with it -- but the kind of change really matters. The reward can move. A wall can appear, changing the dynamics. Or the pixels can just look different without the underlying state changing at all. These three axes map onto three distinct phenomena in cognitive neuroscience: reward revaluation, transition revaluation, and hippocampal remapping. [pause] The five small pictures at the bottom are the five conditions in my environment -- three of them change the layout, two of them only change the observation. That's the whole setup in one slide. Next, the papers these conditions are built from.

## Slide 3 -- What the neuroscience says (50 s)

Two papers anchor this whole project. Momennejad et al., 2017, in Nature Human Behaviour, showed that humans revalue fast after reward change but slow after transition change. Their explanation is the Successor Representation -- you factor value into a reward-agnostic occupancy map psi, and reward weights w; reward revaluation is just updating w. Sanders, Wilson and Gershman, 2020, in eLife, take hippocampal remapping and recast it as hidden-state inference -- rate remapping is graded change under the same state, global remapping is the observation-to-state map itself being replaced. [pause] My five conditions directly operationalize both frameworks.

## Slide 4 -- Where this builds on the class (40 s)

Here's how this project sits on top of what we covered in class. Tutorial 05 gave me the MDP vocabulary -- state, action, reward, policy, value. Tutorial 06 tied value learning to the brain, which is my bridge into Momennejad and Sanders. [pause] Tutorial 07 and 08 gave me the function approximators: MLP heads on top of a CNN encoder. So this project takes those building blocks and snaps three different credit-assignment heads onto the same encoder to compare them.

## Slide 5 -- Three agents, one shared CNN encoder (55 s)

This is the conceptual scaffold for the whole project. A shared CNN encoder turns the 3-by-8-by-8 observation into a feature vector phi of s. On top of that sit three heads. The PPO head is the on-policy actor-critic from Schulman 2017 -- it learns pi and V directly. The SR head, following Dayan 1993 and Momennejad 2017, learns psi of s and a and w, and reconstructs Q as their dot product. The Replay head is DQN with a large experience buffer, Sutton 1990 and Mnih 2015 style -- fastest at absorbing new transitions. [pause] Same encoder. Three very different mechanisms.

## Slide 6 -- Successor Representation in one equation (55 s)

The entire SR trick is this one equation -- Q equals psi dot w. Psi is "where you'll go": the discounted sum of future features, and it's reward-agnostic. W is "what you want": the reward weights. If the reward changes, you don't have to re-learn psi; you just update w. That's Momennejad 2017's prediction, and my `wonly` variant is the direct analog -- encoder and SR head frozen, only w trained. [pause] If SR is doing what it should, `wonly` should recover the reward-change condition faster than a model-free baseline.

## Slide 7 -- Environment: five conditions (35 s)

Here are the five conditions laid out. The first three differ in goal or wall placement -- the layout itself changes, channel semantics stay intact. The last two keep the layout constant but perturb what the CNN sees. `obs_visual` adds a noise mask -- that's the rate-remapping analog. `obs_remap` permutes the three channels -- the global-remapping analog. [pause] State graph unchanged, but the pixel-to-semantics map is broken.

## Slide 8 -- Five hypotheses (45 s)

These are the five predictions I went in with. H1: SR should adapt fastest on reward change, per Momennejad. H2: Replay should adapt fastest on transition change, per Sutton's Dyna. H3 is the critical test -- the crossover dissociation between SR and Replay across the first two conditions. [pause] H4 says all agents drop on `obs_visual` zero-shot but recover with a few-shot adaptation -- that's rate remapping. H5 says `obs_remap` is strictly harder -- global remapping. Every results slide from here on maps back to one of these.

## Slide 9 -- Protocol and metrics (45 s)

Two training phases per agent, per seed. First, train on `stable` to convergence and do zero-shot evaluation on all five conditions. Then reset the optimizer and do a short few-shot adaptation on each changed condition -- 20 batches for PPO, 60 episodes for SR and Replay. [pause] Three seeds, three agents, five conditions. I track four metrics: greedy success rate at snapshot points, AUC over the adaptation window, t-half -- first step where success crosses 0.5 -- and a linear probe of what each CNN has actually learned.

## Slide 10 -- Stable training converges (40 s)

Before adaptation, everything has to converge. PPO hits 1.00 stable-eval success within about ten batches. Replay gets there by roughly 150 episodes. [pause] SR is stranger -- all three seeds hit transient 1.0 peaks mid-training, but they regress to zero by episode 500. The loss is bounded and phi is normalized, so this isn't training collapse. The best-stable checkpointing catches those peaks -- and that's where the adaptation phase starts from. Next I'll show what happens immediately after stable training, before any adaptation.

## Slide 11 -- Zero-shot: who generalizes? (45 s)

Zero-shot -- frozen weights, twenty greedy episodes per condition. PPO is at ceiling on `stable`, `transition`, and `obs_visual`; floor on `reward_change` and `obs_remap`. Replay looks similar in direction but with more seed variance -- two of three seeds generalize to `transition` and `obs_visual`; floor on `reward_change` and `obs_remap`. SR sits at or near floor on most conditions, with small transient positive signals. [pause] The interesting case is `reward_change`: both model-free agents drop to zero, because their policy was shaped for the old goal. That's exactly the setup where SR's revaluation story should pay off. Let's see if it does.

## Slide 12 -- Few-shot adaptation (50 s)

Adapted-phase snapshot. Replay dominates across the board: on `transition_change` and `obs_visual` it's at or near ceiling, on `obs_remap` it's at 0.50. PPO ties Replay on `transition_change`, which is H2 in the right direction even if it's not a Replay-only win. H4, `obs_visual` recovery, is clean for Replay -- 0.67 to 1.00 -- and partial for PPO, which actually regresses from 1.0 down to 0.56 because I fine-tuned at the stable-phase learning rate. [pause] H5, `obs_remap` harder than `obs_visual`, is clean for Replay (0.50 versus 1.00). PPO ties numerically, and SR sits at or near floor for both so the ordering isn't informative. SR-full end-of-phase numbers are at floor across the board. So what about H1?

## Slide 13 -- The Momennejad signal hiding in the data (55 s)

This is the plot I'm proudest of. If you only look at end-of-phase snapshots, SR on `reward_change` is zero, and you'd reject H1. But the per-seed picture is more interesting -- SR-`wonly` hits 1.0 on two of three seeds. One of those seeds actually holds the peak through the whole adaptation window, one peaks then regresses, and one never lifts. AUC captures all three. [pause] SR-`wonly` AUC averages 0.38, above SR-full's 0.12 and PPO's 0.15, and approaches Replay's 0.44. The Momennejad direction is there -- hidden when you average a sustained 1.0 with a regression and a zero. H1 partially supported. The right metric for this hypothesis was AUC, not a snapshot.

## Slide 14 -- The probe, and the diagnosis (55 s)

I was worried the SR failure meant bad features. The probe says the opposite. SR's encoder is the only one that linearly decodes goal identity at 0.94 accuracy, and Manhattan distance at plus 0.50. PPO's goal-identity probe sits just above chance at 0.59, Replay is below chance at 0.37, and both are badly negative on Manhattan. [pause] SR's agent-column R-squared is minus 0.15 -- worse than predicting the mean. That's the trade-off. The optimal path from 6,1 to 1,6 needs column discrimination, and SR's objective has compressed it away. Q-values tie along the column axis, so you get repeated 1.0 peaks instead of a persistent policy. That's a policy-extraction problem, not a representation problem. The fix is action-conditioned features -- Barreto 2017, Lehnert 2024 -- or an explicit Q-margin term.

## Slide 15 -- Takeaways and what's next (40 s)

Five-condition testbed cleanly separates three axes of change. H2, H4, and H5 come out in the predicted direction -- most cleanly for Replay. H1 is supported once you pick the right metric; H3 doesn't replicate but it's explained. The representation probe turned what would have been an ambiguous SR negative into a diagnostic one. [pause] Future work: action-conditioned phi, a Q-margin term, and adaptation-phase LR tuning for PPO. Thank you -- I'm happy to take questions.

---

## Backup slides (on-demand, only if asked)

### Backup 1 -- Full adaptation grid

This is the per-(agent, condition) adaptation curve for every seed. The transient 1.0 peaks on SR-full and SR-`wonly` are the thing AUC picks up and the end-of-phase snapshots miss.

### Backup 2 -- $\phi$-normalization ablation (Lehnert 2024)

Without $\ell_2$-normalizing phi, the total loss exceeds 10^7 and $\|\phi\|$ climbs by several orders of magnitude within 100 episodes. That's the representational-collapse mode Lehnert 2024 predicted -- which is why every other SR result in this project uses the normalized variant.

---

## Timing card (rehearsal)

| Slide | Target | Cumulative |
|---|---|---|
| 1  | 0:15 | 0:15 |
| 2  | 0:45 | 1:00 |
| 3  | 0:50 | 1:50 |
| 4  | 0:40 | 2:30 |
| 5  | 0:55 | 3:25 |
| 6  | 0:55 | 4:20 |
| 7  | 0:35 | 4:55 |
| 8  | 0:45 | 5:40 |
| 9  | 0:45 | 6:25 |
| 10 | 0:40 | 7:05 |
| 11 | 0:45 | 7:50 |
| 12 | 0:50 | 8:40 |
| 13 | 0:55 | 9:35 |
| 14 | 0:55 | 10:30 |
| 15 | 0:40 | 11:10 |

Target landing: 10:00 +/- 30 s. Frames 13 and 14 are the longest; if running over by slide 12, shorten slide 5 (cut the "same encoder, three very different mechanisms" sentence) or slide 9 (skip the metric names; say "four metrics, all on the handout").

---

## Five cues for the day-of

1. **Opening line that steadies nerves.** "For my final project I wanted to see whether..." -- state the question first, before any name-drops.
2. **The one number per act.** Act 1 (setup): "three axes, five conditions." Act 2 (results): "SR-`wonly` AUC 0.38 beats PPO 0.15." Act 3 (diagnosis): "SR encoder is the only one at 0.94 goal identity, and minus 0.15 on agent column."
3. **Recovery line if a slide runs long.** "I'll skip the detail here -- happy to come back to it in Q&A." Keeps momentum.
4. **If a question mid-talk:** "Great question -- can I park it and take it at the end so the story stays linear?"
5. **Close with the paper, not the deck.** "The full write-up is a NeurIPS-format PDF -- happy to share it if anyone wants the per-seed tables." Signals confidence; points away from slide text for the last 30 s.
