# TODO / Roadmap

Updated 2026-08-27. Prioritized, not sequenced by date. "Critical" blocks understanding current
results or represents the project's top open problem; "High" is the next real work; "Medium" is
valuable but not blocking; "Research/Future" is speculative scope, not committed work.

**Since the last pass**: the project moved from diagnosis to intervention on its top problem.
`AUX_DIR_COEF=0.01` (search-direction auxiliary actor loss) was implemented, validated (3/3 seeds
improve `target_lost_rate` at both checkpoints), and **adopted** — the first fix, not just
localization, in this whole investigation. A no-retraining audit of the new baseline found 26%
`target_lost` remains, concentrated in late, first-loss-event-fatal failures. Two follow-ons built
from that audit — `AUX_DIVERSIFY` and `AUX_DIR_RAMP` — were tested and **both rejected**. The
"design and test a fix" item below is updated accordingly: it's no longer "no fix has been
attempted," it's "the two most directly evidence-motivated ideas have been tried and failed —
the next one needs to be genuinely different, not a variant of the shared-direction pull."

## Critical

- **Design and test a fix for the remaining 26% `target_lost_rate`.** Root cause remains the
  actor's own search execution (decisively localized last pass: a scripted controller reacquires
  100% of the time where PPO's own trajectory failed 100% of the time, from identical states).
  `AUX_DIR_COEF=0.01` (a small, always-on nudge toward `unit(_last_known_vel)`) fixed part of this
  and is adopted. Two direct follow-ons from the failure-mode audit have since been tried and
  **rejected**: diversifying the pull direction per agent (`AUX_DIVERSIFY`, mixed result) and
  scaling the pull's strength up with urgency (`AUX_DIR_RAMP`, a clean 3/3-seed regression — see
  `DECISIONS.md` for the leading hypothesis: the pull target gets staler with time, so escalating
  commitment to it as urgency rises is backwards). **Whoever picks this up next should read
  `EXPERIMENT_LOG.md`'s three AUX-intervention entries before proposing a fourth idea** — a
  genuinely different mechanism is needed, not another variant of "pull harder/differently toward
  the same shared cue." Untested candidate directions, both from before `AUX_DIR_COEF` existed and
  still open: (a) a credit-assignment change that makes a productive search action more clearly
  attributable across the many steps between committing to a heading and actually reacquiring,
  since standard per-step PPO credit assignment may be too diffuse over a ~100+-step search; (b)
  something that addresses the first-loss-event-fatal signature specifically (67.9% of failures
  never get a second attempt) rather than the per-step direction. This is an open question for the
  user — discuss before implementing in `envs/formation_env.py` or `training/train.py`. See
  `EXPERIMENT_LOG.md`'s decisive counterfactual and AUX-intervention entries, `KNOWN_ISSUES.md`
  item 12.
- **Get any `NUM_AGENTS=3` Phase-3-era checkpoint to learn at all.** A real deployment blocker:
  `deployment/inference_node.py` hard-requires `NUM_AGENTS=3`, and every `N=3` attempt so far
  (2 under the reverted relative-velocity brake, 2 more under the restored working brake at two
  different network sizes) has come back flat at ~100% `target_lost_rate` from the very first
  rollout. Untried: a smaller network specifically for `N=3` (its `OBS_DIM` is smaller than
  `N=4`'s; only the same two sizes tested at `N=4` were tried), more training steps, or testing
  `N=3` against the pre-Phase-3 mechanism set (known to have worked) to isolate which specific
  Phase 3 mechanism, if any single one, is responsible. See `KNOWN_ISSUES.md` item 18.

## High

- **The collision-safety cost under deterministic execution (8-14% on well-converged checkpoints)
  is still real and still unaddressed.** The fix built for it (the relative-velocity brake) was
  reverted for an unrelated reason (it broke training convergence) — the underlying safety
  question was never actually resolved either way. Once the actor-execution fix above exists and
  produces new well-converged checkpoints, worth re-measuring whether this cost is still present
  before deciding whether it needs its own fix attempt (e.g. the previously-rejected
  reciprocal/50-50-split alternative in `DECISIONS.md`).
- **Verify the vertical-jiggling fix actually worked.** Unchanged across several passes now —
  `CRUISE_ALT_MIN`/`CRUISE_ALT_COEF`/`Z_SMOOTHING_ALPHA` are implemented and smoke-tested, but
  nobody has loaded a post-fix checkpoint and re-measured the original 8-25% sign-flip rate. See
  `KNOWN_ISSUES.md` item 14.
- **Update or replace `readme.txt`** — more stale than ever, still describes a removed per-drone-
  file save scheme, silent on everything from the brake onward. See `KNOWN_ISSUES.md` item 2.
- **Fix `envs/formation_env.py`'s remaining stale docstring paragraphs.** The "SCOPE" note
  claiming `TARGET_DIST` is "not a general-N formula... a 4-agent study" is still present and
  still contradicted by `_PACKING_RATIO`'s actual `{2,3,4}` support. Low priority, cosmetic,
  flagged repeatedly across several doc passes without being fixed.

## Medium

- **UWB-realistic neighbor sensing**, replacing the still-exact-ground-truth neighbor
  observations. Deliberately deferred when vision-tracking was built (see `DECISIONS.md`) — real
  UWB gives clean range, not clean bearing/velocity, so this deserves its own careful design
  pass. Still the natural next realism step.
- **Sensing noise/occlusion on the target reading itself**, and a directional FOV cone (needs
  heading/orientation as a new state dimension first). See `KNOWN_ISSUES.md` item 11.
- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-sum
  switch) before comparing `r_track`/`r_safety`/etc. magnitudes between very old logs and current
  ones. Still unverified whether any analysis has actually needed to account for this.

## Research / Future (not committed, exploratory)

- A genuinely decentralized neighbor-discovery mechanism for the *graph maintenance* itself
  (separate from UWB-realistic sensing above) — `_repair_connectivity` still uses centralized
  ground-truth position knowledge. See `KNOWN_ISSUES.md` item 4.
- An actor architecture that's agent-count-invariant (e.g. attention over a variable number of
  neighbors), enabling actual weight transfer across the `NUM_AGENTS` curriculum instead of
  from-scratch comparative validation at each stage. See `DECISIONS.md`/`KNOWN_ISSUES.md` item 6.
  Would also change the CPU-vs-GPU calculus — GPU has been consistently slower for every network
  size tried so far, including the network-capacity sweep's largest (256/512); this is the kind
  of architecture change that could actually flip that.
- Real drone dynamics (inertia/acceleration limits, actuator delay) instead of the current
  direct-velocity-command kinematic model — still flagged as a "not okay for claiming realistic
  drone collision avoidance" simplification by an earlier supervisor review. Directly relevant to
  the `deployment/` workstream's own findings about zero-inertia mismatch — worth reading
  `deployment/docs/PHASE2_HANDOFF.md` before starting this.
- A learned nominal controller + safety-filter architecture more broadly (the closing-speed brake
  and ground clamp are two narrow instances of this pattern now, not one) — a supervisor review's
  broader "control barrier functions"/"action projection" recommendation is more adopted than
  before but still not a general framework.
- "Genetic/evolutionary" training as an alternative to PPO — raised once, not adopted; PPO is
  what's producing every result in this project so far.
