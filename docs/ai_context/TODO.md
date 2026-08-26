# TODO / Roadmap

Updated 2026-08-26. Prioritized, not sequenced by date. "Critical" blocks understanding current
results or represents the project's top open problem; "High" is the next real work; "Medium" is
valuable but not blocking; "Research/Future" is speculative scope, not committed work.

**Since the last pass**: `main` was fast-forwarded (no longer a pending item). The
relative-velocity brake fix was tested at full scale and **reverted** (net-harmful to
convergence, cause unknown) — no longer "confirm it works," now a closed, reverted experiment. A
network-capacity sweep resolved `N=4`'s hidden-width question but surfaced a new one:
`NUM_AGENTS=3` doesn't learn at all under Phase 3's mechanisms, at any tested size. A deep
investigation (critic-collapse diagnostics, then actor search-action dynamics) **localized
`target_lost`'s root cause to the learned actor's own search execution** — this is now the
project's single most important open item, replacing the older, less-specific "push
`target_lost_rate` further" framing.

## Critical

- **Design and test a fix for the actor's own search-execution problem.** Decisively localized
  (not just suspected): branching a scripted or adaptive-reassignment controller from PPO's own
  real failure states reacquires 100% of the time where PPO's real trajectory failed 100% of the
  time, holding environment/geometry/timeout/starting-state exactly fixed. No fix has been
  designed or attempted yet. Candidate directions, both untested: (a) a training signal that
  specifically rewards heading correction/reassignment once a chosen search direction stops
  making progress, since the productive-agent-count data shows most failures have *zero* agents
  making net progress, not one unlucky agent running out of time; (b) a credit-assignment change
  that makes a productive search action more clearly attributable across the many steps between
  committing to a heading and actually reacquiring, since standard per-step PPO credit assignment
  may be too diffuse over a ~100+-step search. This is an open question for the user — discuss
  before implementing either in `envs/formation_env.py` or `training/train.py`. See
  `EXPERIMENT_LOG.md`'s decisive counterfactual entry, `KNOWN_ISSUES.md` item 12.
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
