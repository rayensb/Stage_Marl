# TODO / Roadmap

Prioritized, not sequenced by date. "Critical" blocks understanding current results;
"High" is the next real work; "Medium" is valuable but not blocking; "Research/Future" is
speculative scope, not committed work.

## Critical

- **`NUM_AGENTS=4` collision persistence — current top-priority open problem.** A 3-seed,
  3M-step validation (2026-08-19) confirmed tracking generalizes cleanly to `N=4` (0-5%
  `target_lost_rate`, matching `N=3` quality, no changes needed) but collision avoidance does
  not: training-time rolling-window data shows 88/1465, 59/1465, and 269/1465 rollouts per seed
  with nonzero `collision_rate`, scattered across the full 3M-step run rather than converging,
  worst in seed3 (still nonzero in the last 10 logged rollouts). Eval-time numbers alone (0-2%)
  understate this. See `KNOWN_ISSUES.md` item 8 and `EXPERIMENT_LOG.md` for the full data.
  Two fixes are proposed below (Phase 1) — **not yet implemented**, pending confirmation.

### Phase 1 — directly targets the confirmed `N=4` collision mechanism (test together, before Phase 2)

- **Multi-pass brake convergence.** The brake's per-pair correction loop in `step()` applies
  sequentially and never re-checks that a later neighbor's correction didn't reintroduce a
  violation of an earlier one. Sweep to convergence instead (repeat until no neighbor needs
  further correction, or a fixed max iteration count) — a standard projection-onto-
  intersection-of-halfspaces technique (POCS), provably convergent. See `KNOWN_ISSUES.md` item
  8.
- **Direct brake-engagement reward penalty.** Add `r_brake = BRAKE_PENALTY_COEF *
  brake_reduction[agent]` (new reward component, negative coefficient) — `brake_reduction` is
  already computed every step but only logged, never fed back into the reward. Closes the gap
  where a policy can keep commanding more-than-safe closing speed without a penalty specific to
  that choice (today's only pressure is `r_safety`'s proximity-based urgent zone, an indirect
  signal that doesn't distinguish "close but not closing" from "close and still pushing
  closer"). Starting coefficient is a guess to tune, like every other weight in `config.py`.

### Phase 2 — formation-quality/generalization, not a bug fix; test separately after Phase 1 lands

- **N-aware safety margin.** `EDGE_TARGET` is currently fixed regardless of `N`. Scale it by
  how many simultaneous neighbors a drone actually has:
  `EDGE_TARGET = SAFE_DIST_EXIT + REACTION_DIST + max(0, K_NEIGHBORS - 2) * REACTION_DIST` —
  unchanged at `N=3` (the validated case), one extra `REACTION_DIST` of margin at `N=4`. Targets
  valence (simultaneous-neighbor count) directly rather than an arbitrary "make N=4 bigger"
  fudge.
- **3D (XYZ) `r_spread`.** Replace the horizontal-only bearing-sort with pairwise 3D angular
  separation between neighbor direction vectors, penalizing the minimum pairwise angle against
  the Tammes-ideal for that neighbor count (180° for 2, 120° for 3). See `KNOWN_ISSUES.md` item
  5 for why this specifically matters more at `N=4` — its exact-consistent formation geometry
  (verified, not contradictory — same item) is inherently non-planar, and nothing currently
  shapes the swarm toward the vertical structure it requires.
- Re-check the brake's multi-agent collision edge case again at whatever `NUM_AGENTS` is tested
  next, after Phase 1 lands — don't assume a fix confirmed at `N=4` automatically holds beyond
  it.

## High

- **Update or replace `readme.txt`** — more stale than ever after this session (see
  `KNOWN_ISSUES.md` item 2).
- **Fix `envs/formation_env.py`'s remaining stale docstring paragraph** (the old "4-agent
  study, not general-N" SCOPE note — contradicted by `config.py`'s actual generalized
  `_PACKING_RATIO` since before this session). A new, accurate docstring paragraph was added
  for vision-tracking without touching this pre-existing one.
- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-
  sum switch) before comparing `r_track`/`r_safety`/etc. magnitudes between logs from before
  and after that commit — still relevant for anyone comparing very old logs to current ones.

## Medium

- **UWB-realistic neighbor sensing**, replacing the still-exact-ground-truth neighbor
  observations (`rel_p`/`rel_v`/`d`). Deliberately deferred when vision-tracking was built
  (see `DECISIONS.md`) — real UWB gives clean range, not clean bearing/velocity, so this
  deserves its own careful design pass, not a quick patch. Natural next realism step now that
  target-sensing is done.
- **Make target motion non-constant-velocity** (occasional direction/speed changes
  mid-episode). Currently the 2-second dead-reckoning grace period in the vision-tracking
  system is mathematically exact, not a real approximation, precisely because the target
  never maneuvers — this change would make that part of the system actually get tested under
  real uncertainty. See `KNOWN_ISSUES.md` item 9.
- **Sensing noise/occlusion on the target reading itself**, and a directional FOV cone
  (needs heading/orientation added as a new state dimension first — a bigger prerequisite
  change than the noise/occlusion pieces). See `KNOWN_ISSUES.md` item 11.
- Consider whether `r_spread`'s horizontal-only limitation is actually causing any
  exploitable behavior, now that there's much more training data to check against.

## Research / Future (not committed, exploratory)

- A genuinely decentralized neighbor-discovery mechanism for the *graph maintenance* itself
  (separate from UWB-realistic sensing above) — `_repair_connectivity` still uses centralized
  ground-truth position knowledge to maintain the lock graph, which is a simulator-level
  shortcut, not just an observation-realism question.
- An actor architecture that's agent-count-invariant (e.g. attention over a variable number
  of neighbors instead of a fixed `EFFECTIVE_K`-sized observation), which would enable actual
  weight transfer across the `NUM_AGENTS` curriculum stages instead of the current
  from-scratch comparative validation at each stage (`DECISIONS.md`).
- Real drone dynamics (inertia/acceleration limits, actuator delay) instead of the current
  direct-velocity-command kinematic model — flagged as a "not okay for claiming realistic
  drone collision avoidance" simplification by an earlier supervisor review, still true.
- A learned nominal controller + safety-filter architecture more broadly (the closing-speed
  brake is a first, narrow instance of this pattern for one specific failure mode — a
  supervisor review's broader recommendation for "control barrier functions"/"action
  projection" as a general design philosophy is only partially adopted so far).
