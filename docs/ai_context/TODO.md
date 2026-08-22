# TODO / Roadmap

Updated 2026-08-22. Prioritized, not sequenced by date. "Critical" blocks understanding
current results; "High" is the next real work; "Medium" is valuable but not blocking;
"Research/Future" is speculative scope, not committed work.

**Everything that was Critical/Phase 1/Phase 2 in the previous version of this file is now
done** — multi-pass brake convergence, the thresholded brake-engagement penalty, the N-aware
safety margin, and 3D `r_spread` (plus the geometry-angle bug fix along the way) were all
implemented and validated (Phase2-combined, 5-seed `N=2/3/4` sweep, 3/3 `N=4` seeds clean at
0% collision). **The `LOST_TIMEOUT_SEC` sweep, active search, and the brake's relative-velocity
reformulation (all "High" as of the previous pass) are also now done** — see
`EXPERIMENT_LOG.md`/`DECISIONS.md`. The roadmap below reflects what's open *now*, not what was
open as of the last pass.

## Critical

- **Confirm the relative-velocity brake fix actually reduces `collision_rate` at training
  scale.** Adversarially verified (`test_brake_relative_velocity.py`), but the mechanism-level
  fix isn't the same claim as "this measurably helps a real training run." 3 seeds/3M steps
  running (`stage-marl-brake-relvel-s{1,2,3}`). **Next step**: once complete, download and
  analyze both eval-time and training-time rolling-window `collision_rate`, comparing against
  the `search-t8`/`search-t10`-seed3 baseline this was meant to fix (8-14% eval collision on
  well-converged checkpoints). See `KNOWN_ISSUES.md` item 13.
- **Verify the vertical-jiggling fix actually worked.** `CRUISE_ALT_MIN`/`CRUISE_ALT_COEF`/
  `Z_SMOOTHING_ALPHA` are implemented and smoke-tested, but the 8-25% vertical-velocity
  sign-flip rate that motivated them was measured *before* the fix — nobody has yet loaded a
  post-fix checkpoint and re-measured it the same way. See `KNOWN_ISSUES.md` item 14.
- **Fast-forward `phase2-combined` to `main`.** Verified as a clean fast-forward, no conflicts,
  but blocked by the permission classifier on a direct push from the agent — needs the user to
  run `git push origin phase2-combined:main` themselves (or confirm another approach). `main`
  is currently far behind everything described in this doc suite.

## High

- **Decide whether/how to push `target_lost_rate` further.** Active search + a long-enough
  timeout can reach ~2-4% `target_lost_rate` when a seed converges well, but training is
  seed-sensitive — 2 of 3 `t10` seeds got stuck worse, and resuming them to 5M steps only
  partially closed the gap (70-72%→41%, 92-96%→83%). Options not yet tried: more training steps
  still (from a fresh run, not a resume, to avoid the LR/entropy-schedule discontinuity a resume
  introduces), a different seed, or a design change to reduce seed-sensitivity itself. Not
  urgent relative to the collision-safety cost item above, which is the more concrete, more
  deployment-relevant open problem right now.
- **Update or replace `readme.txt`** — more stale than ever (still describes a removed
  per-drone-file save scheme, silent on everything from the brake onward). See
  `KNOWN_ISSUES.md` item 2.
- **Fix `envs/formation_env.py`'s remaining stale docstring paragraphs.** The "SCOPE" note
  claiming `TARGET_DIST` is "not a general-N formula... a 4-agent study" is still present and
  still contradicted by `_PACKING_RATIO`'s actual `{2,3,4}` support. The top-line module
  docstring still says "4-agent PettingZoo ParallelEnv" too. Low priority, cosmetic, flagged
  repeatedly across several doc passes without being fixed — worth just doing next time
  anyone is already editing that file for something else.

## Medium

- **UWB-realistic neighbor sensing**, replacing the still-exact-ground-truth neighbor
  observations. Deliberately deferred when vision-tracking was built (see `DECISIONS.md`) —
  real UWB gives clean range, not clean bearing/velocity, so this deserves its own careful
  design pass. Still the natural next realism step.
- **Sensing noise/occlusion on the target reading itself**, and a directional FOV cone (needs
  heading/orientation as a new state dimension first). See `KNOWN_ISSUES.md` item 11.
- **Reconcile reward-component logging scale change** (`63274b1`'s per-step vs. per-episode-
  sum switch) before comparing `r_track`/`r_safety`/etc. magnitudes between very old logs and
  current ones. Still unverified whether any analysis has actually needed to account for this.

## Research / Future (not committed, exploratory)

- A genuinely decentralized neighbor-discovery mechanism for the *graph maintenance* itself
  (separate from UWB-realistic sensing above) — `_repair_connectivity` still uses centralized
  ground-truth position knowledge. See `KNOWN_ISSUES.md` item 4.
- An actor architecture that's agent-count-invariant (e.g. attention over a variable number of
  neighbors), enabling actual weight transfer across the `NUM_AGENTS` curriculum instead of
  from-scratch comparative validation at each stage. See `DECISIONS.md`/`KNOWN_ISSUES.md`
  item 6. Would also change the CPU-vs-GPU calculus (`PHASE2_CHECKPOINT.md`'s "Open, not yet
  decided" section notes GPU has been consistently slower for every network size tried so far,
  including Phase 3's larger one — this is the kind of architecture change that could actually
  flip that).
- Real drone dynamics (inertia/acceleration limits, actuator delay) instead of the current
  direct-velocity-command kinematic model — still flagged as a "not okay for claiming
  realistic drone collision avoidance" simplification by an earlier supervisor review.
  Directly relevant to the `deployment/` workstream's own findings about zero-inertia mismatch
  — worth reading `deployment/docs/PHASE2_HANDOFF.md` before starting this.
- A learned nominal controller + safety-filter architecture more broadly (the closing-speed
  brake and ground clamp are two narrow instances of this pattern now, not one) — a
  supervisor review's broader "control barrier functions"/"action projection" recommendation
  is more adopted than before but still not a general framework.
- "Genetic/evolutionary" training as an alternative to PPO — raised once (see
  `PHASE2_CHECKPOINT.md`), not adopted; PPO is what's producing every result in this project
  so far.
