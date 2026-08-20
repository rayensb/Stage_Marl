# TODO / Roadmap

Updated 2026-08-20. Prioritized, not sequenced by date. "Critical" blocks understanding
current results; "High" is the next real work; "Medium" is valuable but not blocking;
"Research/Future" is speculative scope, not committed work.

**Everything that was Critical/Phase 1/Phase 2 in the previous version of this file is now
done** — multi-pass brake convergence, the thresholded brake-engagement penalty, the N-aware
safety margin, and 3D `r_spread` (plus the geometry-angle bug fix along the way) were all
implemented and validated (Phase2-combined, 5-seed `N=2/3/4` sweep, 3/3 `N=4` seeds clean at
0% collision). See `EXPERIMENT_LOG.md`/`DECISIONS.md`. The roadmap below reflects what's open
*now*, not what was open as of the last pass.

## Critical

- **Confirm the `LOST_TIMEOUT_SEC` sweep and land a working value.** Phase 3's longer episodes
  (`MAX_STEPS` 200→1800) broke `target_lost_rate` almost completely (89-100%, flat from step 1)
  because the dead-reckoning grace period was never rescaled — see `KNOWN_ISSUES.md` item 12.
  A 5-kernel sweep (6s x2, 10s x2, 18s x1) is running; 4 of 5 are confirmed `RUNNING`, the 5th
  is blocked by a Kaggle platform issue (see `KNOWN_ISSUES.md` item 15 — likely to resolve on
  its own, or needs the user to check Kaggle's web UI directly). **Next step**: once all 5
  complete, download and analyze results (`collision_rate`, `target_lost_rate`,
  `tracking_rmse`, plus the training-time rolling-window picture, not just eval-time numbers —
  eval-time understated the real `N=4` collision problem once before) and pick a value to
  adopt as the new default.
- **Verify the vertical-jiggling fix actually worked.** `CRUISE_ALT_MIN`/`CRUISE_ALT_COEF`/
  `Z_SMOOTHING_ALPHA` are implemented and smoke-tested, but the 8-25% vertical-velocity
  sign-flip rate that motivated them was measured *before* the fix — nobody has yet loaded a
  post-fix checkpoint and re-measured it the same way. Do this once the current sweep produces
  checkpoints. See `KNOWN_ISSUES.md` item 14.
- **Fast-forward `phase2-combined` to `main`.** Verified as a clean fast-forward, no conflicts,
  but blocked by the permission classifier on a direct push from the agent — needs the user to
  run `git push origin phase2-combined:main` themselves (or confirm another approach). `main`
  is currently far behind everything described in this doc suite.

## High

- **The "active search when lost" behavior — explicitly requested, not yet started.** When the
  whole swarm loses target contact (not just an individual drone), have drones fan out in
  different directions to search, and communicate the target's location swarm-wide the moment
  any one of them reacquires it. The current system already does two related things well
  (dead-reckoning to last-known position/velocity, and instant swarm-wide sharing on contact —
  both were already implemented before this was requested, see `ARCHITECTURE.md`'s
  `_update_target_track` description) — this is the genuinely new piece: coordinated search
  *after* the dead-reckoning estimate is no longer trusted, instead of just continuing to
  drift toward a stale last-known position. Deliberately **not** bundled into the Phase 3
  changes or the `LOST_TIMEOUT_SEC` sweep, to avoid further compounding attribution difficulty
  while that sweep is still unresolved — treat as its own follow-up once tracking is stable
  again.
- **Reformulate the closing-speed brake with relative velocity, not each agent's own
  velocity.** `v_closing` is currently `dot(v_a, dir_to_b)`; the brake's stress-test
  verification relies on every agent running the identical symmetric formula, an assumption
  that doesn't hold once a real or independently-controlled vehicle is involved. This is no
  longer a purely theoretical footnote — `deployment/inference_node.py` calls `_apply_brake`
  directly against real PX4/Gazebo telemetry. See `KNOWN_ISSUES.md` item 13 and
  `deployment/docs/PHASE2_HANDOFF.md`.
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
