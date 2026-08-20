# Architecture

Reference, not a copy: exact code lives in the files cited. Structural/conceptual
descriptions below were verified directly against commit `62a685d` on `phase3-resilience`
(2026-08-20) by reading the actual functions, not from memory of older code — trust the file
over any specific line number here if it's drifted since.

## Module map

```
config.py                    single source of truth for all constants
envs/formation_env.py        FormationEnv3D(ParallelEnv) — simulation + reward
training/networks.py         Actor (shared policy), CentralCritic (shared trunk, per-agent heads)
training/buffer.py           RolloutBuffer — per-agent storage + GAE
training/train.py            main training loop (CTDE-PPO)
training/checkpoint.py       save/load checkpoint + best-actor snapshot
training/logger.py           CSV metric logging (per-rollout rows)
training/plot_training.py    reads the CSV log, produces a diagnostic subplot grid
training/evaluate.py         deterministic evaluation of a trained/checkpointed actor
training/diagnose_horizon.py runs a frozen checkpoint well past its training horizon,
                              without stopping on first failure, to see how behavior
                              degrades past what it ever trained on (added 2026-08-20)
test_env.py                  minimal smoke test (random-action steps, no training)
test_geometry.py             verifies _IDEAL_NEIGHBOR_ANGLE against an analytical
                              regular-simplex construction (added 2026-08-20)
deployment/                  real PX4/Gazebo/ROS2 SITL inference pipeline for the
                              trained policy — a separate, independent workstream
                              sharing this worktree/branch history; see below
```

## `deployment/` exists now — this is not a pure-simulation-only repo anymore

An earlier version of this document claimed "no ROS2, Gazebo, or PX4 integration exists
anywhere in the repo." That claim is now **false** and has been for a while — `deployment/`
contains a real PX4 SITL + Gazebo + ROS2 inference pipeline (`inference_node.py`,
`launch_px4_instance.sh`, a pure-simulation demo, multi-instance real-neighbor operation) built
by a concurrent conversation sharing this same worktree and git history (commit `bebe232` and
others are interleaved directly into this branch's own commit log — not a separate unmerged
branch). Its own documentation lives in `deployment/docs/` (`PHASE2_HANDOFF.md`,
`TESTING_METHODOLOGY.md`) and is **out of scope for this `docs/ai_context/` suite** — this
suite documents the MARL research/training side (`config.py`, `envs/`, `training/`), not the
deployment pipeline. Treat `deployment/docs/` as the authoritative source for anything
PX4/Gazebo/ROS2-specific.

The two workstreams are coupled in a few concrete, worth-knowing ways:
- `deployment/inference_node.py` depends on `_apply_brake`'s exact `(corrected_vel,
  brake_reduction)` return signature — the closing-speed brake is deliberately reimplemented
  in/called from the deployment pipeline, not just the training env, since it's not part of
  the learned network and a deployed swarm needs it too (see `DECISIONS.md`).
  `_apply_brake`'s 2026-08-20 instrumentation additions were written to preserve that
  signature specifically for this reason.
- `training/diagnose_horizon.py`'s finding (tracking error nearly quadrupling once a frozen
  checkpoint runs well past its 10s training horizon, peaking in a 20-30s window) was
  identified as landing *directly inside* the 15-40s window where the real PX4/Gazebo
  deployment actually crashed — real evidence connecting a training-side gap to a
  deployment-side failure, not just a plausible story. This is what motivated Phase 3
  (below). See `deployment/docs/PHASE2_HANDOFF.md` for the deployment-side account.
- A coordination doc at the repo root, `PHASE2_CHECKPOINT.md`, is shared by both threads —
  check it (and its "do not modify without checking with the user first" note) before
  touching `envs/formation_env.py`, `config.py`, or `training/*.py` if you're not sure whether
  another session has in-flight work on them.

## Environment: `FormationEnv3D` (`envs/formation_env.py`)

Implements PettingZoo's `ParallelEnv` — all agents act simultaneously each step.

- **Agents**: `NUM_AGENTS` drones (`drone1`...`droneN`), each with 3D position/velocity
  state, driven by a simple kinematic integrator (`DT=0.05`s per step). Commanded velocity is
  scaled **per-axis** since Phase 3 (2026-08-20): horizontal (X/Y) capped at
  `MAX_ACTION_SPEED`, vertical (Z) separately capped at `MAX_ACTION_SPEED_Z = 0.6 *
  MAX_ACTION_SPEED` — real drones have different climb-rate authority than lateral speed, and
  the sim used to cap all 3 axes identically. `REACTION_DIST`/`SAFE_DIST_ENTER`/
  `SAFE_DIST_EXIT` still derive from the larger `MAX_ACTION_SPEED`, which stays a valid (if
  now slightly conservative on Z) worst-case bound. See the closing-speed brake and ground
  clamp below, either of which can further reduce the *effective* velocity actually applied.
- **Observation** (`OBS_DIM = OBS_OWN_DIM + 7 * EFFECTIVE_K`, `OBS_OWN_DIM=18`, `_get_obs`):
  own-state block: own velocity (3), relative position/distance/velocity to the tracked target
  *estimate* (3+1+3=7, sourced from the swarm's vision-based track, not ground truth),
  `has_direct_contact` (1), track `confidence` (1), track `age` normalized by
  `LOST_TIMEOUT_STEPS` (1), `observer_count` normalized by `NUM_AGENTS` (1), relative direction
  to the centroid of currently in-contact teammates (3), `centroid_valid` (1) — plus 7 values
  per locked neighbor (`EFFECTIVE_K = min(K_NEIGHBORS, NUM_AGENTS - 1)`, `K_NEIGHBORS =
  NUM_AGENTS - 1` currently, so every agent is always locked/observed). At `NUM_AGENTS=4`,
  `OBS_DIM = 18 + 7*3 = 39`. Unchanged in shape since the vision-tracking redesign — Phase
  1/2/3 added new deterministic mechanisms and reward terms, not new observation fields.
- **Action**: `ACT_DIM=3` continuous (velocity command in 3D), squashed through `tanh` in the
  policy, then scaled per-axis as described above.
- **Closing-speed brake** (`_apply_brake`, `step()`): a deterministic, non-learned
  action-space safety layer, computed *before* positions update each step, entirely separate
  from the reward. For any pair of agents already inside `SAFE_DIST_ENTER`, caps only the
  *closing* component of each agent's commanded velocity, ramping linearly to zero exactly at
  `COLLISION_DIST`. Lateral/evasive motion is untouched; a no-op outside `SAFE_DIST_ENTER`.
  **Multi-pass since 2026-08-19** (`ac98d67`'s predecessor commit): the per-neighbor
  correction sweeps repeatedly across all neighbors (up to `NUM_AGENTS` passes, POCS-style
  projection onto an intersection of halfspaces) until no further correction is needed,
  instead of one sequential pass — a single pass left a real, confirmed gap at `N=4` where
  correcting for one neighbor could reintroduce a violation of another's constraint. See
  `DECISIONS.md` for the full derivation, the "limit-convergent, not N-pass-exact" precision
  added after review, and the adversarial stress-test verification. Exposes three diagnostics
  via `self._last_brake_passes`/`_last_brake_violation`/`_last_brake_k_active` (passes used,
  worst residual violation, per-agent simultaneous-threat count for solo/multi engagement
  splitting), logged as `mean_brake_passes`/`max_brake_violation`/`mean_brake_solo`/
  `mean_brake_multi`. `brake_reduction` (speed actually removed) is also fed back into the
  reward as `r_brake` (see below), not just logged.
- **Ground clamp** (`_apply_ground_clamp`, Phase 3, 2026-08-20): the ground-plane counterpart
  to the brake, added because the sim had no ground plane/failure mode at all until this
  point. Single-pass, not multi-pass — the ground doesn't move and can't conflict with itself
  the way two agents' constraints can, so one pass is exact. Caps downward Z velocity as
  altitude approaches `GROUND_Z=0`, ramping to zero at the floor, using `MAX_ACTION_SPEED_Z`
  (not the horizontal cap). Applied in `step()` right after the brake. Exposes
  `ground_reduction` per agent, mirroring `brake_reduction`.
- **Z-velocity smoothing** (Phase 3, 2026-08-20): a deterministic low-pass filter — each
  step's commanded Z velocity is blended with the *previous actual* Z velocity
  (`Z_SMOOTHING_ALPHA=0.3`: 30% new, 70% carried over) — applied in `step()` **before** the
  brake/ground clamp, so those safety-critical corrections always retain final say over what
  is a comfort-only smoothing pass, never the other way around. Added after empirically
  confirming (not assuming) vertical jiggling on a trained checkpoint: 8-25% of steps per
  agent flipping vertical-velocity sign, one agent briefly at `z=0.98`.
- **Vision-based cooperative target tracking** (`_update_target_track()`, called once per
  step after positions update): each drone gets a direct reading of the target only when
  within `SENSOR_RANGE` (omnidirectional — no heading/FOV-cone state exists in this sim). Any
  drone with direct contact shares its reading with the whole swarm instantly. If nobody
  currently has contact, the swarm dead-reckons from the last known position/velocity for up
  to `LOST_TIMEOUT_STEPS` — `self._track_confidence` decays linearly 1.0→0.0 over that window.
  Beyond it, `self._target_lost=True` and the episode terminates. `self.pos_t` (ground truth)
  is never read by `_get_obs`/`_get_reward` — only by the diagnostic-only `_dist_to_target()`
  (`true_track_err` in `infos`, and `evaluate.py`'s `tracking_rmse`).
  **Target motion is dynamic since Phase 3** (2026-08-20, closes the old `KNOWN_ISSUES.md`
  item 9): `_target_dir`/`_target_speed` used to be sampled once in `reset()` and held for the
  whole episode, making the dead-reckoning grace period mathematically exact whenever used,
  not a real approximation. `step()` now calls `_resample_target_motion()` (same distribution
  `reset()` uses) every `TARGET_REDIRECT_INTERVAL_STEPS` (500 steps = 25s), so a mid-episode
  redirect while a drone is out of contact can genuinely mislead its dead-reckoned estimate —
  the grace period is now actually being tested against real estimate drift.
  **`LOST_TIMEOUT_SEC` is env-overridable since 2026-08-20** (was a flat `2.0`) — see
  `CURRENT_STATE.md`/`EXPERIMENT_LOG.md` for why (a fixed grace period tuned against
  `MAX_STEPS=200` broke badly once `MAX_STEPS` grew to 1800, and a sweep is in progress to
  find a working value at the new episode length).
- **Neighbor graph — mutual k-NN with connectivity repair** (`_compute_all_candidates`,
  `_relock_all`, `_repair_connectivity`): unchanged since the vision-tracking redesign — each
  agent's `k` nearest neighbors are computed from ground-truth positions, only *mutual* locks
  are kept, and `_repair_connectivity` (union-find) force-connects disconnected components.
  Still explicitly documented as using centralized simulator knowledge — a deliberate
  simplification, not a decentralized sensing model. See `DECISIONS.md`/`KNOWN_ISSUES.md`
  item 4.
- **`reset()`**: spawns agents at `uniform(TARGET_DIST, TARGET_DIST + 2*REACTION_DIST)` from
  the target, then runs a **joint fixed-point loop** (Phase 3, 2026-08-20, up to 20
  iterations) alternating `_resolve_overlaps()` (de-overlaps agents in unconstrained 3D) and a
  ground-safety Z-clamp (keeps spawn altitude safely above `GROUND_Z`) — replacing an earlier
  naive version that applied the Z-clamp once, before `_resolve_overlaps()`, which could
  collapse inter-agent vertical separation and cause an immediate step-0 collision. The joint
  version was verified across 500 seeds at `N=2/3/4` with zero violations of either constraint
  before shipping. `_resolve_overlaps()` is still never called from `step()` — it's a
  spawn-time-only mechanism, not an in-episode safety net (that's the brake's job).
- **`step()`**: applies Z-smoothing, then the brake, then the ground clamp (in that order —
  see above for why), advances physics, resamples target motion if due, updates the target
  track, recomputes neighbor locks if needed, computes `colliding_agents` and
  `ground_struck_agents` (both per-agent-involved, not global-to-everyone, for the same
  credit-assignment reason — see `DECISIONS.md`), returns `(obs, rewards, terms, truncs,
  infos)` per PettingZoo's Parallel API. `terms[a]=True` on **collision, `target_lost`, or
  `ground_strike`** (three real terminal failure modes, no bootstrap on any) —
  `infos[a]["collision"]`/`["target_lost"]`/`["ground_strike"]` distinguish which. Truncation
  is the separate `MAX_STEPS` time limit — **raised from a flat 200 (10s) to an
  env-overridable default of 1800 (90s) in Phase 3**, after `diagnose_horizon.py` showed
  tracking quality degrading substantially once flown longer than the old training horizon
  (see the `deployment/` section above).
- **`_get_reward()`** returns `(total, components_dict)`. **Eleven components** as of Phase 3
  (grew from 8): `track`, `spread`, `safety`, `cohesion`, `collision`, `velocity`, `joint`,
  `contact`, `brake` (2026-08-19), `ground` (2026-08-20), `altitude` (2026-08-20). Exact
  weights/thresholds live in `config.py` — check it directly rather than duplicating numbers
  here.
  - **`r_spread`** (corrected 2026-08-20): true pairwise **3D** angular separation between
    locked-neighbor direction vectors (was horizontal/XY-bearing-only), penalizing how far the
    minimum pairwise angle sits from `_IDEAL_NEIGHBOR_ANGLE` (60°, analytically proven for any
    neighbor count under full connectivity — see `DECISIONS.md` for the geometry-conflation
    bug this replaced and `test_geometry.py` for the verification). The module's former "known
    2D limitation" note is resolved — this is no longer horizontal-only.
  - **`r_brake`** (2026-08-19): `BRAKE_PENALTY_COEF * max(0, brake_reduction -
    BRAKE_PENALTY_THRESHOLD)` — only the excess above a derived threshold is penalized, not
    brake engagement from zero (a v1 linear-from-zero version measurably fixed collision but
    pushed the policy to avoid the brake's trigger zone altogether, widening the formation and
    hurting tracking — see `DECISIONS.md`).
  - **`r_ground`**: graduated penalty as altitude approaches `GROUND_Z` (ramp shape mirrors
    `r_safety`'s urgent zone) plus `GROUND_STRIKE_PENALTY` (-300, matching collision's
    magnitude) on the step an agent actually strikes the ground.
  - **`r_altitude`**: a *softer*, separate concept from `r_ground` — a graduated penalty for
    loitering below `CRUISE_ALT_MIN=1.5`, a comfort preference rather than a crash-avoidance
    boundary. No bonus for flying high.
  - **`r_contact`**: unchanged since vision-tracking — ramps toward `CONTACT_URGENT_COEF` as
    `steps_since_contact` approaches `LOST_TIMEOUT_STEPS`, plus `TARGET_LOST_PENALTY` on the
    terminating step. Identical for every agent (swarm-wide signal).

## Config (`config.py`, ~380 lines) — every constant is derived, not guessed

Every constant has an inline comment recording *why* it has that value. This is a deliberate
project convention — **do not remove or shorten these comments** when editing `config.py`.

Notable derivations beyond what's already described above (see the file for full reasoning):

- `SAFE_DIST_ENTER`/`SAFE_DIST_EXIT`/`GROUND_SAFE_ENTER`/`GROUND_SAFE_EXIT` all derive from the
  same `REACTION_DIST` physical quantity — the ground zone reuses the inter-agent derivation
  language against a fixed floor instead of another agent's position.
- `EDGE_TARGET` is now **N-aware** (Phase2-combined, 2026-08-19): `SAFE_DIST_EXIT +
  REACTION_DIST + max(0, K_NEIGHBORS - 2) * REACTION_DIST` — adds one extra `REACTION_DIST` of
  margin per simultaneous neighbor beyond the validated `N=3` case (unchanged at `N=3`, +1.20
  at `N=4`). `TARGET_DIST` is still solved backward from `EDGE_TARGET` via `_PACKING_RATIO`
  (the Tammes/regular-simplex packing ratio), so this change flows through automatically.
- `MIN_DIAMETER`/`DIAMETER_FLOOR_WEIGHT`: still `0.0` (disabled, not deleted) — unchanged
  since it was found to fight the sensor-range constraint once vision-tracking existed.
- `MAX_ACTION_SPEED_Z = 0.6 * MAX_ACTION_SPEED` (Phase 3): a starting guess at a real-ish
  climb-vs-lateral speed ratio, not measured — flagged as the first thing to retune if it
  doesn't move the needle.
- `LOST_TIMEOUT_SEC`: env-overridable since 2026-08-20 (default still `2.0`, but a sweep of
  6/10/18s is in progress — see `CURRENT_STATE.md`).
- `CRUISE_ALT_MIN=1.5`, `CRUISE_ALT_COEF=-10.0`, `Z_SMOOTHING_ALPHA=0.3`: the altitude-comfort
  and vertical-smoothing constants (Phase 3, 2026-08-20) — see `DECISIONS.md`.
- `TARGET_REDIRECT_INTERVAL_STEPS=500`: dynamic target motion cadence (Phase 3).

## Networks (`training/networks.py`)

- **`Actor`**: 2-layer MLP, hidden width **128** (raised from 64 in Phase 3, 2026-08-20, as
  part of the sustained-flight bundle — a larger network was hypothesized as needed to handle
  the longer episodes/richer dynamics, not measured in isolation). Outputs mean + log_std
  (clamped to `[LOG_STD_MIN=-2.0, LOG_STD_MAX=0.5]`). Policy is a **tanh-squashed Gaussian** —
  `get_action()`/`evaluate()` both apply the correct Jacobian log-probability correction.
  `entropy` in `evaluate()` is `-logp` (Monte Carlo estimate under the true squashed
  distribution), deliberately not `dist.entropy()` on the pre-squash Gaussian.
- **`CentralCritic`**: shared `trunk` (hidden width **256**, raised from 128 in Phase 3,
  same bundle) taking the joint observation (`OBS_DIM * NUM_AGENTS`), then `heads =
  nn.Linear(hidden, num_agents)` — one scalar value output per agent. Kept at 2x the Actor's
  hidden width, same ratio as before Phase 3.
- **`Actor.get_log_std()`**: returns the clamped `log_std` directly — the diagnostic that
  originally distinguished exploration-noise collapse from action-magnitude saturation and led
  to the closing-speed brake. Unchanged.

## Actor sharing — the key architectural choice in `train.py`

Unchanged: **one** `Actor` instance, not one per agent. All `NUM_AGENTS` agents' rollout data
pools into a single training batch for that one actor. The critic keeps its per-agent head
structure, indexed via `head_idx` per transition. Rollout-time action sampling is batched (one
`actor.get_action()` call across all currently-alive agents per step).

## Training loop (`training/train.py`) — CTDE-PPO

High-level loop, one iteration = one rollout + one training phase:

1. Collect `ROLLOUT_LEN` env steps (batched actor inference), storing per-agent `(obs,
   joint_obs, action, old_logp, reward, done, value)` into `RolloutBuffer`. Truncation
   (time-limit, not collision/target_lost/ground_strike) bootstraps the critic's next-obs
   value estimate into the stored reward — matches CleanRL's truncation handling.
   **`ROLLOUT_LEN` is now derived, not independently hardcoded** (Phase 3, 2026-08-20):
   `ROLLOUT_LEN = MAX_STEPS * ROLLOUT_EPISODES` (`ROLLOUT_EPISODES=8`), so raising `MAX_STEPS`
   automatically keeps the same "episodes per rollout" ratio (14,400 at the new
   `MAX_STEPS=1800` default) instead of silently starving PPO's per-update sample diversity
   the way a fixed `ROLLOUT_LEN=2048` would have (only ~1 episode per rollout at the new
   horizon).
2. Compute per-agent GAE + normalized advantages, pool across agents into one batch for the
   actor.
3. Run up to `EPOCHS=10` passes over the pooled batch in `BATCH_SIZE=256` minibatches: PPO
   clipped objective + entropy bonus for the actor, MSE for the critic, gradient clipping on
   both, Adam with linear LR annealing to 0 over `TOTAL_STEPS`.
4. **Two independent early-stop / trust-region checks**: `TARGET_KL=0.02`, per-minibatch and
   per-epoch.
5. **Entropy-coefficient schedule**: linear anneal, overridden by a flat rate during
   plateau-triggered "entropy recovery."
6. Log one row per rollout, save a regular checkpoint every rollout, save a "best actor"
   snapshot whenever `best_score` improves.

### Entropy recovery, two decoupled "best" trackers, additional diagnostics

Unchanged in mechanism from the previous documentation pass — see `DECISIONS.md` for the
full reasoning chain (a plain `entropy < 0` trigger missed a real regression; a
`ReduceLROnPlateau`-style `collision_rate`-based trigger replaced it;
`RECOVERY_MAX_TRIGGERS` was tuned once, `5 → 20`, then the underlying hypothesis was falsified
by measurement and reverted).

`best_score` is still lexicographic (`collision_rate` rounded to 2dp, then `target_lost_rate`,
then `-track`) — **now with `ground_strike_rate` folded in as a 4th tier** (Phase 3,
2026-08-20), same reasoning each time a new real failure mode was added: a checkpoint that's
safe on the older criteria but frequently strikes the ground isn't actually the best one.

Additional per-rollout diagnostics logged: `log_std_mean`, `mean_action_abs`,
`mean_brake_reduction`, `target_lost_rate`, and since Phase 1/3: `mean_brake_passes`,
`max_brake_violation`, `mean_brake_solo`, `mean_brake_multi`, `ground_strike_rate`.

## Buffer (`training/buffer.py`)

Unchanged: `RolloutBuffer` stores obs/actions/logp/rewards/values **per agent** (dicts keyed
by agent name). `compute_gae` runs per-agent. `get_tensors(agent, last_value)` returns that
agent's tensors with advantage normalization already applied, per-agent, before pooling.

## Checkpointing (`training/checkpoint.py`)

Unchanged: single-file scheme. `save_checkpoint`/`load_checkpoint` store one `"actor"` key.
`save_best_actor(actor, run_id="")` writes actor weights only to
`models/actor_best[_run_id].pt`.

## Logging (`training/logger.py`) and plotting (`training/plot_training.py`)

One CSV row per rollout. `FIELDS` (see `training/logger.py`) now includes
`ground_strike_rate`, `r_ground`, `r_altitude`, `mean_brake_passes`, `max_brake_violation`,
`mean_brake_solo`, `mean_brake_multi` alongside the existing throughput/PPO-diagnostic/
entropy-recovery/best-checkpoint/geometry/reward-component fields. `RUN_ID` suffixes the log
filename so parallel multi-seed runs don't collide. `plot_training.py` mirrors `train.py`'s
`RUN_ID` resolution and renders a diagnostic grid covering all of the above — check the script
directly for the current exact panel layout rather than trusting a specific grid size here, it
has grown incrementally each phase and wasn't re-counted for this pass.

## Evaluation (`training/evaluate.py`)

Deterministic (no sampling) rollout of a saved actor. `load_actors(model_dir, run_id, device,
best=False)` loads the single shared-actor file, `{agent_name: actor}` with the same object
for every key. Per-episode metrics written to CSV: `collided, target_lost` and now
`ground_strike` (Phase 3), read explicitly from `infos`, not inferred from `terms` (all three
set `terms[a]=True`). `episode_len, min_dist, tracking_rmse, avg_spacing_std, avg_diameter,
avg_speed, avg_confidence`. `tracking_rmse` remains deliberately ground-truth-based. CLI:
`--episodes, --model-dir, --run-id, --seed (default 0), --device, --save-trajectory, --best`.

## Sustained-horizon diagnostic (`training/diagnose_horizon.py`, added 2026-08-20)

Loads a frozen checkpoint and runs it for a much longer duration than its training horizon
(e.g. 60s against a 10s `MAX_STEPS=200` training horizon, at the time it was built), **without
terminating on the first failure** — the point is specifically to observe *how* behavior
degrades past what the policy ever trained on, not just whether it eventually fails. This is
what surfaced the tracking-degradation evidence that motivated Phase 3 — see the `deployment/`
section above and `EXPERIMENT_LOG.md`.

## Data/control flow summary

```
config.py  ──constants──▶  envs/formation_env.py  ◀──actions──  training/train.py
                                     │                                  │
                                     └──(obs, reward, done)────────────▶│
                                                                        │
                            training/networks.py (Actor, CentralCritic)│
                            training/buffer.py (per-agent GAE storage) │
                                                                        ▼
                            training/logger.py ──▶ logs/training_log[_<run_id>].csv
                            training/checkpoint.py ──▶ checkpoints/, models/
                                                                        │
                                                                        ▼
                                              training/plot_training.py (diagnostics)
                                              training/evaluate.py (deterministic eval → CSV)
                                              training/diagnose_horizon.py (sustained-flight check)
                                                                        │
                                                                        ▼
                                              deployment/inference_node.py (real PX4/Gazebo/ROS2,
                                              separate workstream — see deployment/docs/)
```

## What is NOT in this codebase (as of this writing)

- No true decentralized sensing/communication model for the *neighbor* graph (still centrally
  computed by the simulator — see `_repair_connectivity` above and `DECISIONS.md`). This
  remains inconsistent with how the *target* is sensed (vision-based, range-limited,
  shared-on-contact) — see `KNOWN_ISSUES.md` item 10, deliberately deferred.
- No sensing noise, occlusion, or directional FOV cone on the target reading itself (item 11).
- No literal cross-agent-count weight transfer (`OBS_DIM` differs between `NUM_AGENTS=2` and
  `3`/`4` — see `DECISIONS.md`).
- The brake's no-crossing guarantee uses each agent's *own* velocity toward another, not
  relative velocity — a known gap once independently-controlled real vehicles are involved
  (not yet reformulated — see `KNOWN_ISSUES.md` and `deployment/docs/PHASE2_HANDOFF.md`).
- **What used to be true and no longer is**: target motion is no longer constant-velocity
  (Phase 3 added periodic redirects); `r_spread` is no longer horizontal-only (now true 3D);
  the brake is no longer single-pass; there is no longer "no ROS2/Gazebo/PX4 integration" —
  see the `deployment/` section at the top of this document.
