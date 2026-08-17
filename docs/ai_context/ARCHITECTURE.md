# Architecture

Reference, not a copy: exact code lives in the files cited. Line numbers/counts below were
accurate as of commit `b14fe48` and have **not** been re-verified against the current
`b59c139` (post closing-speed-brake, post vision-tracking) — the files have grown
substantially this session. Trust the file over any specific line number here; the
structural/conceptual descriptions were updated for `b59c139`, the numbers weren't all
re-checked.

## Module map

```
config.py                    single source of truth for all constants
envs/formation_env.py        FormationEnv3D(ParallelEnv) — simulation + reward
training/networks.py         Actor (shared policy), CentralCritic (shared trunk, per-agent heads)
training/buffer.py           RolloutBuffer — per-agent storage + GAE
training/train.py            main training loop (CTDE-PPO)
training/checkpoint.py       save/load checkpoint + best-actor snapshot
training/logger.py           CSV metric logging (per-rollout rows)
training/plot_training.py    reads the CSV log, produces a 7x2 diagnostic subplot grid
training/evaluate.py         deterministic evaluation of a trained/checkpointed actor
test_env.py                  minimal smoke test (200 random-action steps, no training)
```

No ROS2, Gazebo, or PX4 integration exists anywhere in the repo (confirmed by directory
inspection — there is no `ros2_ws`, no `.sdf`/`.world` files, no MAVSDK/MAVROS imports).
This is a pure Python/PyTorch simulation, not a robotics-middleware project, as of this
writing.

## Environment: `FormationEnv3D` (`envs/formation_env.py`)

Implements PettingZoo's `ParallelEnv` — all agents act simultaneously each step.

- **Agents**: `NUM_AGENTS` drones (`drone1`...`droneN`), each with 3D position/velocity
  state, driven by a simple kinematic integrator (`DT=0.05`s per step, `MAX_ACTION_SPEED`
  caps commanded speed) — but see the closing-speed brake below, which can reduce the
  *effective* commanded velocity actually applied.
- **Observation** (`OBS_DIM = OBS_OWN_DIM + 7 * EFFECTIVE_K`, `OBS_OWN_DIM=18`, `_get_obs`):
  own-state block grew from 10 to 18 values with vision-based tracking (2026-08-17, see
  below) — own velocity (3), relative position/distance/velocity to the tracked target
  *estimate* (3+1+3=7, same slots as before but now sourced from the swarm's track rather
  than ground truth), `has_direct_contact` (1), track `confidence` (1), track `age`
  normalized by `LOST_TIMEOUT_STEPS` (1), `observer_count` normalized by `NUM_AGENTS` (1),
  relative direction to the centroid of currently in-contact teammates (3), `centroid_valid`
  (1) — plus 7 values per locked neighbor (`EFFECTIVE_K = min(K_NEIGHBORS, NUM_AGENTS - 1)`,
  `K_NEIGHBORS = NUM_AGENTS - 1` currently, so every agent is always locked/observed — see
  `DECISIONS.md`). At `NUM_AGENTS=4`, `OBS_DIM = 18 + 7*3 = 39`.
- **Action**: `ACT_DIM=3` continuous (velocity command in 3D), squashed through `tanh` in
  the policy (see Networks below).
- **Closing-speed brake** (`step()`, 2026-08-17): a deterministic, non-learned action-space
  safety layer, computed *before* positions update each step, entirely separate from the
  reward. For any pair of agents already inside `SAFE_DIST_ENTER`, caps only the *closing*
  component of each agent's commanded velocity (the part of the vector actually shrinking the
  gap), ramping linearly to zero exactly at `COLLISION_DIST`. Lateral/evasive motion is
  untouched; a no-op outside `SAFE_DIST_ENTER`. `brake_reduction` (how much speed was actually
  removed) is exposed via `infos` and logged (`mean_brake_reduction`) so it's verifiable the
  mechanism is actually engaging rather than sitting dormant. See `DECISIONS.md` for why this
  replaced six failed reward-shape/schedule attempts, and `KNOWN_ISSUES.md` item 8 for the one
  known gap in its no-crossing guarantee (proven for two agents at a time, not simultaneous
  multi-threat braking).
- **Vision-based cooperative target tracking** (`_update_target_track()`, called once per
  step after positions update, 2026-08-17): replaced ground-truth target telemetry. Each
  drone gets a direct reading of the target only when within `SENSOR_RANGE` (omnidirectional
  — no heading/FOV-cone state exists in this sim). Any drone with direct contact shares its
  reading with the whole swarm instantly (averaged if multiple have contact simultaneously,
  currently a no-op since readings are noiseless, but structurally correct for when per-drone
  sensing noise is added). If nobody currently has contact, the swarm dead-reckons from the
  last known position/velocity (`self._last_known_pos`/`self._last_known_vel`) for up to
  `LOST_TIMEOUT_STEPS` (2s) — `self._track_confidence` decays linearly 1.0→0.0 over that
  window. Beyond it, `self._target_lost=True` and the episode terminates (a second real
  failure mode alongside collision — see `_get_reward`/termination below). `self.pos_t`
  (ground truth) is deliberately never read by `_get_obs`/`_get_reward` after this change —
  only by the diagnostic-only `_dist_to_target()` (exposed as `true_track_err` in `infos`,
  and used by `evaluate.py`'s `tracking_rmse`, which is intentionally ground-truth-based).
  See `DECISIONS.md` for the full design reasoning (why omnidirectional not a FOV cone, why
  `SENSOR_RANGE`'s formula, why UWB-realistic neighbor sensing was deliberately deferred).
- **Neighbor graph — mutual k-NN with connectivity repair** (`_compute_all_candidates`,
  `_relock_all`, `_repair_connectivity`): each agent's `k` nearest neighbors are computed
  from ground-truth positions, then only *mutual* locks are kept (both sides pick each
  other) as an edge, forming a lock graph. `_repair_connectivity` (union-find based) then
  force-connects any disconnected components by linking their closest cross-component pair,
  evicting the locking agent's farthest existing lock if it's already at its `k` cap.
  **Explicitly documented in the module docstring as using centralized simulator knowledge
  of all agent positions** — this is a deliberate simplification, not a decentralized
  sensing model; see `DECISIONS.md`. Re-locking happens only inside `step()`'s neighbor
  update, not continuously — locks persist across steps and are only recomputed on the
  normal per-step cadence (this is the "decentralized execution, not decentralized topology
  maintenance" framing in the module docstring).
- **`_resolve_overlaps()`** is called **only from `reset()`**, never from `step()` (confirmed
  via grep) — it exists to de-overlap the random initial spawn, not as an in-episode
  safety mechanism. Spawn radius: `uniform(TARGET_DIST, TARGET_DIST + 2*REACTION_DIST)` from
  the target.
- **`step()`**: applies the closing-speed brake, advances physics, recomputes neighbor locks,
  updates the target track, computes `colliding_agents` (the set of agents actually involved
  in a collision this step — used for **per-agent-involved** collision-penalty attribution
  rather than penalizing every agent globally whenever any collision occurs; see
  `DECISIONS.md`), returns `(obs, rewards, terms, truncs, infos)` per PettingZoo's Parallel
  API. `terms[a]=True` now signals **either** a collision **or** `target_lost` (both are real
  terminal states, no bootstrap on either) — `infos[a]["collision"]` and
  `infos[a]["target_lost"]` distinguish which, since `train.py`/`evaluate.py` need to log them
  as separate metrics rather than conflating "any termination" with "collision" (a bug found
  and fixed this session — see `KNOWN_ISSUES.md` history in `EXPERIMENT_LOG.md`). Truncation
  is still the separate `MAX_STEPS=200` time limit.
- **`_get_reward()`** returns `(total, components_dict)`. Components (all logged
  individually, see Logging below): `track`, `spread`, `safety`, `cohesion`, `collision`,
  `velocity`, `joint`, `contact` (new, 2026-08-17 — see below). Exact weights/thresholds live
  in `config.py` — don't duplicate the numeric values here, they change; check `config.py`
  directly.
- **`r_contact`** (new): ramps from 0 (swarm currently has direct target contact) toward
  `CONTACT_URGENT_COEF` as `steps_since_contact` approaches `LOST_TIMEOUT_STEPS`, plus
  `TARGET_LOST_PENALTY` on the terminating step — the same `r_safety`-ramp +
  `r_collision`-terminal two-tier shape already used elsewhere in this file. Identical for
  every agent (a swarm-wide signal, not per-agent blame), so one agent regaining contact
  helps everyone's reward — a genuinely cooperative incentive, not just a per-agent one.
- **Known 2D limitation**: `r_spread`'s formation-spread computation is horizontal-only
  (documented in the module docstring) — vertical (z-axis) spread is not part of that
  reward term as of this writing.

## Config (`config.py`, 135 lines) — every constant is derived, not guessed

Every constant has an inline comment recording *why* it has that value (prior failed
values, measured results, or a geometric/physical derivation). This is a deliberate project
convention — **do not remove or shorten these comments** when editing `config.py`; they are
the record `DECISIONS.md` refers back to.

Notable derivations (see the file for the full reasoning, not just the numbers):

- `SAFE_DIST_ENTER`/`SAFE_DIST_EXIT` are derived from a reaction-time/closing-speed physical
  model (`REACTION_DIST = 2 * MAX_ACTION_SPEED * (N_REACT * DT)`), not picked by hand.
- `TARGET_DIST` is derived per `NUM_AGENTS` from the Tammes problem (optimal packing of
  points on a sphere) via `_PACKING_RATIO` — a dict keyed by supported agent counts
  (`{2, 3, 4}` currently; `config.py` raises `ValueError` for any other `NUM_AGENTS`, so this
  is generic in mechanism but only pre-derived for 3 specific curriculum stages).
- `COHESION_LIMIT` and other margins are expressed as ratios of the physically-derived
  distances above, not independent magic numbers.
- `MIN_DIAMETER`/`DIAMETER_FLOOR_WEIGHT` (2026-08-14): a floor on swarm diameter, added to
  push back against the swarm converging too tight. **`DIAMETER_FLOOR_WEIGHT` is currently
  `0.0` (disabled, not deleted)** as of 2026-08-17 — it was found to actively fight the
  vision-tracking sensor-range constraint once the closing-speed brake made it redundant for
  its original collision-safety purpose. See `DECISIONS.md`.
- `SENSOR_RANGE`/`LOST_TIMEOUT_SEC`/`LOST_TIMEOUT_STEPS`/`CONTACT_URGENT_COEF`/
  `TARGET_LOST_PENALTY` (2026-08-17): the vision-tracking constants. `SENSOR_RANGE` reuses the
  existing spawn-radius formula (`TARGET_DIST + 2*REACTION_DIST`) rather than a new number.
  `LOST_TIMEOUT_SEC=2.0` is currently *exact* dead-reckoning, not an approximation, since the
  target moves at constant velocity for the whole episode — see `KNOWN_ISSUES.md` item 9.

## Networks (`training/networks.py`)

- **`Actor`**: 2-layer MLP, hidden width 64. Outputs mean + log_std (clamped to
  `[LOG_STD_MIN=-2.0, LOG_STD_MAX=0.5]` — this clamp was added specifically to fix an
  entropy-runaway bug, see `EXPERIMENT_LOG.md`). Policy is a **tanh-squashed Gaussian** —
  `get_action()` (stochastic sampling, used during rollout collection) and `evaluate()`
  (used during PPO updates: returns `logp` and `entropy` for a given action) both apply the
  correct Jacobian log-probability correction for the tanh squashing. `entropy` in
  `evaluate()` is computed as `-logp` (a Monte Carlo estimate under the true squashed
  distribution) — **deliberately not** `dist.entropy()` on the pre-squash Gaussian, which
  was a previously-identified and fixed bug (ignores the squashing, systematically
  overstates entropy).
- **`CentralCritic`**: shared `trunk` (hidden width 128) taking the **joint observation**
  (all agents' obs concatenated, `OBS_DIM * NUM_AGENTS`) as input, then `heads =
  nn.Linear(hidden, num_agents)` — one scalar value output per agent from the shared trunk.
  This is the "centralized" half of CTDE: the critic sees everyone's observations even
  though each actor only sees its own + neighbors'.
- **`Actor.get_log_std()`** (added 2026-08-17): returns the clamped `log_std` directly (it's
  a single obs-independent `nn.Parameter`, cheap to read anytime). Added specifically to
  distinguish two different causes of the `entropy` metric declining — `log_std` actually
  shrinking vs. the mean action saturating near tanh's boundary while `log_std` stays flat —
  which turned out to be the key diagnostic that led to the closing-speed brake. See
  `DECISIONS.md`/`EXPERIMENT_LOG.md`.

## Actor sharing — the key architectural choice in `train.py`

There is **one** `Actor` instance, not one per agent (`training/train.py:122`). All
`NUM_AGENTS` agents' rollout data is pooled into a single training batch for that one actor
(`train.py:282-296`, `torch.cat` across agents). The critic is *not* pooled the same way —
it's a single network but keeps its per-agent head structure and is indexed via `head_idx`
per transition (`train.py:310`). Rollout-time action sampling is also batched — one
`actor.get_action()` call across all currently-alive agents per step
(`train.py:166-170`), not a per-agent Python loop.

## Training loop (`training/train.py`, 409 lines) — CTDE-PPO

High-level loop, one iteration = one rollout + one training phase:

1. Collect `ROLLOUT_LEN=2048` env steps (batched actor inference, see above), storing
   per-agent `(obs, joint_obs, action, old_logp, reward, done, value)` into `RolloutBuffer`.
   Truncation (time-limit, not collision) bootstraps the critic's next-obs value estimate
   into the stored reward (`train.py:180-191`) — matches CleanRL's truncation handling.
2. Compute per-agent GAE + normalized advantages (`RolloutBuffer.get_tensors`, see Buffer
   below), pool across agents into one batch for the actor.
3. Run up to `EPOCHS=10` passes over the pooled batch in `BATCH_SIZE=256` minibatches:
   PPO clipped objective + entropy bonus for the actor, MSE for the critic, gradient
   clipping (`max_grad_norm=0.5`) on both, Adam optimizers with linear LR annealing to 0
   over `TOTAL_STEPS`.
4. **Two independent early-stop / trust-region checks**: `TARGET_KL=0.02` triggers an early
   break **per-minibatch** (not just per-epoch — this was a specific fix, see
   `EXPERIMENT_LOG.md`) and again as an epoch-level average check.
5. **Entropy-coefficient schedule**: linear anneal `ENT_COEF_START=0.01 → ENT_COEF_END=0.001`
   over `TOTAL_STEPS`, **overridden** by a flat `ENTROPY_RECOVERY_ENT_COEF=ENT_COEF_START`
   whenever plateau-triggered "entropy recovery" is active (see below).
6. Log one row per rollout to the CSV log (`training/logger.py`), save a regular checkpoint
   every rollout, and save a "best actor" snapshot whenever `best_score` improves.

### Entropy recovery (plateau-triggered exploration boost)

A `ReduceLROnPlateau`-style mechanism, but boosting entropy coefficient instead of cutting
LR (`train.py:239-258`). Tracks `collision_rate` over the same rolling 50-episode window
used for all other rolling metrics:
- New best (must clear `BEST_MIN_DELTA=0.01` to count, avoiding noise) → reset patience.
- No new best for `RECOVERY_PATIENCE=15` rollouts → trigger recovery: hold flat entropy
  coefficient for `RECOVERY_COOLDOWN=10` rollouts before re-judging.
- `RECOVERY_MAX_TRIGGERS=5` caps total interventions per run so a genuinely stuck run
  finishes on the normal schedule rather than oscillating forever.

This exists specifically because a plain "entropy < 0" trigger (tried previously) missed a
real regression: an `NUM_AGENTS=3` shared-actor run relapsed (`collision_rate` 0.02 → 0.25+)
without entropy ever going negative. See `EXPERIMENT_LOG.md` for that run.

### Two separate "best" trackers — deliberately decoupled

1. `best_collision_rate`/`since_best`/`cooldown_remaining` (`train.py:146-149`) — drives
   **only** entropy-recovery patience. Decided right after rollout collection, using
   `collision_rate` alone, before epoch training runs.
2. `best_score = (round(collision_rate, 2), round(target_lost_rate, 2), -comp_avgs['track'])`
   — drives the actual **best-checkpoint save** (`save_best_actor`). Computed *after* epoch
   training (needs `comp_avgs`), lexicographic: collision rate is primary (rounded to 2dp so
   meaningfully-equal safety levels tie), `target_lost_rate` is the second tier (added
   2026-08-17 alongside vision-tracking — a checkpoint that's collision-free but frequently
   loses the target isn't actually the best one, same reasoning as the original criterion),
   `track` component is the final tie-breaker. Originally 2-level; extended to 3 when
   `target_lost` became a second real failure mode. Exists because pure collision-rate-best
   was observed to prefer a policy that's "safe" only by spreading out excessively — worse
   tracking and swarm diameter than that same run's final-step model (see
   `EXPERIMENT_LOG.md`).

### Additional per-rollout diagnostics logged (2026-08-17)

`log_std_mean` (`Actor.get_log_std()`, see Networks above), `mean_action_abs` (mean |action|
across the rollout — tanh-saturation indicator), `mean_brake_reduction` (mean speed removed by
the closing-speed brake per agent per step — near-zero most of the time, confirms the brake
isn't constantly overriding the policy), and `target_lost_rate` (rolling-window rate,
analogous to `collision_rate` but tracked in a separate `recent_target_lost` deque — reading
this from `terms` alone would incorrectly conflate collision and target_lost episodes, since
both set `terms[a]=True`; both `collided` and `target_lost_now` are read explicitly from
`infos` instead).

## Buffer (`training/buffer.py`, 63 lines)

`RolloutBuffer` stores obs/actions/logp/rewards/values **per agent** (dicts keyed by agent
name, not a single shared array — `self.values = {a: np.zeros(...) for a in self.agents}`).
`compute_gae` runs per-agent. `get_tensors(agent, last_value)` returns that agent's
`(obs, joint_obs, action, old_logp, advantage, return)` tensors, with **advantage
normalization already applied** (`adv = (adv - adv.mean()) / (adv.std() + 1e-8)`,
`buffer.py:54`) per-agent, before the caller pools across agents. (This directly contradicts
a claim in a past external code review that advantage normalization was missing — verified
false by reading this file.)

## Checkpointing (`training/checkpoint.py`, 46 lines)

Single-file scheme (matches the shared-actor architecture): `save_checkpoint`/
`load_checkpoint` store one `"actor"` key (not a per-agent dict). `save_best_actor(actor,
run_id="")` writes actor weights only to `models/actor_best[_run_id].pt`. Regular training
checkpoints and best-actor snapshots are separate files.

## Logging (`training/logger.py`) and plotting (`training/plot_training.py`)

One CSV row per rollout (~2048 steps). Includes throughput (`steps_per_sec`,
`collect_steps_per_sec`), PPO diagnostics (`approx_kl`, `clip_frac`, `early_stop_kl`),
entropy-recovery state, best-checkpoint tracking, swarm geometry (`mean_pairwise`,
`std_pairwise`, `swarm_diameter`), all 8 reward components individually (`r_track` ...
`r_joint`, `r_contact`), `target_lost_rate` alongside `collision_rate`, and (2026-08-17)
`log_std_mean`/`mean_action_abs`/`mean_brake_reduction` (see Training loop above). `RUN_ID`
suffixes the log filename so parallel multi-seed runs don't collide. `plot_training.py`
mirrors `train.py`'s exact `RUN_ID` resolution logic and renders an **8x2** diagnostic grid
(grew from 7x2 with the `log_std`/`mean_action_abs`/brake panels; the "Failure Rates" panel
now overlays `collision_rate` and `target_lost_rate` together, and "Reward Components"
includes `r_contact`).

## Evaluation (`training/evaluate.py`)

Deterministic (no sampling) rollout of a saved actor. `load_actors(model_dir, run_id,
device, best=False)` loads the single shared-actor file and returns `{agent_name: actor}`
with the **same object** for every key (one shared policy, not per-agent weights), falling
back to the regular checkpoint if the dedicated model file is missing. Per-episode metrics
written to CSV: `collided, target_lost` (new 2026-08-17, read explicitly from `infos`, not
inferred from `terms` — same collision/target_lost distinction as `train.py`), `episode_len,
min_dist, tracking_rmse, avg_spacing_std, avg_diameter, avg_speed, avg_confidence` (new —
mean track confidence over the episode). `tracking_rmse` is deliberately still computed
against **ground truth** (`env.pos_t`), not the swarm's tracked estimate — evaluation should
measure objective performance, independent of what the policy could or couldn't know at the
time (that's what the reward is scored against instead). CLI: `--episodes, --model-dir,
--run-id, --seed (default 0), --device, --save-trajectory, --best`.

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
```

## What is NOT in this codebase (as of this writing)

- No ROS2, Gazebo, PX4, MAVSDK/MAVROS integration.
- No real hardware interface.
- No true decentralized sensing/communication model for the *neighbor* graph (it's centrally
  computed by the simulator — see `_repair_connectivity` above and `DECISIONS.md`). Note this
  is now inconsistent with how the *target* is sensed — target sensing became genuinely
  realistic (vision-based, range-limited, shared-on-contact) on 2026-08-17, while neighbor
  sensing is still exact ground truth. See `KNOWN_ISSUES.md` item 10 — deliberately deferred,
  not forgotten.
- No sensing noise, occlusion, or directional FOV cone on the target reading itself (item 11).
- No non-constant-velocity target motion — the target moves in a straight line at constant
  speed for the whole episode, fixed at `reset()` (item 9).
- No literal cross-agent-count weight transfer (`OBS_DIM` differs between `NUM_AGENTS=2` and
  `3`/`4`, so a `NUM_AGENTS=2` checkpoint's Actor architecture is incompatible with `3`/`4` —
  see `DECISIONS.md`).
- `NUM_AGENTS=4` has not yet been trained/validated with the current (post closing-speed-brake,
  post vision-tracking) code — everything in this doc as of `b59c139` has only been verified
  at `NUM_AGENTS=3`. This is the next real work item — see `TODO.md`/`SESSION_HANDOFF.md`.
