# Decisions — what was chosen, and why, so it doesn't get re-litigated

Each entry: the decision, the alternative(s) considered/rejected, and why. Source is either
an inline comment in the code (cited) or prior conversation context reconstructed alongside
the code (marked accordingly). If you're about to propose one of the "rejected" alternatives
below, read the reasoning first — several of these were measured, not guessed.

## Shared actor (parameter sharing) instead of per-agent actors

**Decision**: one `Actor` instance trained on all agents' pooled rollout data
(`training/train.py:116-122`).
**Rejected alternative**: independent actor per agent (`NUM_AGENTS` separate networks).
**Why**: the drones are homogeneous — identical dynamics, same local/neighbor-relative
observation shape — so pooling means each gradient step uses `NUM_AGENTS`x the data a
per-agent actor would see. The critic keeps a per-agent head instead (it needs to output
distinct values per agent even from shared state), so only the actor's training was pooled,
deliberately, not the whole architecture.

## Curriculum by `NUM_AGENTS` is staged comparative validation, not weight transfer

**Decision**: train and evaluate independently at `NUM_AGENTS=2`, then `3`, then `4`,
comparing results at each stage, rather than initializing the `N`-agent actor from the
`N-1`-agent checkpoint.
**Why not weight transfer**: `OBS_DIM = 10 + 7*EFFECTIVE_K` depends on `NUM_AGENTS` (via
`EFFECTIVE_K = min(K_NEIGHBORS, NUM_AGENTS-1)`), so the Actor's input layer shape literally
differs between `NUM_AGENTS=2` (`EFFECTIVE_K=1`) and `NUM_AGENTS=3`/`4` (`EFFECTIVE_K=2`).
A checkpoint trained at one `OBS_DIM` cannot be loaded into a network with a different
`OBS_DIM` without surgery that hasn't been built. The curriculum's actual value is
*validating the reward/config design generalizes* across agent counts, not warm-starting.

## `TARGET_DIST` derived from the Tammes problem (sphere-packing), not hand-picked

**Decision**: `_PACKING_RATIO` in `config.py` encodes the optimal-packing ratio for `{2, 3,
4}` points on a sphere; `TARGET_DIST` is computed from it and the physically-derived
`_EDGE_TARGET`. Unsupported `NUM_AGENTS` values raise `ValueError` rather than silently
falling back to a guess.
**Why**: gives a formation spacing that's geometrically justified for each specific agent
count instead of one constant reused across curriculum stages, where it would be
systematically wrong for at least some of them.

## `SAFE_DIST_ENTER`/`EXIT` and `REACTION_DIST` derived from a physical reaction-time model

**Decision**: `REACTION_DIST = 2 * MAX_ACTION_SPEED * (N_REACT * DT)` — a closing-speed ×
reaction-time-window model — rather than an arbitrary safety margin.
**Why**: ties the safety buffer to an actual physical quantity (how far two drones closing
at max speed can travel in the time it takes the policy to react, at `N_REACT=10` steps of
reaction budget) so it scales sensibly if `DT` or `MAX_ACTION_SPEED` ever change.

## Mutual k-NN neighbor graph with connectivity repair — centralized, not decentralized

**Decision**: neighbor locking uses ground-truth positions computed by the simulator
(`_compute_all_candidates`, `_relock_all`, `_repair_connectivity` in
`envs/formation_env.py`), explicitly documented in the module docstring as centralized.
**Why accepted despite being centralized**: it's a simulation-level simplification for
*topology maintenance* (deciding who's whose neighbor), while each agent's *policy* only
consumes its own + locked-neighbors' relative observations — i.e., decentralized execution
of the control policy, but centralized maintenance of the sensing graph. This is called out
explicitly as a known simplification, not something believed to already be a decentralized
sensing model. If the project needs a genuinely decentralized neighbor-discovery mechanism
later (e.g. for eventual real-hardware or ROS2/Gazebo work), that's a real design change,
not a bug fix — see `TODO.md`.

## Per-agent-involved collision-penalty attribution, not global-to-all

**Decision**: `step()` computes `colliding_agents` (the specific set of agents actually
involved in a given collision event) and only those agents receive the collision penalty
that step, rather than every agent in the episode being penalized whenever any collision
occurs anywhere in the swarm.
**Why**: motivated by MARL credit-assignment literature — a global penalty's signal-to-noise
ratio for any individual agent degrades as `NUM_AGENTS` grows (most of the penalty an agent
receives is for collisions it wasn't involved in), which actively hurts learning at higher
agent counts. Per-agent-involved attribution keeps the penalty informative regardless of
`NUM_AGENTS`.

## Global cohesion penalty replacing a per-neighbor "diverge" penalty

**Decision** (commit `78e1feb`): replaced a per-neighbor divergence penalty with a single
global cohesion penalty based on swarm-wide spread.
**Why**: the per-neighbor version had a "relock loophole" — an agent could satisfy its
per-neighbor divergence constraint locally while the swarm's overall diameter drifted, by
exploiting how neighbor locks get reassigned. A global cohesion term can't be gamed the same
way because it doesn't depend on which specific pairs are currently locked.

## Reward reweighting: track x4, safety magnitude reduced (commit `78e1feb`)

**Decision**: increased the tracking reward's relative weight substantially and reduced the
safety penalty's magnitude, alongside the cohesion-penalty replacement above.
**Why** (reconstructed from commit chronology and the `63274b1` follow-up "fix
cohesion/safety conflict causing collision-rate collapse"): the original balance apparently
let the safety and cohesion terms conflict in a way that collapsed the collision rate rather
than improving it — the exact mechanism of that conflict and how `63274b1` addresses it is
the most important thing to verify against the pending `NUM_AGENTS=4` run's actual results
(see `KNOWN_ISSUES.md` — this fix chain is unverified end-to-end as of this writing).

## Plateau-triggered entropy recovery instead of a fixed-calendar entropy floor

**Decision**: a `ReduceLROnPlateau`-style mechanism watching rolling `collision_rate`
(`train.py:44-65`) rather than a higher fixed `ENT_COEF_END` floor.
**Rejected alternative, with data**: raising `ENT_COEF_END` from `0.001` to `0.004` and
extending `TOTAL_STEPS` to 3M was tried first. Per the inline comment at `train.py:25-36`:
collision_rate plateaued at 0.78 for the last 2M steps (worse than any prior run) and
entropy *still* collapsed to -3.92 anyway — a fixed higher floor gave the policy more time
to commit harder to trading safety for tracking precision (which the reward genuinely
permitted at the time), it didn't stop the collapse. See `EXPERIMENT_LOG.md` for this run's
full details.
**Rejected alternative #2**: an `entropy < 0` trigger for recovery (tried before the
collision-rate-based version). Missed a real regression — an `NUM_AGENTS=3` shared-actor run
relapsed (`collision_rate` 0.02 → 0.25+) without entropy ever going negative, so that
trigger never fired when it should have.

**Follow-up fix (2026-08-14): `RECOVERY_MAX_TRIGGERS` raised 5 → 20.** The mechanism design
above was correct but underpowered for `TOTAL_STEPS=600_000`: a full `NUM_AGENTS=3`, 3-seed
run showed all 5 triggers spent by rollout ~110-125 (`5 * (RECOVERY_PATIENCE=15 +
RECOVERY_COOLDOWN=10)` = 125 of ~293 total rollouts), after which entropy annealed
unprotected for the remaining ~60% of training and `collision_rate` relapsed to 26-46% by the
end in every seed — the exact collapse `63274b1` was meant to fix, still present, just moved
later in training. `--best` checkpoints (2-8% collision_rate) proved the policy itself was
fine; it just wasn't protected long enough. See `KNOWN_ISSUES.md` item 1 and
`EXPERIMENT_LOG.md`'s N=3 3-seed analysis for the full evidence. This fix is itself
**unverified** as of this writing — pending a single-seed `NUM_AGENTS=3` rerun.

## Two decoupled "best" trackers (collision-rate-only patience signal vs. lexicographic checkpoint save)

**Decision**: `best_collision_rate` (drives only entropy-recovery patience) is tracked
separately from `best_score` (drives the actual checkpoint save, lexicographic:
`collision_rate` rounded to 2dp, then `-track` as tie-breaker).
**Why**: pure collision-rate-best was observed to prefer a checkpoint that achieved safety
by spreading the formation out excessively — worse `tracking_rmse` and `swarm_diameter` than
that same run's own final-step model. Rounding collision rate before comparing groups
meaningfully-equal safety levels together so tracking quality breaks ties within that group,
while a genuinely safer policy still always wins outright on the primary criterion.

## Gradient clipping (`max_grad_norm=0.5`) on both actor and critic

**Decision**: standard PPO/SB3-default gradient norm clipping, applied right before each
optimizer step (`train.py:320`, `325`).
**Why** (per inline comment): matters more than usual in this environment because a large
negative collision-penalty spike sits next to much smaller ordinary per-step rewards, which
can produce occasional large advantage excursions and correspondingly large gradients absent
clipping.

## Per-minibatch KL early stop, not just per-epoch

**Decision**: `TARGET_KL=0.02` is checked after every minibatch update, not only at the end
of each epoch (`train.py:340-345`).
**Why**: the previous per-epoch-only version let a single bad minibatch be followed by the
rest of that epoch's updates before anything reacted — checking per-minibatch reacts to a
trust-region violation immediately instead of after further damage.

## `entropy` computed as `-logp` in `Actor.evaluate()`, not `dist.entropy()`

**Decision**: entropy for the PPO entropy bonus is a Monte Carlo estimate under the true
tanh-squashed distribution (`-logp` of the sampled/evaluated action), not the closed-form
entropy of the pre-squash Gaussian.
**Why**: `dist.entropy()` on the underlying Gaussian ignores the tanh squashing entirely and
systematically overstates the policy's actual entropy — this was a previously-identified and
fixed bug.

## `log_std` clamped to `[-2.0, 0.5]`

**Decision** (commit `fc4c46f` "Clamp log_std (fix entropy runaway)"): bound the policy's
output log standard deviation.
**Why**: without a clamp, log_std could grow unbounded during training, which is the
"entropy runaway" the commit message names — the exact failure mode (what unclamped
log_std did to training) is worth checking `EXPERIMENT_LOG.md`/git history for the full
story if you need it, but the clamp itself is confirmed present and active in the current
`training/networks.py`.

## CPU as the default training device, not CUDA

**Decision**: `DEVICE = os.environ.get("DEVICE", "cpu")` (`train.py:74`) — CPU unless
explicitly overridden.
**Why**: measured on Kaggle, not assumed — CPU sustains ~330-440 steps/sec vs. CUDA's ~190
steps/sec for this specific network (2-layer, hidden=64/128). The network is small enough
that per-call GPU dispatch overhead dominates over any compute benefit. `steps_per_sec` is
logged every rollout regardless of device, specifically so any future default change (e.g.
if the network grows) is also a measured decision, not a guess.

## `_resolve_overlaps()` called only from `reset()`, not `step()`

**Decision**: overlap resolution runs once at episode start to de-overlap the random spawn,
and is not re-invoked during the episode.
**Why** (inferred from design intent, not an inline comment — flagged as reconstructed
reasoning): it exists to fix a spawn-time artifact, not as an in-episode collision-avoidance
mechanism — collision avoidance during the episode is the policy's job via the safety
reward term, and having the simulator silently teleport/adjust agents mid-episode to resolve
overlaps would undermine training the policy to avoid the situation in the first place.

## Closing-speed brake: a deterministic action-space safety layer, not another reward term

**Decision**: cap the *closing* component of an agent's commanded velocity (the part actually
shrinking the gap to a specific other agent) once already inside `SAFE_DIST_ENTER`, ramping
linearly to zero exactly at `COLLISION_DIST`. Lives in `envs/formation_env.py:step()`, applied
to every agent-pair using pre-step positions (so it reflects simultaneous action). Doesn't
touch the reward at all.
**Rejected alternatives, with data (see `EXPERIMENT_LOG.md` for full numbers)**: raising
`RECOVERY_MAX_TRIGGERS` 5→20 (falsified — entropy barely moved despite far more frequent
boosts), reshaping `r_safety`'s bonus zone to peak at `EDGE_TARGET` (falsified — 40% collision
rate, no better than before), adding a diameter floor (insufficient — engaged but overwhelmed
by `r_track`'s pull). **Six consecutive reward-shape/schedule interventions failed** across
commits `78e1feb` through `67ccfd7` before this.
**Why a structural change instead of another reward guess**: instrumentation
(`log_std_mean`/`mean_action_abs`, see `EXPERIMENT_LOG.md`) showed the actual driver of the
collision-rate collapse was never entropy/exploration-noise collapsing — `log_std` stayed
nowhere near its floor. The policy was learning to commit to larger-magnitude actions over
training, and a drone closing at high commanded speed has less room to recover from a close
encounter than one making smaller corrections, regardless of how much stochastic noise sits on
top. No amount of reward shaping had addressed *that* variable, because none of the prior
fixes had ever measured it. This also matches an independent recommendation from a supervisor
review earlier in the project's history ("I strongly prefer a learned nominal controller +
deterministic safety layer... rather than expecting PPO to discover collision avoidance purely
from a `-300` reward").
**Result**: `collision_rate=0.000` across every seed/duration tested since (6/6 configurations
at exactly 0%, later 8/9 across 3M-step 3-seed replication, one isolated exception below).
**Known limitation, found later, not swept under the rug**: the no-crossing property was
proven algebraically for two agents at a time (given the current constants, the worst-case
per-step gap reduction is a fixed fraction of the remaining gap, never reaching zero in finite
steps) — it was never separately proven for an agent braking against two simultaneous threats
from different directions. A 3M-step run produced exactly one isolated collision event (and a
separate 1/100 eval-episode reading elsewhere), consistent with this gap, not a regression.
See `KNOWN_ISSUES.md`.

## Vision-based cooperative target tracking, replacing ground-truth telemetry

**Decision**: every drone previously received the target's exact position/velocity every
step, regardless of distance — flagged from the start (`docs/ai_context`'s original writeup)
as a deliberate but unexamined simplification. Replaced with a sensing model: each drone gets
a direct reading only when the target is within `SENSOR_RANGE`; the swarm shares any drone's
current reading instantly; if nobody currently has contact, the swarm dead-reckons from the
last known position/velocity for up to `LOST_TIMEOUT_STEPS` (2s), after which the episode
terminates (`target_lost`) — a second real mission-failure mode alongside collision.
**Why omnidirectional range-limited, not a directional FOV cone**: the simulation has no
heading/orientation state anywhere — drones are point masses with position and velocity only.
A true camera-FOV cone needs heading added as a new state dimension (plus how it's actuated),
a materially bigger change. An omnidirectional multi-camera rig is a realistic enough
approximation for a small quadrotor and was the explicitly scoped-down choice for this pass.
**Why `SENSOR_RANGE = TARGET_DIST + 2*REACTION_DIST`**: reuses the *exact* formula the env
already uses for initial spawn radius (`reset()`), rather than inventing a new constant —
drones start every episode in direct contact by construction (spawn distance's upper bound
equals `SENSOR_RANGE`), a "just detected it, closing in" narrative for free.
**Why UWB-realistic neighbor sensing was explicitly deferred, not bundled in**: UWB is the
right technology for *inter-drone* sensing (works without GPS or line-of-sight — the
presumed real scenario is GPS-denied), but gives clean range, not clean bearing or velocity —
modeling that honestly would touch the neighbor observation too, on top of everything already
changing for the target. Given a fresh, sharp lesson this session about bundling too many
changes into one untested commit, this was deliberately scoped out as a separate, future
design pass.
**Why `r_track`/`r_velocity` score against the tracked estimate, not `self.pos_t` directly**:
using ground truth in the reward while the observation uses the estimate would reproduce
exactly the observation/reward mismatch class a supervisor review already caught once for
`r_safety` (see the `K_NEIGHBORS = NUM_AGENTS - 1` decision above) — not repeating it here.
`_dist_to_target()` (ground truth) is kept only as a diagnostic (`true_track_err` in `infos`,
and `evaluate.py`'s `tracking_rmse`, which is deliberately ground-truth-based since evaluation
should measure objective performance, not what the policy could excuse itself for not
knowing) — never fed into the reward or observation.
**Why the diameter floor (added in the collision-fix chain above) was disabled, not
recalibrated**: it was built to stop dangerous tight convergence *before the brake existed*.
With the brake handling that deterministically and independent of diameter, the floor was
found to be actively counterproductive for the new task — pulling the swarm's diameter up
toward/above `MIN_DIAMETER=10` geometrically implies a larger individual radius from the
target for evenly-spaced agents, pushing drones past `SENSOR_RANGE` more than necessary.
Measured: disabling it (`DIAMETER_FLOOR_WEIGHT` 1.0→0.0, not deleted) roughly halved
`target_lost_rate` (31-40%→18-14% at matched 600k steps) with no tracking regression. See
`EXPERIMENT_LOG.md` for the full before/after data.
**Result** (3-seed, 3M steps, no diameter floor): 0-3% `target_lost_rate`, 0-1%
`collision_rate` (the same rare brake edge case, not new), best tracking accuracy seen all
session (1.52-2.99 RMSE). Merged to `main` (`b59c139`).
**Deliberately deferred beyond this change** (not attempted, not forgotten): sensing
noise/occlusion on the target reading itself, a directional FOV cone (needs heading), and
non-constant-velocity target motion (the 2s dead-reckoning grace period is currently
*mathematically exact*, not an approximation, since the target moves at constant velocity for
the whole episode — this only becomes a real approximation once target motion is made less
trivial, which is itself a natural next step, not yet taken).
