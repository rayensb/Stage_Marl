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

## Multi-pass brake convergence (POCS), replacing a single sequential sweep

**Decision**: `_apply_brake` (`envs/formation_env.py`) repeats its per-neighbor closing-speed
correction across all neighbors until no further correction is needed, or `NUM_AGENTS` passes
are used, whichever comes first — instead of one sequential sweep through neighbors.
**Rejected alternative**: the original single-pass version (correct for exactly one active
threat, but a later neighbor's correction could silently reintroduce a violation of an
earlier neighbor's constraint, never re-checked).
**Why**: each per-neighbor closing-speed cap is a halfspace constraint on velocity
(`dot(v, dir_to_b) <= max_closing`), and the correction applied is exactly the Euclidean
projection onto that halfspace. Repeating the sweep until stable is POCS (projection onto an
intersection of convex sets) — a standard, provably-limit-convergent technique whenever the
intersection is nonempty, which it always is here physically (`v=0` satisfies every
constraint, since `max_closing` is never negative). This directly targets the confirmed
`NUM_AGENTS=4` mechanism: each agent has 3 simultaneous "others" under full connectivity
(`K_NEIGHBORS=NUM_AGENTS-1`) instead of `NUM_AGENTS=3`'s 2, and the single-pass version's
training-time `collision_rate` stayed nonzero and scattered across an entire 3M-step run
instead of converging (see `EXPERIMENT_LOG.md`'s `N=4` validation entry).
**Important precision, added after a supervisor review pushed back on an earlier overclaim**:
"converges in the limit" (proven) is a weaker claim than "`NUM_AGENTS` passes reaches exact,
zero-residual convergence" (not proven — `NUM_AGENTS` is used only as a cheap iteration cap,
not a derived bound). The early-exit is what actually terminates the loop in practice; if it
never fires, the loop stops at the cap with whatever residual violation remains, silently. Two
diagnostics (`_last_brake_passes`, `_last_brake_violation`) were added specifically to make
this checkable rather than assumed — see the brake-instrumentation entry below.
**Verification**: adversarially stress-tested (not just trained-and-hoped) — 2 agents head-on
and 4 agents simultaneously converging on a shared centroid, both starting outside
`SAFE_DIST_ENTER` and always commanding `MAX_ACTION_SPEED` toward each other/the centroid
every step. Minimum distance asymptotically approached but never crossed `COLLISION_DIST`,
zero residual constraint violation after `_apply_brake` on every step of both tests. Not a
formal proof, and specific to this system: `v_closing` is each agent's *own* velocity
component toward the other, not the relative closing velocity (`v_a - v_b`) — the stress
tests hold because every agent runs through the identical symmetric formula, with no way for
one side to be unconstrained while the other closes at speed. That symmetry assumption stops
holding once a real, independently-controlled vehicle is involved — see
`deployment/docs/PHASE2_HANDOFF.md` and `KNOWN_ISSUES.md`. Reformulating with explicit
relative velocity is queued, not yet done.
**Result**: validated as part of the Phase2-combined bundle (below) — 3/3 `N=4` seeds clean at
0% `collision_rate` across a 5M-step run, first time `N=4` converged cleanly. See
`EXPERIMENT_LOG.md`.

## Direct brake-engagement reward penalty — v1 (linear) tested, replaced with v2 (threshold)

**Decision**: add `r_brake`, a reward penalty proportional to how much closing speed the
brake actually removed (`brake_reduction`, already computed every step, previously only
logged) — closing a gap the indirect `r_safety` urgent-zone penalty leaves open: that penalty
is proximity-based, not action-based, so a policy could keep commanding more-than-safe closing
speed without anything specifically penalizing *that choice*, only ambient proximity.
**v1 (linear from zero)**: `r_brake = BRAKE_PENALTY_COEF * brake_reduction`. **Result**:
`collision_rate` genuinely fixed (88/59/269 nonzero rollouts out of 1465 per seed dropped to
14/10/15, all three seeds clean for their entire last 10 rollouts) — but `swarm_diameter`/
`avg_min_dist` both ran noticeably wider than before and `tracking_rmse` worsened slightly.
**Diagnosis**: a linear-from-zero penalty can't distinguish a trivial, routine nudge (a
fraction of a percent of `MAX_ACTION_SPEED`) from a real emergency correction — it taxes both,
so the policy's cheapest way to minimize the penalty is to avoid entering the brake's trigger
zone at all (spread out more than necessary), not just avoid needing a large correction once
inside it — "a good driver rarely triggers ABS" doesn't mean a good driver's ABS light can
never so much as flicker.
**v2 (threshold)**: `r_brake = BRAKE_PENALTY_COEF * max(0, brake_reduction -
BRAKE_PENALTY_THRESHOLD)` — zero penalty below the threshold (matches "normal braking" being
free), only the excess above it is taxed. `BRAKE_PENALTY_THRESHOLD` is derived from the
brake's own geometry, not picked independently: `0.5 * MAX_ACTION_SPEED` corresponds to a
full-speed closing approach already halfway through the safety zone toward `COLLISION_DIST` —
a genuinely aggressive, late correction, not a minor one. `BRAKE_PENALTY_COEF` itself
unchanged from v1 (only the zero-point moved).
**Result**: validated as part of the Phase2-combined bundle — collision stayed at 0% (3/3
`N=4` seeds) without the v1 diameter/tracking regression. See `EXPERIMENT_LOG.md`.

## Brake instrumentation (`brake_passes`, `brake_violation`, solo/multi split)

**Decision**: `_apply_brake` now stashes three diagnostics on `self` instead of returning
them (to keep its return signature unchanged — `deployment/inference_node.py` depends on the
exact `(corrected_vel, brake_reduction)` unpack): `_last_brake_passes` (how many POCS passes
this call took), `_last_brake_violation` (worst remaining constraint violation after the loop
— should be ~0 if converged before the pass cap), `_last_brake_k_active` (how many other
agents were within `SAFE_DIST_ENTER` of each agent, letting callers split `brake_reduction`
into solo vs. multi-neighbor engagement). Threaded to 4 new `training_log_*.csv` columns
(`mean_brake_passes`, `max_brake_violation`, `mean_brake_solo`, `mean_brake_multi`).
**Why**: turns the multi-pass convergence's "should converge, verified only by stress test"
claim into something checkable against every real training rollout, not just two synthetic
scenarios — and the solo/multi split is specifically what the N-aware-margin hypothesis (next
entry) needs to check whether its targeted mechanism (simultaneous-neighbor count) is the
right one.
**Status**: pure logging addition, doesn't change training behavior — safe to build on
without re-validating anything upstream of it.

## N-aware safety margin — `EDGE_TARGET` scaled by simultaneous-neighbor count

**Decision**: `EDGE_TARGET = SAFE_DIST_EXIT + REACTION_DIST + max(0, K_NEIGHBORS - 2) *
REACTION_DIST` — adds one extra `REACTION_DIST` of margin per simultaneous neighbor beyond
the validated `NUM_AGENTS=3` case (`K_NEIGHBORS=2`). Unchanged at `N=3` (`max(0, 2-2)=0`), one
extra `REACTION_DIST` (1.20) at `N=4` (`max(0, 3-2)=1`).
**Rejected framing**: an arbitrary "make `N=4` bigger" fudge (e.g. a flat multiplier on
`TARGET_DIST` regardless of mechanism).
**Why this form specifically**: targets *valence* (how many simultaneous "others" a drone's
safety layer has to negotiate with at once), the actual mechanism the brake's multi-pass fix
addresses, rather than `N` directly — `K_NEIGHBORS = NUM_AGENTS - 1` is full connectivity in
this project, so valence and `N` happen to move together here, but the formula is written
against the mechanism, not the coincidence.
**Known risk, flagged before testing**: a wider formation could push drones toward/past
`SENSOR_RANGE` more than necessary — the same failure mode the old diameter floor caused once
vision-tracking existed (see the diameter-floor entry above). Worth checking tracking metrics
specifically, not just collision, once tested.
**Status**: tested standalone (single seed, `N=4`) alongside the brake fixes, then folded into
the Phase2-combined bundle for the full validation — see `EXPERIMENT_LOG.md`.

## 3D (XYZ) `r_spread`, and a geometry bug caught by supervisor review before it shipped broken

**Decision**: replace `r_spread`'s horizontal-only (XY bearing-sort) construction with true
pairwise 3D angular separation between locked-neighbor direction vectors — for each pair of a
drone's neighbors, the angle between their direction vectors from that drone, penalizing how
far the *minimum* pairwise angle sits from an ideal.
**Why**: `N=4`'s exact-consistent target formation (a regular tetrahedron — see the geometric
feasibility check in `EXPERIMENT_LOG.md`) is inherently non-planar, and the old horizontal-only
shaping was blind to the one axis (vertical) where reaching that optimum actually requires
structure. See `KNOWN_ISSUES.md` item 5 (now resolved) for the full mechanistic argument.
**The bug**: the first implementation used `_PACKING_RATIO`'s angle (109.47°/120° for
`K=2`/`3`) as the ideal — the angle a formation's *target* sees between two drones (a global,
target-viewpoint quantity). `r_spread` actually penalizes the angle a *drone* sees between its
own neighbors (a local, drone-viewpoint quantity) — a different geometric quantity that is not
interchangeable with the first, despite both being expressible as degrees between two drones.
Using the wrong one was pushing neighbor directions wider than the actual target formation
has, fighting convergence rather than helping it.
**The fix**: `_IDEAL_NEIGHBOR_ANGLE = math.pi / 3` (60°) — proven, not guessed: placing
`k+1` vertices as standard basis vectors (a textbook regular-simplex construction; `k=2` gives
the `N=3` equilateral triangle, `k=3` gives the `N=4` regular tetrahedron), the angle between
any two edges meeting at a shared vertex has `cos = 1/2` for *any* `k` — exactly 60°,
independent of neighbor count. Both `K=2` (`N=3`) and `K=3` (`N=4`) map to the same constant,
which is why `_IDEAL_NEIGHBOR_ANGLE` is a single module-level constant, not a per-`K` dict.
**Verification infrastructure added specifically because this bug shipped once**:
`test_geometry.py` computes both the global (target-viewpoint) and local (drone-viewpoint)
angles from the same regular-simplex construction, asserts they're different quantities,
and asserts `_IDEAL_NEIGHBOR_ANGLE` matches the local one exactly (to `1e-9`) — this is the
check that would have caught the bug before it shipped, not after.
**Timeline note, for provenance when reading old logs**: the Phase2-combined 5-seed
validation run (`EXPERIMENT_LOG.md`) was already in flight on Kaggle when the angle bug was
found and fixed (commit `ac98d67`) — that validation used the *wrong* (109.47°/120°) angle.
The corrected 60° version was re-tested in isolation afterward (`xyz-spread-fixed` branch,
`spread_fixed_seed1`) specifically to check whether the bug had actually cost anything
measurable, before trusting the combined result. See `EXPERIMENT_LOG.md` for both results.

## Phase2-combined: merge four independently-motivated fixes, then validate as a bundle

**Decision**: after the brake multi-pass fix, brake-engagement penalty, N-aware margin, and
XYZ `r_spread` were each individually motivated by the confirmed `N=4` collision mechanism
(`KNOWN_ISSUES.md` items 5 and 8), merge all four into one branch (`phase2-combined`) and run
one 5-seed validation sweep (`N=2,3,4`) rather than testing each in complete isolation through
to a full multi-seed conclusion.
**Tension with the project's own stated discipline**: `AI_CONTEXT.md`'s working-pattern note
explicitly warns that bundling multiple changes into one tested commit has burned this project
before (twice, both times flagged afterward as a lapse) and recommends isolating one variable
per test. This decision accepts that risk deliberately, not by accident, for a specific
reason: the first two fixes (brake multi-pass, brake-engagement penalty) target the same
confirmed mechanism and were proposed together from the start (`TODO.md`'s original Phase 1);
the second two (N-aware margin, XYZ spread) are a distinct "formation-quality" pair, tested
standalone first (single-seed each, per `EXPERIMENT_LOG.md`) before being folded in — so the
bundle is two already-individually-checked pairs combined for a final full-scale validation,
not four totally untested ideas thrown together at once. Kaggle's 5-concurrent-session cap was
also a practical factor in not running every permutation separately.
**Result**: 3/3 `N=4` seeds clean (0% collision), best seed's `tracking_rmse` 1.43 — the best
of the project to that point; 1/1 `N=3` clean collision with some `target_lost` noise in one
seed (a second `N=3` seed, `phase2_n3_seed2`, was run afterward specifically to check whether
that noise was real or seed-variance — it wasn't, see `EXPERIMENT_LOG.md`); 1/1 `N=2` clean.
**Status**: this is the validated reference point Phase 3 (below) branches from. `main` itself
was not fast-forwarded to it in this pass — a manual `git push origin phase2-combined:main` is
still pending (blocked by the permission classifier on a direct push from the agent; verified
as a clean fast-forward, no conflicts) — see `TODO.md`.

## Phase 3: bundling five sustained-flight changes as an explicit "leap of faith," not a lapse

**Decision**: bundle five changes (longer episodes, a larger network, ground awareness,
per-axis Z/XY speed limits, dynamic target motion) into one branch (`phase3-resilience`, off
`phase2-combined`) and test them together, rather than isolating each first.
**Why this one is different from the Phase2-combined bundling above**: this was an explicit,
informed user call, not a default — `training/diagnose_horizon.py` had shown the best
Phase2-combined checkpoint's tracking error nearly quadrupling (0.95 → 3.66) once run for 60
simulated seconds instead of its 10-second training horizon, in a window (20-30s) that lines
up with where the real PX4/Gazebo deployment actually crashed (15-40s) — real, load-bearing
evidence, not a hunch (see `deployment/docs/PHASE2_HANDOFF.md`). Given that evidence and
limited time, the user chose to test the whole resilience hypothesis at once, with an
explicitly agreed fallback already in place: if the combined run doesn't produce good results,
isolate each change individually to find out which one(s) actually mattered. This makes the
bundling a deliberate, risk-acknowledged trade of attribution clarity for speed, with a
pre-agreed recovery plan — not the same failure mode `AI_CONTEXT.md`'s working-pattern note
warns about (which was about *accidentally* losing attribution, not choosing to).
**Result**: mixed, and the bundling's risk materialized partially, but diagnosably. Safety
(collision, ground_strike) succeeded completely: 0% collision across all 3 seeds,
`ground_strike_rate` literally 0/209 rollouts in every seed. Tracking failed catastrophically:
`target_lost_rate` 89-100%, flat from the very first logged rollout through the full 3M steps,
never improving in any seed. Because the failure was flat-from-step-1 rather than a mid-run
regression, and because collision (subject to the same "9x fewer episodes" risk from the
longer-episode change) converged fine, the cause was isolable *without* reverting to
one-variable-at-a-time testing — see the `LOST_TIMEOUT_SEC` entry below.
**Status**: root cause diagnosed and a fix (env-overridable `LOST_TIMEOUT_SEC`, swept across
6/10/18s) is in progress, not yet confirmed. See `EXPERIMENT_LOG.md`/`CURRENT_STATE.md`.

## Ground awareness: clamp + reward + termination, mirroring the brake's pattern exactly

**Decision**: give the ground plane (previously not modeled at all — confirmed by search, not
an oversight being fixed blind) the same three-part treatment already validated for
inter-agent collision: a deterministic `_apply_ground_clamp` (single-pass, not multi-pass —
unlike two drones, the ground doesn't move and can't itself be a conflicting constraint, so
one pass is exact), a graduated `r_ground` reward penalty as altitude approaches `GROUND_Z`,
and a `ground_strike` termination flag, distinct from `collision`/`target_lost` (same
"distinct failure modes stay distinctly labeled" pattern already established).
**Why reuse the pattern instead of reward-shaping alone**: reward shaping alone was already
proven insufficient for the inter-agent case (six failed attempts before the brake) — no
reason to expect it'd fare better against a hard physical floor. `GROUND_URGENT_COEF` mirrors
`SAFETY_URGENT_COEF`'s ramp shape; `GROUND_STRIKE_PENALTY` matches `r_collision_global`'s -300
magnitude (not the softer -200 `TARGET_LOST_PENALTY`) — a real ground strike is judged at
least as severe as an inter-agent collision, not a softer, recoverable-in-principle failure
like losing target contact.
**A real bug found while smoke-testing, not shipped blind**: naive ground-safety spawn
clamping (applied once in `reset()`, before `_resolve_overlaps()`) could collapse inter-agent
vertical separation and cause an immediate step-0 collision. Fixed by alternating
`_resolve_overlaps()` and the Z-clamp in a joint fixed-point loop (up to 20 iterations) in
`reset()` instead of applying each once in sequence — verified across 500 seeds at `N=2/3/4`
with zero violations of either constraint before this shipped.
**Result**: `ground_strike_rate` was 0/209 rollouts in every one of the 3 Phase 3 validation
seeds — the mechanism worked cleanly from the very first test, no iteration needed (unlike the
brake, which needed the multi-pass fix after its first `N=4` test). See `EXPERIMENT_LOG.md`.

## Cruise-altitude preference + deterministic Z-velocity smoothing

**Decision**: two separate, additive mechanisms for vertical flight quality, distinct from
`GROUND_SAFE_ENTER`'s hard crash-avoidance boundary: `r_altitude`, a graduated reward penalty
for loitering below `CRUISE_ALT_MIN=1.5` (a comfort preference, not a safety mechanism — no
bonus for flying high, only a penalty for dipping low); and `Z_SMOOTHING_ALPHA`-based
deterministic blending of each step's commanded Z velocity with the previous step's *actual*
Z velocity, applied in `step()` *before* the brake/ground clamp so those safety-critical
corrections always retain final say over a comfort-only smoothing pass.
**Why confirmed empirically before implementing anything**: the user reported jiggling; rather
than assume it was real (or assume a specific cause), a standalone script loaded the best
Phase 3 checkpoint and ran 400 deterministic steps, measuring 8-25% of steps per agent flipping
vertical-velocity sign, with one agent briefly dipping to `z=0.98`. Confirmed real before any
fix was written.
**Why a deterministic smoothing filter, not a reward-only fix**: same "don't just hope the
policy learns it" reasoning already validated for collision (the brake) and the ground (the
clamp) — a reward term can only ever bias behavior statistically; a filter guarantees the
property. `Z_SMOOTHING_ALPHA=0.3` (30% new command, 70% carried over) is a starting guess at
enough damping to visibly reduce sign-flipping without making vertical response sluggish —
flagged as the first thing to retune if it overshoots either way, same spirit as every other
"starting guess" constant in `config.py`.
**Status**: implemented and bundled into the same branch as the `LOST_TIMEOUT_SEC` fix under
explicit time pressure ("include the altitude thing then test all we don't have much time") —
smoke-tested locally, not yet re-measured against a trained checkpoint post-fix. Verifying the
jiggling is actually reduced (not just that the code runs) is a pending step once the current
Kaggle sweep completes — see `TODO.md`.

## `LOST_TIMEOUT_SEC` made env-overridable and swept, instead of guessing one new value

**Decision**: `LOST_TIMEOUT_SEC` (previously a flat `2.0`, hardcoded) became
`float(os.environ.get("LOST_TIMEOUT_SEC", 2.0))`, and 5 Kaggle kernels were launched sweeping
6/10/18 seconds (2/2/1 seeds) rather than picking one new value analytically.
**Why the original 2.0 broke**: it was tuned against `MAX_STEPS=200` (10s) episodes and never
revisited when Phase 3 raised `MAX_STEPS` to 1800 (90s) — a fixed 2-second grace window has
~9x more independent opportunities to be exceeded by an ordinary, recoverable contact gap over
a 9x longer episode, even with zero change in the swarm's actual moment-to-moment tracking
competence. This diagnosis was reasoned from evidence, not assumed: `target_lost_rate` was
flat at 89-100% from the very first logged rollout (not a mid-training regression, which would
suggest a learning-dynamics cause), and `collision_rate` — subject to the same "9x fewer
episodes" data-starvation risk from the longer-episode change — converged fine, arguing against
pure data-starvation as the explanation and pointing specifically at the timeout/episode-length
mismatch instead.
**Why sweep instead of computing a new fixed value**: no clean formula maps "how long should
dead-reckoning be trusted" to `MAX_STEPS` — it depends on how often real contact gaps occur and
how long they typically last, which is a property of the trained policy's behavior, not a
constant that can be derived the way `REACTION_DIST` or `SAFE_DIST_ENTER` were. Measuring
across a spread of values is the honest way to find a working one, consistent with this
project's established pattern of measuring rather than guessing at reward/config constants.
**Status**: in progress. 4 of 5 kernels (6s x2, 10s x2) launched and running; the 18s kernel
was blocked by Kaggle's 5-concurrent-session cap even after the other 4 were confirmed running
and everything else confirmed complete — see `KNOWN_ISSUES.md` and `SESSION_HANDOFF.md` for
the live troubleshooting state.
