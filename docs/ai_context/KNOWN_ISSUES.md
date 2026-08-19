# Known Issues

Open problems as of commit `b59c139`, 2026-08-17 (vision-tracking merged to `main`). For
issues that were investigated and resolved, see `EXPERIMENT_LOG.md` (they're experiments with
a conclusion, not open issues).

## 1. [RESOLVED] Collision-rate collapse recurred late in training

**Resolved 2026-08-17.** This item went through several wrong diagnoses before the real one:
`RECOVERY_MAX_TRIGGERS` raised 5→20 (falsified — entropy barely moved), `r_safety`'s bonus
zone reshaped to peak at `EDGE_TARGET` (falsified — no better), a diameter floor added
(insufficient — engaged but overwhelmed). Direct instrumentation (`log_std_mean` vs.
`mean_action_abs`) finally showed the real driver: the policy was learning to commit to
larger-magnitude actions over training, not losing exploration noise — `log_std` never
approached its floor in any of these runs. The actual fix was a **deterministic action-space
safety layer** (the closing-speed brake, `envs/formation_env.py:step()`), not a reward change
at all — see `DECISIONS.md` and `EXPERIMENT_LOG.md` for the full chain and data.
**Current status**: `collision_rate=0.000` across 6/6 tested configurations at the time the
brake landed, and 8/9 across a later 3-seed, 3M-step replication (one isolated exception, see
item 8 below). Not revisited further as an open issue — see item 8 for the one known caveat.

## 2. `readme.txt` is stale and describes a removed architecture

**Symptom**: `readme.txt` documents a per-drone-file save scheme
(`models/actor_droneN[_SEED].pt`) that no longer exists — `train.py` saves one shared file.
It's silent on the shared-actor architecture, best-checkpoint tracking, entropy recovery,
`evaluate.py --best`, several env vars (`NUM_AGENTS`, `N_REACT`, `VELOCITY_WEIGHT`, `SEED`,
`DEVICE`, and now `TOTAL_STEPS`), and every fix/feature from this session: the closing-speed
brake, vision-based cooperative target tracking (`SENSOR_RANGE`/`LOST_TIMEOUT`/confidence
tracking), `target_lost` as a second failure mode, and the `r_collision` logging fix. Gap has
only grown since this item was first written.
**Cause**: normal doc drift — the file wasn't updated as the architecture evolved from
per-drone actors to a shared actor.
**Status**: confirmed, not fixed (out of scope for the documentation task that created this
`docs/ai_context/` system — project-logic/doc changes were explicitly excluded from that
task). Low urgency functionally (doesn't affect running code) but will mislead a human or AI
reading only `readme.txt` instead of `docs/ai_context/`.
**Next step**: rewrite `readme.txt` to match current architecture, or replace it with a
pointer to `docs/ai_context/AI_CONTEXT.md`.

## 3. Reward-component logging semantics changed mid-project (per-episode-sum → per-step)

**Symptom**: commit `63274b1`'s message states it changed reward-component metrics from
per-episode-sum to per-step logging "to avoid episode-length artifact." This means CSV logs
produced *before* `63274b1` have `r_track`/`r_safety`/etc. values on a different scale
(confounded by episode length) than logs produced after.
**Status**: UNVERIFIED whether any analysis has re-normalized or excluded pre-fix logs when
comparing across runs. If comparing an `NUM_AGENTS=3` run's reward components against a
post-`63274b1` `NUM_AGENTS=4` run, check which side of this commit each log was produced on
before drawing conclusions from the raw numbers.
**Next step**: when analyzing any CSV log, check its git-log-relative provenance (or the
`total_steps`/`episode` ratio, which reveals episode length and can hint at which logging
convention was in effect) before comparing magnitudes across runs.

## 4. Neighbor graph is centralized, not decentralized (design limitation, not a bug)

**Symptom**: `_repair_connectivity` and friends in `envs/formation_env.py` use ground-truth
global position knowledge to maintain the mutual-k-NN neighbor lock graph.
**Status**: this is a **documented, deliberate simplification** (see `DECISIONS.md`), not a
bug — flagged here only because if the project's goals ever require genuinely decentralized
sensing (e.g. moving toward real hardware or ROS2/Gazebo integration), this is a real
architecture item, not a quick fix. See `TODO.md` research/future section.

## 5. `r_spread` formation-spread metric is horizontal-only — now a plausible, mechanistically-grounded contributor to the `N=4` difficulty, not just an unverified limitation

**Symptom**: documented directly in the `envs/formation_env.py` module docstring — the
spread reward component only considers horizontal (x/y) spacing, not vertical (z) spread.
**Geometric feasibility check (2026-08-19, verified)**: it was raised whether `TARGET_DIST`
(~4.78 at `N=4`) and `EDGE_TARGET` (7.80, fixed regardless of `N`) are even simultaneously
satisfiable — can every drone really be 4.78 from the target *and* 7.80 from every other drone
at once? Checked directly: yes, exactly. `_PACKING_RATIO[4] = (8/3)**0.5` is the
circumradius-to-edge ratio of a **regular tetrahedron**, and `TARGET_DIST` is solved backward
from `EDGE_TARGET` using exactly that ratio (`config.py`) — a real regular tetrahedron with
circumradius 4.78 and edge 7.80 satisfies both constraints exactly, no contradiction (same
check for `N=3`: equilateral triangle on a great circle, also exact). **The target geometry
itself is not infeasible.**
**What's actually different at `N=4`, and why this item now matters more**: a triangle (`N=3`'s
exact solution) is trivially planar — consistent with what `r_spread` can already see and
shape. A regular tetrahedron (`N=4`'s exact solution) is inherently **non-planar** — no 4
points can be simultaneously center-equidistant and pairwise-equidistant while coplanar.
`r_track` and `r_safety` jointly imply the tetrahedron as a reward optimum in principle, but
`r_spread` — the only term with any angular-arrangement shaping — is blind to the one axis
(vertical) where reaching that optimum actually requires structure at `N=4`. The swarm has to
stumble onto the correct out-of-plane arrangement through pairwise-distance terms alone, with
no active shaping toward it, unlike `N=3` where the correct arrangement is exactly what the
existing horizontal-only shaping already rewards.
**Status**: not confirmed as *the* cause of the `N=4` collision persistence (item 8 above has
its own independently-sufficient mechanism), but plausible as a contributing factor and worth
fixing regardless.
**Proposed fix (not yet implemented, queued in `TODO.md` Phase 2)**: replace the 2D bearing-sort
construction with pairwise 3D angular separation between neighbor direction vectors
(`arccos(dot(dir_i, dir_j))`), penalizing when the minimum pairwise angle is below the
Tammes-ideal angle for that neighbor count (180° for 2 neighbors, 120° for 3 — reusing the same
spherical-packing logic `_PACKING_RATIO` already uses for `TARGET_DIST`, applied locally around
each drone instead of globally around the target).

## 6. Curriculum can't literally transfer weights across `NUM_AGENTS`

**Symptom**: `OBS_DIM` differs between `NUM_AGENTS=2` (`EFFECTIVE_K=1`) and `NUM_AGENTS=3`/`4`
(`EFFECTIVE_K=2`), so an `Actor` checkpoint trained at one agent count has an incompatible
input-layer shape at another.
**Status**: not a bug — see `DECISIONS.md` for why the curriculum is comparative validation
rather than transfer learning. Listed here as a known limitation in case a future session is
tempted to "fix" this by trying to load a mismatched checkpoint — it won't work without
first building explicit surgery (e.g. padding/slicing the input layer, or an
architecture that's agent-count-invariant), which hasn't been attempted.

## 7. Local sandbox has no working CUDA

**Symptom**: this dev machine's NVIDIA driver is too old for the installed PyTorch/CUDA
version, so `DEVICE=cuda` silently isn't usable locally.
**Status**: not a project bug — doesn't block anything, since CPU is the project's actual
measured-optimal default anyway (see `DECISIONS.md`). Documented so a future session doesn't
waste time debugging what looks like a CUDA failure but is just this machine's driver.
**Impact**: no local run can validate GPU-path behavior specifically, if that ever becomes
relevant (e.g. if the network grows enough that CUDA becomes faster again).

## 8. Closing-speed brake's no-crossing property fails under simultaneous multi-threat braking — now confirmed at `NUM_AGENTS=4`, not just a theoretical gap

**Symptom**: the brake's algebraic no-crossing guarantee (worst-case per-step gap reduction is
a fixed fraction of the remaining gap, never reaching `COLLISION_DIST` in finite steps) was
derived for two agents at a time. The per-pair velocity correction in `step()` is applied
sequentially — correcting for neighbor B, then neighbor C using the already-B-corrected
velocity, without re-checking that C's correction didn't reintroduce a violation of B's
constraint. At `NUM_AGENTS=3` each agent already has 2 simultaneous "others" (`K_NEIGHBORS =
NUM_AGENTS-1` is full connectivity) and this rarely mattered; at `NUM_AGENTS=4` each agent has
3.
**Status**: **confirmed, not just theoretical**, by a 3-seed, 3M-step `NUM_AGENTS=4` validation
(2026-08-19). Eval-time numbers look fine in isolation (0-2% collision rate, 100 episodes), but
the training-time rolling-window data (50-episode window, ~1465 logged rollouts per seed — a
much larger sample) tells a different story: 88/1465 (seed1), 59/1465 (seed2), and 269/1465
(seed3) rollouts showed nonzero `collision_rate`, scattered across the *entire* 3M-step run in
every seed, not converging to zero the way `NUM_AGENTS=3` did. Seed3 still showed nonzero
`collision_rate` (0.02-0.04) in its last 10 logged rollouts, through step 3,000,320 — not
resolved by the end of training. `mean_brake_reduction` confirms why: continuous, substantial
engagement throughout the whole run (0.002-0.016) at `N=4`, vs. `N=3`'s sparse, occasional
engagement (0.0001-0.0069) — more simultaneous neighbors means the brake is doing real,
constant work instead of sitting mostly dormant. Separately: the reward gives no *direct*
penalty on brake engagement itself, only an indirect one (`r_safety`'s urgent-zone penalty
shares the brake's trigger threshold, `SAFE_DIST_ENTER`, but is proximity-based, not
action-based — a policy can keep commanding more-than-safe closing speed without anything
specifically penalizing that choice beyond ambient proximity). See `EXPERIMENT_LOG.md` for the
full run data.
**Proposed fixes (not yet implemented, queued in `TODO.md` Phase 1)**: (1) sweep the per-pair
brake correction to convergence — repeat the pass across all neighbors until no further
correction is needed, instead of one sequential pass. Each per-neighbor constraint is a
halfspace in velocity space, and repeated projection onto an intersection of halfspaces
(POCS/alternating-projections) is a standard, provably-convergent technique for exactly this
situation. (2) Add a direct reward penalty proportional to `brake_reduction` (already computed
every step, currently only logged) so the policy is penalized specifically for *needing* the
brake, not just for proximity. Both target the same confirmed mechanism; proposed to test
together.

## 9. Target motion is still constant-velocity, so the vision-tracking dead-reckoning grace period is currently exact, not approximate

**Symptom**: `_target_dir`/`_target_speed` are sampled once in `reset()` and never updated —
the target moves in a straight line at constant speed for the whole episode. This means the
2-second dead-reckoning grace period (`LOST_TIMEOUT_STEPS`, see `config.py`) is currently
mathematically *exact* whenever it's used, not a real approximation of an uncertain estimate.
**Status**: not a bug — a deliberate, explicitly-flagged scope decision (see `DECISIONS.md`'s
vision-tracking entry) made to keep that one change controlled. Documented here so a future
session doesn't mistake "the dead-reckoning grace period isn't really being tested" for "the
mechanism doesn't work" — it works, it just hasn't been tested under real estimate
uncertainty yet.
**Next step**: making the target occasionally change direction/speed mid-episode is the
natural follow-up — it would make the grace period's confidence-decay behavior actually
matter, and is a good candidate for the next controlled, single-variable test in this area.

## 10. UWB-realistic neighbor sensing still deferred

**Symptom**: neighbor observations (`rel_p`, `rel_v`, `d` to locked neighbors in `_get_obs`)
are still exact ground truth, unlike the target, which now goes through the vision-tracking
system. `DECISIONS.md` explains why this was deliberately scoped out of the vision-tracking
change rather than bundled in.
**Status**: not started. Real UWB gives clean range, not clean bearing/velocity — this would
be a comparably-sized design change to the vision-tracking one, deserving its own careful pass
(sensing model, what's observable, how it interacts with the existing mutual-k-NN neighbor
lock graph) rather than a quick patch.

## 11. Sensing noise, occlusion, and a directional FOV cone are all still deferred

**Symptom**: the vision-tracking system's target reading is noiseless whenever a drone is in
range (exact ground truth), has no occlusion model (a drone directly between another drone and
the target doesn't block detection), and detection is omnidirectional (no heading state exists
in this sim to support a directional camera cone).
**Status**: explicitly out of scope for the change that added vision-tracking, not forgotten —
see `DECISIONS.md`. A FOV cone specifically requires adding heading/orientation as a new state
dimension first, which is a bigger prerequisite change than the other two.
