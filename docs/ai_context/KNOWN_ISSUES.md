# Known Issues

Updated 2026-08-22 against commit `94216fd`/`3eae55b` on `phase3-resilience` (not yet merged to
`main` — see `TODO.md`). Since the last pass (`62a685d`): item 12 (`target_lost_rate`) is
substantially better but not fully resolved (active search); item 13 (brake relative velocity)
is now implemented and adversarially verified, pending Kaggle-scale confirmation; two new items
(16, 17) were found and resolved this pass. For issues that were investigated and resolved, see
`EXPERIMENT_LOG.md` (they're experiments with a conclusion, not open issues) — items below are
only marked `[RESOLVED]` inline rather than removed, so the numbering stays stable and the
resolution history is visible in place.

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

## 5. [RESOLVED] `r_spread` formation-spread metric is horizontal-only

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
**Resolved 2026-08-19/20**: replaced with true pairwise 3D angular separation between locked
neighbors' direction vectors, penalizing the minimum pairwise angle's deviation from
`_IDEAL_NEIGHBOR_ANGLE`. **The first implementation shipped with a real bug**: it reused
`_PACKING_RATIO`'s angle (109.47°/120°, the *global*, target-viewpoint angle between two
drones as seen from the target/center) where the correct quantity is the *local*,
drone-viewpoint angle between two neighbors as seen from one drone — a different geometry
entirely, proven to be exactly 60° for any neighbor count under full connectivity (regular-
simplex construction, see `DECISIONS.md`). Caught by supervisor review before the bug's
practical cost was ever measured; a standalone re-test with a single seed
(`xyz-spread-fixed`/`spread_fixed_seed1`) after the fix found no obvious measurable harm from
the original wrong angle in that one test, but the fix was correct to make regardless — the
*reasoning* was wrong even where a specific outcome happened not to visibly suffer.
`test_geometry.py` was added specifically as the regression check that would have caught this
before it shipped, and now guards against it recurring. See `DECISIONS.md` and
`EXPERIMENT_LOG.md` for the full timeline and numbers.

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

## 8. [RESOLVED] Closing-speed brake's no-crossing property failed under simultaneous multi-threat braking at `NUM_AGENTS=4`

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
**Resolved (Phase 1/1b, 2026-08-19)**: both proposed fixes were implemented and tested
together, then refined once. (1) The per-pair brake correction now sweeps to convergence
(POCS-style repeated projection, up to `NUM_AGENTS` passes) instead of one sequential pass —
see `DECISIONS.md` for the precise "limit-convergent, not N-pass-exact" claim and the
adversarial stress-test verification. (2) A direct reward penalty on `brake_reduction` was
added, first linear-from-zero (v1), then refined to only tax the excess above a derived
threshold (v2) after v1 measurably fixed collision but pushed the policy toward an overly wide
formation (see `DECISIONS.md`). **Confirmed fixed, not just theoretically addressed**: the
Phase2-combined 5-seed validation (`EXPERIMENT_LOG.md`) showed all 3 `N=4` seeds clean at 0%
`collision_rate`, both at eval-time and in the training-time rolling-window picture that
previously told a much worse story than eval-time numbers alone. **Residual, not fully closed**:
the no-crossing proof still assumes each agent's own velocity toward another, not relative
velocity — see item 13 below, which is the same underlying gap restated with its current,
more concrete (deployment-relevant) stakes.

## 9. [RESOLVED] Target motion was constant-velocity, so the vision-tracking dead-reckoning grace period was mathematically exact, not approximate

**Symptom (historical)**: `_target_dir`/`_target_speed` were sampled once in `reset()` and
never updated — the target moved in a straight line at constant speed for the whole episode,
making the dead-reckoning grace period mathematically exact whenever used, not a real
approximation of an uncertain estimate.
**Resolved (Phase 3, 2026-08-20)**: `step()` now calls `_resample_target_motion()` (same
distribution `reset()` uses) every `TARGET_REDIRECT_INTERVAL_STEPS` (500 steps = 25s) — a
90-second episode now sees ~3 mid-episode redirects. Dead-reckoning is now a genuine
approximation that can actually mislead a drone while it's out of contact, the realistic case
this was always meant to be tested against.
**What this resolution surfaced, not fixed by it**: making the grace period a real
approximation (rather than removing the constant-velocity assumption's *masking effect* on a
separate, pre-existing problem) is very likely part of why Phase 3's `target_lost_rate` result
was so severe — see item 12 below, which is the current, actually-open problem in this area.
Item 9 itself (motion realism) is resolved; item 12 (the grace period's *length* at the new
episode duration) is not the same claim and is still open.

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

## 12. [PARTIALLY RESOLVED] `target_lost_rate` near-total failure at `MAX_STEPS=1800`

**Symptom**: Phase 3's 3-seed `N=4` validation (2026-08-20) showed `target_lost_rate` 89-100%,
flat from the very first logged rollout through the full 3M-step run in every seed — never
once improving. Collision and ground-strike were both essentially unaffected by the same
bundle (0-1% and literally 0/209, respectively), so this is specific to target tracking, not a
general regression from the bundle.
**Diagnosis (reasoned from evidence, not assumed)**: `LOST_TIMEOUT_SEC` (a fixed 2.0s grace
period, `LOST_TIMEOUT_STEPS=40`) was never rescaled when Phase 3 raised `MAX_STEPS` from 200
to 1800 (9x). An ordinary, recoverable contact gap now has ~9x more independent windows per
episode in which to exceed a still-2-second grace period, with zero change in the swarm's
actual moment-to-moment tracking competence. The flat-from-step-1 shape (rather than a
mid-training regression) and collision's clean convergence despite being subject to the same
"9x fewer episodes per fixed step budget" risk both argue against pure data-starvation as the
explanation, and point specifically at the timeout/episode-length mismatch. See
`EXPERIMENT_LOG.md`'s Phase 3 entry and `DECISIONS.md` for the full reasoning.
**Sweep result**: not a clean dose-response — 3 of 5 completed runs stayed catastrophically
broken (85-96% `target_lost_rate`) regardless of timeout length; identical 10s configuration
produced both the worst result (93%) and one of the two working ones (4%) across its two seeds.
Grace-period length alone was never the primary lever. `LOST_TIMEOUT_SEC`'s default was raised
to 6.0 (not all the way to 10/18, given the lack of a clean signal).
**Superseded by active search, a much bigger lever**: rather than continuing to sweep the
timeout constant, active search (each agent fans out in its own fixed heading on loss of
contact, instead of every agent coasting toward the same stale point) was built and validated.
`contact_fraction` improved from ~16-21% to 85-90.7% at the 6s timeout, with the median loss
event landing just 1 step past the cutoff — a genuine, large improvement, though
`target_lost_rate` (60-81%) is still not solved. A follow-up bracket (active search + 8s/10s
timeout) found the mechanism *can* reach excellent tracking (99%+ contact, ~2-4%
`target_lost_rate`) when a seed converges well, but training is seed-sensitive — 2 of 3 `t10`
seeds got stuck in a much worse regime despite identical configuration. Resuming those two
stuck seeds to 5M steps confirmed "more training helps" (70-72%→41% and 92-96%→83%
`target_lost_rate`) but didn't fully close the gap within that budget.
**New finding, not previously known**: when a seed *does* converge to the tight, confident
tracking behavior active search enables, deterministic execution exposes a real collision-safety
cost (8-14%) not seen at this magnitude since the original `N=4` brake gap — see item 13 below
and the train-vs-eval discrepancy item further down. This is now the more actionable finding
than `target_lost_rate` itself.
**Status**: partially resolved. Active search is a real, adopted improvement; `target_lost_rate`
is meaningfully better than the Phase 3 catastrophe (89-100%) but not fully solved even at good
seeds' best case (~2-4%). Not under further active investigation this pass — the currently
higher-priority, more tractable follow-up is the collision-safety cost this same convergence
exposes (item 13). See `EXPERIMENT_LOG.md` for full data.

## 13. [RESOLVED, pending Kaggle-scale confirmation] Closing-speed brake assumed each agent's own velocity, not relative velocity

**Symptom**: `_apply_brake`'s `v_closing` is `dot(v_a, dir_to_b)` — agent A's own velocity
component toward B — not the relative closing velocity `(v_a - v_b)`. The brake's adversarial
stress-test verification (`DECISIONS.md`) holds specifically because every agent in this sim
runs through the identical symmetric formula every step, with no way for one side to be
unconstrained while the other closes at speed.
**Why this matters more now than when first noted**: this was originally flagged as a
theoretical footnote. It's now concretely relevant — `deployment/inference_node.py` calls
`_apply_brake` directly against real PX4/Gazebo vehicle telemetry (see `ARCHITECTURE.md`), and
a real or simulated vehicle's actual velocity is not something this code controls or can
assume is running the same symmetric correction on the same schedule. The symmetry assumption
the stress tests relied on doesn't automatically hold in that setting.
**Why this became urgent, not just theoretical, right when it was fixed**: the
deterministic-vs-stochastic collision investigation (see the new item below) found real
collisions under exactly the symmetric, no-noise conditions this formula's own reasoning said
should be impossible — direct evidence the gap was being exercised, not just a hypothetical
concern about external vehicles.
**Resolved 2026-08-22**: `v_closing` reformulated to `dot(v_a - v_b, dir_to_b)` (the true mutual
closing rate) in both the multi-pass correction loop and the post-hoc violation diagnostic —
see `DECISIONS.md`. Re-verified against both adversarial scenarios (now a committed regression
test, `test_brake_relative_velocity.py`, not just a one-off manual claim) — still zero residual
violation. **Not yet confirmed to actually reduce `collision_rate` at training scale** — a
3-seed/3M-step Kaggle validation is in progress. Don't cite this as fully closed until that
completes; see `EXPERIMENT_LOG.md`/`SESSION_HANDOFF.md`.

## 14. Vertical jiggling — fixed, but only smoke-tested, not yet re-validated against a trained checkpoint

**Symptom**: a trained Phase 3 checkpoint, run deterministically for 400 steps, showed 8-25%
of steps per agent flipping vertical-velocity sign, one agent briefly dipping to `z=0.98` —
confirmed empirically (not just accepted on the user's report) before any fix was written.
**Fix implemented**: `CRUISE_ALT_MIN=1.5`/`CRUISE_ALT_COEF=-10.0` (a graduated `r_altitude`
reward preference, no hard clamp) plus `Z_SMOOTHING_ALPHA=0.3` (a deterministic low-pass
filter on commanded Z velocity, applied before the brake/ground clamp so those retain final
say) — see `DECISIONS.md`.
**Status**: implemented, smoke-tested (imports clean, `test_env.py` passes, episodes run to
completion), bundled into the same branch and Kaggle sweep as the `LOST_TIMEOUT_SEC` fix
(item 12) under explicit time pressure. **Not yet re-measured against an actual trained
checkpoint** — the 8-25% sign-flip figure above is the *pre-fix* baseline; whether the fix
actually reduces it by a meaningful amount is an open question until the current sweep's
checkpoints can be loaded and measured the same way. Don't cite this as "fixed and confirmed"
until that check happens — see `TODO.md`.

## 15. [RESOLVED] Kaggle "Maximum batch CPU session count of 5" blocked a launch even with only 4 kernels confirmed running

**Symptom**: pushing the 5th kernel in the `LOST_TIMEOUT_SEC` sweep (`stage-marl-timeout18-
seed1`) failed with Kaggle's 5-concurrent-session cap error, even after individually
confirming (via `kaggle kernels status` on each kernel by its correct slug) that only 4
kernels were `RUNNING` and every other kernel on the account was `COMPLETE`.
**Investigation so far**: the account-wide `kaggle kernels list` command's `lastRunTime`
sort order turned out to be unreliable for determining "what's running right now" (kernels
pushed hours later appeared far down the default listing, behind kernels pushed earlier the
same day) — don't trust it for this purpose; check specific kernels' status individually
instead. No Chrome browser was connected in this session to inspect Kaggle's web UI directly
(the authoritative source for currently-running sessions), and the `kaggle` CLI has no
"list active sessions" or "stop session" command independent of a specific kernel ref.
**Leading hypothesis, not confirmed**: an orphaned session tied to an old, interrupted version
of one of the kernels (the initial `timeout6-seed1` push was interrupted mid-flight earlier in
this session, before being re-pushed with the altitude fix included) — `kernels status`
appears to report only the latest version's session, so an old version's session could still
be silently consuming a slot.
**Resolved 2026-08-21**: cleared on its own after roughly 4-4.5 hours of the 4 legitimate
kernels running (no manual intervention on Kaggle's web UI was used) — consistent with the
orphaned-session hypothesis: whatever stale session was holding the 5th slot most likely hit
its own time limit and was force-terminated by Kaggle. `stage-marl-timeout18-seed1` pushed
successfully and confirmed `RUNNING` once the slot freed. If this recurs on a future kernel
batch, the same wait-it-out approach worked here — retry the push periodically rather than
assuming something is permanently broken, though checking Kaggle's web UI directly would
likely resolve it faster if it happens again.

## 16. [RESOLVED] Train-vs-eval collision-rate discrepancy

**Symptom**: deterministic eval `collision_rate` ran consistently higher than the trailing
training-time (stochastic) rate across many runs project-wide — most starkly `search-t8-s1`/
`s2`, train ~1-1.2% vs eval 8-14%. Worth documenting explicitly since a future session seeing
this pattern again might otherwise suspect an eval bug rather than recognize a known, explained
effect.
**Investigated, not assumed**: running the identical frozen checkpoint deterministically vs.
stochastically on identical eval seeds showed determinism alone explains most/all of the gap for
a well-converged policy — a policy shaped by entropy-driven exploration can converge its *mean*
action to a knife-edge equilibrium riskier than the noise-perturbed behavior actually seen
during training. For a less-converged case, a residual gap remained between two stochastic
numbers, most likely a small-sample artifact in training's 50-episode rolling window rather than
a real difference in policy quality.
**Status**: resolved/explained — see `DECISIONS.md`'s "Trusting deterministic eval..." entry and
`EXPERIMENT_LOG.md` for the full investigation. **Practical rule going forward**: judge safety
claims from deterministic eval numbers, not training-time rolling numbers.

## 17. [RESOLVED] Unpushed local commits silently tested by ~8 Kaggle kernels

**Symptom**: two feature commits (active search, the `DISABLE_TARGET_LOST_TERMINATION` ablation
flag) were made locally but not pushed to `origin/phase3-resilience` before several Kaggle
kernels claiming to validate them had already launched and completed — those runs actually
cloned whatever was on `origin` at the time, silently testing stale code.
**Found via**: a missing-columns anomaly in the downloaded eval CSVs (the new features' expected
output fields weren't present), not assumed or guessed.
**Status**: resolved — corrected runs relaunched as "v2." **Standing rule adopted**: confirm
`git status` shows up to date with `origin` immediately before every Kaggle launch from this
worktree, every time, not just after this specific mistake. See `PHASE2_CHECKPOINT.md` for the
full incident record and `ENVIRONMENT.md` for the workflow note.
