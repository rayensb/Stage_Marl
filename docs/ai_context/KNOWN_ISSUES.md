# Known Issues

Open problems as of commit `b14fe48`, 2026-08-14. For issues that were investigated and
resolved, see `EXPERIMENT_LOG.md` (they're experiments with a conclusion, not open issues).

## 1. Collision-rate collapse recurs late in training -- CONFIRMED, root cause identified

**Status update (2026-08-14, this session)**: the `a40db9b`..`63274b1` reward/stability chain
did **not** fix the collapse. Confirmed against a completed `NUM_AGENTS=3`, 3-seed run
(`eval_n3_*.csv` / `eval_best_n3_*.csv` / `training_log_n3_*.csv`, all local in `~/Downloads/`
at the time of this analysis) -- all 3 seeds show the identical pattern: `collision_rate`
drops near 0 by ~step 100-150k, stays low through ~step 220-250k, then climbs back to
**26-46%** by `TOTAL_STEPS=600k` on the final model. `--best` checkpoints (saved mid-run,
during the good window) stay at 2-8% collision_rate, at the cost of ~1.8x worse
`tracking_rmse` and ~25% larger `swarm_diameter` -- confirming the underlying policy the run
finds is fine, it just isn't preserved past the point it started degrading.

**Root cause (verified, not hypothesis)**: `RECOVERY_MAX_TRIGGERS=5` was exhausted in every
seed by rollout ~110-125 (`5 * (RECOVERY_PATIENCE=15 + RECOVERY_COOLDOWN=10)` = 125 rollouts
= ~step 220-250k at `ROLLOUT_LEN=2048`) -- matching exactly where each seed's
`entropy_coefficient` plot stops showing recovery spikes and collision_rate starts climbing.
For the remaining ~60% of the 600k-step run, entropy anneals unprotected toward its floor,
the policy converges harder onto the tight target formation (see `config.py`'s `TARGET_DIST`
comment on the razor-thin safety margin at the mathematical optimum), and safety erodes.
**Fix applied this session**: `RECOVERY_MAX_TRIGGERS` raised 5 -> 20 in `training/train.py`
(see the inline comment there for the full derivation) -- gives ~500 rollouts of recovery
budget, comfortably more than the ~293 rollouts in a full run, so the cap stops binding
before `TOTAL_STEPS` is reached.
**Still needed**: this fix is itself unverified against a completed run -- the next
single-seed `NUM_AGENTS=3` run (before spending a 3-seed `NUM_AGENTS=4` Kaggle batch on it)
should confirm `collision_rate` no longer relapses in the back half of training. The pending
`NUM_AGENTS=4`, 3-seed Kaggle run referenced in earlier versions of this doc was deliberately
**not** launched on the pre-fix code -- see `SESSION_HANDOFF.md`.

## 2. `readme.txt` is stale and describes a removed architecture

**Symptom**: `readme.txt` documents a per-drone-file save scheme
(`models/actor_droneN[_SEED].pt`) that no longer exists — `train.py` saves one shared file.
It's silent on the shared-actor architecture, best-checkpoint tracking, entropy recovery,
`evaluate.py --best`, several env vars (`NUM_AGENTS`, `N_REACT`, `VELOCITY_WEIGHT`, `SEED`,
`DEVICE`), and the more recent training-loop fixes (gradient clipping, per-agent-involved
collision attribution, per-minibatch KL early stop).
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

## 5. `r_spread` formation-spread metric is horizontal-only

**Symptom**: documented directly in the `envs/formation_env.py` module docstring — the
spread reward component only considers horizontal (x/y) spacing, not vertical (z) spread.
**Status**: documented limitation, unresolved. Whether this matters depends on whether the
target formation is expected to have meaningful vertical structure; UNVERIFIED whether this
has caused any observed training artifact (e.g. agents exploiting vertical separation to
avoid the spread penalty while remaining close in 3D).

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
