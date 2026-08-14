# Known Issues

Open problems as of commit `b14fe48`, 2026-08-14. For issues that were investigated and
resolved, see `EXPERIMENT_LOG.md` (they're experiments with a conclusion, not open issues).

## 1. Latest reward/stability fix chain is unverified end-to-end

**Symptom / status**: commits `a40db9b` through `63274b1` (log_std clamp, per-component
reward logging, reward reweighting, diverge→cohesion replacement, entropy-annealing fix,
and the `63274b1` fix for "cohesion/safety conflict causing collision-rate collapse") landed
as a chain of fixes but, as of the last known state of this project, had **not** been
confirmed against a completed training run.
**What's already been done**: an `NUM_AGENTS=4`, 3-seed Kaggle training run was reportedly
launched specifically to validate this chain.
**What's still needed**: check `SESSION_HANDOFF.md` for whether results have arrived; if so,
analyze `collision_rate`, `r_cohesion`, `r_safety` trends over training and confirm the
collision-rate collapse described in `63274b1`'s own commit message doesn't recur. If
results haven't arrived yet, this is the top-priority thing to follow up on.

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
