# Current State

As of commit `b14fe48` on `main`, 2026-08-14. Working tree was clean at last check.

## Confirmed working (verified by reading code / running locally)

- **Environment mechanics**: `FormationEnv3D` resets, steps, computes observations and the
  7-component reward, and terminates on collision / truncates at `MAX_STEPS`. `test_env.py`
  (200 random-action steps) is the smoke test — UNVERIFIED whether it was re-run after the
  most recent commits (`a40db9b`..`63274b1`); re-run it before trusting the env post-fixes
  (`COMMANDS.md` has the exact command).
- **Shared-actor CTDE-PPO training loop**: architecture is implemented as designed —
  single shared `Actor`, multi-head `CentralCritic`, batched rollout inference, pooled actor
  training batch, per-agent GAE with advantage normalization, gradient clipping, per-
  minibatch + per-epoch KL early stopping, LR + entropy annealing, plateau-triggered entropy
  recovery, lexicographic best-checkpoint selection. All of this is implemented and was
  exercised in prior `NUM_AGENTS=2`/`3` runs (see `EXPERIMENT_LOG.md`).
- **Curriculum infra**: `NUM_AGENTS` env var, `SEED`-based `RUN_ID` suffixing (log/
  checkpoint/model filenames), and `_PACKING_RATIO`-derived `TARGET_DIST` all work as
  designed for `NUM_AGENTS ∈ {2, 3, 4}`.
- **Evaluation**: `evaluate.py` loads a shared-actor model file (regular or `--best`) and
  produces a per-episode metrics CSV. Confirmed CSVs exist from prior `NUM_AGENTS=3` eval
  runs (see below).

## Unverified (implemented, not yet confirmed by a completed run)

- **The most recent reward/stability fix chain** (commits `a40db9b` "clamp log_std" through
  `63274b1` "fix cohesion/safety conflict causing collision-rate collapse; log per-step not
  per-episode-sum reward metrics") has **not been confirmed by a completed training run** as
  of the last known state of this project. A `NUM_AGENTS=4`, 3-seed Kaggle run was reported
  as launched to validate this chain; whether it has finished and what it showed is the
  single most important thing to check first in a new session — see `SESSION_HANDOFF.md`.
- Whether the per-step (vs. per-episode-sum) reward-component logging fix in `63274b1`
  changes the *interpretation* of prior `NUM_AGENTS=2`/`3` runs' logged `r_*` columns
  (their raw CSV values may be on the old, episode-length-confounded scale) — UNVERIFIED,
  worth checking before directly comparing old and new runs' component magnitudes.

## Known-stale documentation

- **`readme.txt` is confirmed stale.** It describes a per-drone-file save scheme
  (`models/actor_droneN[_SEED].pt`) that no longer exists in the code — `train.py` now saves
  a single shared file (`models/actor[_<run_id>].pt`, `train.py:402`). It also doesn't
  mention: the shared-actor architecture, best-checkpoint tracking, entropy recovery,
  `evaluate.py --best`, the `NUM_AGENTS`/`N_REACT`/`VELOCITY_WEIGHT`/`SEED`/`DEVICE` env
  vars, or the gradient-clipping/credit-attribution/KL-early-stop fixes. Per the rule this
  documentation task was created under, this was **not** fixed as part of writing these
  context docs — it's flagged here as a known discrepancy for whoever picks this up next.

## Data present locally (outside the repo, not git-tracked)

Evaluation result CSVs for `NUM_AGENTS=3` runs exist in `~/Downloads/` on this machine
(`eval_best_n3_2.csv`, `eval_best_n3_3.csv` seen in this session — 100 episodes each,
deterministic eval of the `--best` checkpoint for two different seeds). Spot pattern from a
skim: collision rate looks low (a handful of `collided=True` rows out of 100 in each file,
episode lengths on those rows well under `MAX_STEPS=200`, i.e. genuine early collisions, not
truncations mislabeled). **This is an informal skim, not an analysis** — do not treat this
paragraph as a substitute for actually loading and computing statistics over these CSVs if
asked to evaluate the `NUM_AGENTS=3` results properly.

## Known broken / open problems

See `KNOWN_ISSUES.md` for the full list with symptoms and investigation history. Headline
item: the `63274b1` commit message itself names the problem it was fixing ("cohesion/safety
conflict causing collision-rate collapse") — confirm via the pending `NUM_AGENTS=4` run
whether that fix actually resolved it, since this hasn't been independently verified since
that commit landed.

## Local environment caveat

This sandbox's CUDA is unavailable (old NVIDIA driver) — `DEVICE` always falls back to
`cpu` here regardless of the env var. This matches the project's actual default anyway (see
`DECISIONS.md`: CPU was measured faster than CUDA on Kaggle for this network size), so it
doesn't block local smoke-testing, but it means no local run can validate GPU-path behavior
if that ever becomes relevant.
