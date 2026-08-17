# Current State

As of commit `b59c139` on `main`, 2026-08-17 (vision-tracking merged). Working tree clean.

## Confirmed working (verified against completed, 3-seed-replicated Kaggle runs)

- **Collision avoidance, at `NUM_AGENTS=3`**: the closing-speed brake
  (`envs/formation_env.py:step()`) makes collision a deterministic, action-space-enforced
  near-non-event rather than something the policy has to learn purely from reward. Verified:
  `collision_rate=0.000` across every seed/duration tested from the point the brake landed
  (6/6 configurations at exactly 0%), and 8/9 in a later 3-seed, 3M-step replication after
  vision-tracking was also added (one isolated exception — see Known limitation below). This
  resolved a collision-rate collapse that six consecutive reward-shape/schedule fixes (commits
  `78e1feb` through `67ccfd7`) had failed to fix — see `EXPERIMENT_LOG.md`/`DECISIONS.md`.
- **Vision-based cooperative target tracking, at `NUM_AGENTS=3`**: replaced ground-truth
  target telemetry with sensor-range-limited detection, swarm-shared contact, dead-reckoning,
  and `target_lost` episode termination (see `ARCHITECTURE.md`). After disabling a
  now-counterproductive diameter floor and training to 3M steps (3 seeds): `target_lost_rate`
  0-3%, `collision_rate` 0-1%, `tracking_rmse` 1.52-2.99 — the best tracking accuracy measured
  all session, better than the pre-vision-tracking ground-truth-telemetry numbers. See
  `EXPERIMENT_LOG.md` for the full progression (31-40% → 18-14% → 11-4% → 0-3%
  `target_lost_rate` across four configurations).
- **Environment mechanics**: `FormationEnv3D` resets, steps, computes observations (now
  `OBS_DIM = 18 + 7*EFFECTIVE_K`) and reward (now 8 components including `r_contact`),
  terminates on collision **or** `target_lost`, truncates at `MAX_STEPS`. `test_env.py` was
  re-run locally after every change this session and confirmed passing.
- **Shared-actor CTDE-PPO training loop**: unchanged in its core structure from before this
  session — single shared `Actor`, multi-head `CentralCritic`, batched rollout inference,
  pooled actor training batch, per-agent GAE with advantage normalization, gradient clipping,
  per-minibatch + per-epoch KL early stopping, LR + entropy annealing, plateau-triggered
  entropy recovery, lexicographic best-checkpoint selection (now 3-level: collision_rate,
  target_lost_rate, track).
- **Curriculum infra**: `NUM_AGENTS` env var, `SEED`-based `RUN_ID` suffixing,
  `_PACKING_RATIO`-derived `TARGET_DIST`, and now `TOTAL_STEPS` env var (added 2026-08-17 so
  longer-training tests don't need a hardcode-then-revert cycle) all work as designed.
- **Evaluation**: `evaluate.py` now reports `target_lost`/`avg_confidence` alongside the
  original metrics; `tracking_rmse` remains deliberately ground-truth-based (see
  `ARCHITECTURE.md`) even though the reward now scores against the tracked estimate.

## Unverified (implemented, not yet confirmed by a completed run)

- **`NUM_AGENTS=4`**. Everything above has only been validated at `NUM_AGENTS=3`. This is the
  clear next step — see `TODO.md`. At `N=4`, `K_NEIGHBORS=NUM_AGENTS-1=3` means every agent
  is still always locked/observed (the fix that mattered specifically at `N=4`, closing an
  observation/reward mismatch — see `DECISIONS.md`), so there's reason to expect this
  transfers, but it hasn't been run.
- **The closing-speed brake's guarantee for simultaneous multi-threat braking.** Proven
  algebraically for two agents at a time; not for an agent braking against two neighbors at
  once. A rare (~1%) collision rate showed up at `N=3` with the brake active over long/
  replicated runs — worth specifically re-checking at `N=4`, where more agents means more
  opportunities for this case. See `KNOWN_ISSUES.md` item 8.
- **UWB-realistic neighbor sensing, sensing noise/occlusion, a directional FOV cone, and
  non-constant-velocity target motion** — all deliberately deferred when vision-tracking was
  built, not attempted. See `KNOWN_ISSUES.md` items 9-11.

## Known-stale documentation

- **`readme.txt` is confirmed stale and has gotten more so.** It describes a per-drone-file
  save scheme that no longer exists, and is silent on essentially everything from this
  session (closing-speed brake, vision-tracking, `target_lost`, the `r_collision` logging
  fix). See `KNOWN_ISSUES.md` item 2.
- **`envs/formation_env.py`'s module docstring has one stale paragraph** (the "SCOPE" note
  claiming `TARGET_DIST` is "not a general-N formula... a 4-agent study") — `config.py`'s
  `_PACKING_RATIO` has been generalized to `NUM_AGENTS ∈ {2,3,4}` since before this session
  started. Flagged, not fixed (out of scope for the changes made this session, which added a
  *new*, accurate paragraph to the same docstring without touching this pre-existing one).

## Data present locally (this machine, gitignored, not git-tracked)

`stage/logs/` and `stage/models/` now contain timestamped, commit-tagged archives of every
Kaggle run from this session (from the original N=3 diagnosis through the final 3-seed,
3M-step vision-tracking validation) — filenames follow `<original_name>_<timestamp>_<short
commit tag>.<ext>`. This is a genuinely useful local record of the whole investigation; check
filenames for which commit/config each result corresponds to before drawing conclusions from
an old file.

## Known broken / open problems

See `KNOWN_ISSUES.md` for the full current list. Headline items as of this writing: `N=4` is
unvalidated (not broken, just untested — the actual next step), and the brake's rare
multi-agent edge case (item 8) is worth watching as agent count increases.

## Local environment caveat

Unchanged from before: this sandbox's CUDA is unavailable (old NVIDIA driver), `DEVICE`
always falls back to `cpu` here, which matches the project's actual measured-optimal default
anyway (see `DECISIONS.md`). This session additionally set up Kaggle API access from this
machine (`~/.kaggle-venv`, a dedicated venv for the `kaggle` CLI) — training runs can now be
launched, polled, and their output downloaded directly from here rather than requiring manual
Kaggle-notebook copy/paste. See `ENVIRONMENT.md`.
