# Current State

As of commit `62a685d` on `phase3-resilience`, 2026-08-20. `main` is still at `b59c139`,
substantially behind — see `TODO.md`. Working tree has a few untracked `deployment/`-side
files (not this doc suite's concern — see `ARCHITECTURE.md`).

## Confirmed working (verified against completed, multi-seed-replicated Kaggle runs)

- **Collision avoidance is now solid at `NUM_AGENTS=4`, not just `N=3`.** The previous headline
  open problem — the closing-speed brake's single-pass correction leaving a real gap under
  simultaneous multi-threat braking — is resolved: multi-pass POCS-style convergence plus a
  thresholded brake-engagement reward penalty. The Phase2-combined 5-seed validation showed
  3/3 `N=4` seeds clean at 0% `collision_rate`, both at eval-time and in the training-time
  rolling-window picture that previously told a much worse story than eval-time numbers alone.
  See `EXPERIMENT_LOG.md`/`DECISIONS.md`.
- **3D (XYZ) formation spread**, replacing the old horizontal-only `r_spread` — correctly
  targets the local, drone-viewpoint ideal angle (60°, analytically proven, verified by
  `test_geometry.py`) after an initial implementation conflated it with a different, global
  quantity. An isolated re-test after the fix found no measurable difference from the buggy
  version in that one test, but the fix was correct to make regardless. See `KNOWN_ISSUES.md`
  item 5, `DECISIONS.md`.
- **N-aware safety margin** — `EDGE_TARGET` scales with simultaneous-neighbor count, validated
  as part of the same combined sweep with no tracking regression.
- **Vision-based cooperative target tracking, at `NUM_AGENTS=3` and `4`.** Unchanged in
  mechanism from the last major pass — sensor-range-limited detection, swarm-shared contact,
  dead-reckoning, `target_lost` termination. Clean at both agent counts under the
  Phase2-combined validation (0-5% `target_lost_rate` at `N=4`, comparable at `N=3`).
- **Ground awareness (Phase 3, new).** A deterministic ground clamp (mirrors the brake's
  pattern, single-pass since the ground doesn't move) plus a graduated `r_ground` reward and
  `ground_strike` termination. **Worked cleanly from the very first test** — `ground_strike_rate`
  was 0/209 rollouts in every one of the 3 Phase 3 validation seeds, no iteration needed
  (unlike the brake, which needed a second pass at the fix).
  A real spawn-safety bug (naive Z-clamping before overlap resolution could collapse
  inter-agent vertical separation) was found and fixed during smoke-testing, before it ever
  reached Kaggle — see `DECISIONS.md`.
- **Per-axis (Z vs XY) action scaling** and a **larger network** (Actor hidden 64→128, Critic
  128→256) — both part of the Phase 3 bundle, no issues found specifically attributable to
  either in isolation (the bundle's one serious failure, `target_lost`, is attributed to a
  different cause — see "Known broken" below).
- **Shared-actor CTDE-PPO training loop**: unchanged in core structure. `best_score`'s
  lexicographic checkpoint-selection criterion now has a 4th tier (`ground_strike_rate`),
  same reasoning each time a new real failure mode was added.
- **Curriculum infra**: `NUM_AGENTS`, `SEED`/`RUN_ID`, `TOTAL_STEPS`, and now `MAX_STEPS`/
  `LOST_TIMEOUT_SEC` env vars all work as designed (`ROLLOUT_LEN` now derives from `MAX_STEPS`
  automatically rather than needing independent tuning).
- **Evaluation and diagnostics**: `evaluate.py` now reports `ground_strike` alongside
  `collided`/`target_lost`. A new tool, `training/diagnose_horizon.py`, runs a frozen
  checkpoint well past its training horizon without stopping on first failure — this is what
  originally surfaced the tracking-degradation evidence that motivated the whole Phase 3
  bundle (see below).

## Known broken / open problems

**Headline: Phase 3's longer episodes broke target tracking almost completely, root cause
diagnosed, fix in progress.** The Phase 3 bundle (longer episodes, larger network, ground
awareness, per-axis dynamics, dynamic target motion) was tested together as an explicit,
user-directed "leap of faith." Ground awareness and per-axis dynamics succeeded cleanly.
Collision stayed at ~0% (unaffected, as expected — an independent mechanism). But
`target_lost_rate` was 89-100% across all 3 seeds, flat from the very first logged rollout
through the full 3M-step run, never once improving. **Diagnosed, not just observed**:
`LOST_TIMEOUT_SEC` (a fixed 2.0s dead-reckoning grace period) was never rescaled when
`MAX_STEPS` grew 9x (200→1800) — an ordinary, recoverable contact gap now has ~9x more
independent opportunities per episode to exceed a still-2-second window. See
`KNOWN_ISSUES.md` item 12 and `EXPERIMENT_LOG.md` for the full reasoning and data.
**Fix in progress, not yet confirmed**: `LOST_TIMEOUT_SEC` is now env-overridable; a 5-kernel
sweep (6s x2, 10s x2, 18s x1) is running. 4 of 5 kernels confirmed `RUNNING`; the 5th is
blocked by an unrelated Kaggle platform issue (see `KNOWN_ISSUES.md` item 15) — likely to
resolve on its own or need the user to check Kaggle's web UI directly.

**Secondary, same branch: vertical jiggling — fixed in code, not yet re-verified.** A trained
Phase 3 checkpoint showed 8-25% of steps per agent flipping vertical-velocity sign (measured
directly, not assumed, after a user report). A cruise-altitude reward preference plus
deterministic Z-velocity smoothing were implemented and smoke-tested, bundled into the same
branch/sweep under explicit time pressure. **Nobody has yet loaded a post-fix checkpoint and
re-measured the sign-flip rate** — don't treat this as confirmed-fixed until that happens. See
`KNOWN_ISSUES.md` item 14.

**Brake's relative-vs-absolute-velocity gap, now concretely relevant.** `_apply_brake` assumes
each agent runs the identical symmetric correction formula — an assumption `deployment/
inference_node.py` breaks by calling it against real PX4/Gazebo telemetry, where the other
vehicle's actual behavior isn't controlled by this code. Not yet reformulated. See
`KNOWN_ISSUES.md` item 13.

**`main` is far behind.** `phase2-combined` (the fully-validated reference point) has not been
fast-forwarded to `main` — needs the user to run the push themselves (permission-classifier
blocked from the agent). See `TODO.md`.

## Unverified (implemented, not yet confirmed by a completed run)

- The `LOST_TIMEOUT_SEC` sweep and the altitude/jiggling fix (both above — this is the main
  unverified surface right now).
- UWB-realistic neighbor sensing, sensing noise/occlusion, a directional FOV cone — all still
  deliberately deferred, unchanged from before. See `KNOWN_ISSUES.md` items 10-11.

## Known-stale documentation

- **`readme.txt`** — more stale than ever, unchanged issue across several doc passes now. See
  `KNOWN_ISSUES.md` item 2.
- **`envs/formation_env.py`'s module docstring** still has the old "4-agent study, not
  general-N" SCOPE paragraph, contradicted by `_PACKING_RATIO`'s actual `{2,3,4}` support.
  Flagged repeatedly, still not fixed, still low priority/cosmetic.

## Data present locally (this machine, gitignored, not git-tracked)

`stage/logs/` and `stage/models/` contain timestamped, commit-tagged archives of Kaggle runs.
Filenames follow `<original_name>_<timestamp>_<short commit tag>.<ext>` — check the tag before
comparing across runs, given how many distinct configurations (pre-brake, brake-only,
vision-tracking with/without floor, Phase 1, Phase 1b, N-aware-margin, xyz-spread,
Phase2-combined, Phase 3) now exist in this project's history. Phase2-combined's best `N=4`/
`N=3`/`N=2` checkpoints are saved as `actor_best_{1,2,3}_20260820-1255_3f9069c-n4-5m-phase2-
seed{1,2,3}.pt`, `actor_best_n3_1_...`, `actor_best_n2_1_...`.

## Local environment caveat

Unchanged: this sandbox's CUDA is unavailable, `DEVICE` always falls back to `cpu` here,
matching the project's measured-optimal default anyway. Kaggle API access from this machine
(`~/.kaggle-venv`) continues to work, with two gotchas worth knowing before touching it: the
Kaggle account username is `rayensboui` (not `rayensb`, the GitHub org name — easy to
transpose), and the account-wide `kernels list` command's default sort order is not reliable
for determining what's currently running — check specific kernels by name via `kernels
status` instead. See `ENVIRONMENT.md` for both in detail.
