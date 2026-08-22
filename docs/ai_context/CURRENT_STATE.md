# Current State

As of commit `94216fd`/`3eae55b` on `phase3-resilience`, 2026-08-22. `main` is still at
`b59c139`, substantially behind — see `TODO.md`. Working tree has a few untracked
`deployment/`-side files (not this doc suite's concern — see `ARCHITECTURE.md`).

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
  `collided`/`target_lost`, plus (2026-08-21/22) contact/loss-streak tracking, reacquisition
  counts, pooled true-tracking-error percentiles (mean/P95/max, not just RMSE), and a
  reacquisition-*time* distribution (steps/seconds, not just counts). A new tool,
  `training/diagnose_horizon.py`, runs a frozen checkpoint well past its training horizon
  without stopping on first failure — this is what originally surfaced the tracking-degradation
  evidence that motivated the whole Phase 3 bundle (see below).
- **Active search (2026-08-21), validated as a real, major improvement.** When the whole swarm
  loses contact, each agent now fans out in its own fixed heading instead of every agent
  coasting toward the same increasingly stale point. `contact_fraction` improved from ~16-21%
  to 85-90.7% at the original 6s timeout; a follow-up bracket (8s/10s timeout) showed the
  mechanism can reach 99%+ contact / ~2-4% `target_lost_rate` when a seed converges well — see
  "Known broken" below for the two things this surfaced, not fixed by it.
- **Closing-speed brake reformulated with relative velocity (2026-08-22), adversarially
  verified, Kaggle-scale validation in progress.** See "Known broken" below.

## Known broken / open problems

**Headline, now two-part: `target_lost_rate` is much better but not solved, and solving it
exposed a new collision-safety cost.** Active search closed most of Phase 3's `target_lost`
catastrophe (89-100% → 60-81% at the same 6s timeout, and down to ~2-4% for well-converged
seeds at 8-10s) — see `KNOWN_ISSUES.md` item 12 for the full history. But when a seed *does*
converge to that tight, confident tracking behavior, **deterministic execution exposes a real
collision-safety cost (8-14%) not seen at this magnitude since the original `N=4` brake gap.**
Investigated directly (not assumed): the same frozen checkpoint run deterministically vs.
stochastically on identical eval seeds shows determinism itself explains most/all of this —
a policy shaped by exploration noise can converge its mean action to a knife-edge equilibrium
riskier than the noise-perturbed behavior actually seen during training. This is why
deterministic eval numbers, not training-time rolling numbers, should be trusted for safety
claims going forward — see `DECISIONS.md`.

**Fix implemented and adversarially verified, Kaggle-scale confirmation in progress.** The
brake's closing-speed check (`v_closing`) now uses the true mutual closing rate
(`dot(v_a - v_b, dir_to_b)`) instead of each agent's own velocity alone — the old formula's
adversarial-test guarantee only held because every agent ran the identical symmetric formula
with nothing to perturb it, exactly the condition deterministic execution reproduces. Both
adversarial stress tests (now a committed regression test, `test_brake_relative_velocity.py`)
still pass. **Not yet confirmed to reduce `collision_rate` at training scale** — a 3-seed/3M-step
Kaggle validation is running. See `KNOWN_ISSUES.md` item 13, `EXPERIMENT_LOG.md`.

**Two seeds resumed to 5M steps to test "does more training help" — partially yes, not fully.**
The two `t10`-bracket seeds that got stuck in a bad tracking regime were resumed (not retrained)
from their ~3.01M-step checkpoints to 5M steps. Both improved substantially (70-72%→41% and
92-96%→83% `target_lost_rate`) but neither closed the gap to the well-converged seeds' ~2-4%.
More training is a real, contributing factor, not the whole story. See `EXPERIMENT_LOG.md`.

**Secondary: vertical jiggling — fixed in code, still not re-verified.** Unchanged from the
last pass — a cruise-altitude reward preference plus deterministic Z-velocity smoothing were
implemented and smoke-tested after confirming an 8-25% per-step vertical-velocity sign-flip
rate on a trained Phase 3 checkpoint. **Still nobody has loaded a post-fix checkpoint and
re-measured the sign-flip rate** — don't treat this as confirmed-fixed until that happens. See
`KNOWN_ISSUES.md` item 14.

**`main` is far behind.** `phase2-combined` (the fully-validated reference point) has not been
fast-forwarded to `main` — needs the user to run the push themselves (permission-classifier
blocked from the agent). See `TODO.md`.

## Unverified (implemented, not yet confirmed by a completed run)

- The relative-velocity brake fix's actual effect on training-time `collision_rate` (adversarial
  tests pass; Kaggle validation running) — this is the main unverified surface right now.
- The vertical-jiggling fix (still not re-measured against a trained checkpoint, unchanged from
  the last pass).
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
