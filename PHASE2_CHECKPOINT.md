# Phase 2 checkpoint — active reference point, 2026-08-20

**If you are a different session or agent working in this shared
worktree: this is live, in-progress, partially-validated work.** Do not
modify `envs/formation_env.py`, `config.py`, `training/*.py`, merge or
rebase `phase2-combined`/`xyz-spread-fixed`/`phase3-resilience`, or
launch/cancel Kaggle runs under the `stage-marl-phase2-*`,
`stage-marl-spread-fixed-*`, or `stage-marl-phase3-*` names without
checking with the user first. Silent changes here can invalidate
in-flight Kaggle runs or make it impossible to tell which results map to
which code state. The deployment work in `deployment/` is a separate,
independent track (see `deployment/docs/PHASE2_HANDOFF.md`) and is fine to
keep working on -- this note is about the training/env code specifically.

**Update 2026-08-20 (later)**: `phase3-resilience` (branch, off
`phase2-combined` `8b724bb`) is now also active -- see "Phase 3" section
below. `phase2-combined` itself is unchanged and still the validated
reference point; Phase 3 is a separate, not-yet-validated leap-of-faith
bundle on top of it.

**Update 2026-08-21 (important, read if anything about `target_lost`
results looks inconsistent)**: commits `13b9e38` (active search) and
`b15f391` (`DISABLE_TARGET_LOST_TERMINATION` ablation) were made locally
but **not pushed to origin** until well after several Kaggle runs claiming
to test them had already launched and completed. Those runs (the first
`stage-marl-active-search-seed{1,2,3}`, `stage-marl-search-t2-seed1`,
`stage-marl-search-t18-seed1`, and the first `stage-marl-ablation-noterm-
seed{1,2,3}`, all "version 1") actually cloned whatever was on origin at
the time (`62a685d`) -- i.e. plain `LOST_TIMEOUT_SEC=6` with **no active
search and no ablation flag at all**. Their eval CSVs confirm this
directly (old 10-column schema, no `contact_fraction`/etc., and seed-for-
seed numbers matching the pre-active-search 6s baseline almost exactly).
`origin/phase3-resilience` is now up to date (`b15f391`, pushed). Relaunched
correctly as "version 2": `stage-marl-ablation-noterm-seed{1,2,3}` and
`stage-marl-active-search-seed{1,2}` (2 of 3, Kaggle's 5-session cap only
had 2 free slots after the 3 ablation seeds). **Always confirm
`git status` shows "up to date with origin" immediately before pushing any
new Kaggle kernel from this worktree** -- this is now a standing
pre-launch check, not just a one-off fix.

## What this reference point is

Four combined fixes for the N=4 collision problem, validated:

1. Multi-pass brake convergence (POCS-style repeated projection).
2. Thresholded brake-engagement reward penalty.
3. N-aware safety margin (`EDGE_TARGET` scales with simultaneous-neighbor
   count).
4. XYZ (true 3D) formation spread -- **note**: the 5-seed validation run
   below used the *pre-fix* angle (109.47/120 deg, wrong). The angle was
   corrected to 60 deg (commit `ac98d67`) after that run had already
   launched. `xyz-spread-fixed` (branch, isolated from the other three
   fixes) is re-testing the corrected version now -- see "Currently
   running" below.

Full code + rationale: https://claude.ai/code/artifact/5704e9f9-1e81-4752-92ad-e959a034c954

**Validated**: 5-seed Kaggle run, 5M steps, `phase2-combined` branch --
3x N=4 (all clean: 0% collision, best seed's `tracking_rmse` 1.43, the
best of the project), 1x N=3 (clean collision, some `target_lost` noise in
one seed -- second seed running now to check), 1x N=2 (clean). Best
checkpoints saved in `models/` as `actor_best_{1,2,3}_20260820-1255_
3f9069c-n4-5m-phase2-seed{1,2,3}.pt`, `actor_best_n3_1_...`,
`actor_best_n2_1_...`.

**New finding (2026-08-20, `training/diagnose_horizon.py`)**: training
episodes are 10 simulated seconds (`MAX_STEPS=200`, by design -- see
`DECISIONS.md`, this is correct for how PPO learns and should not be
changed casually). Running the best N=4 checkpoint for 60 simulated
seconds instead (frozen policy, no retraining, doesn't terminate on the
first failure so degradation past the training horizon is actually
visible) shows tracking error climbing from 0.95 (0-10s, matches
training) to a peak of 3.66 (20-30s) before recovering to 1.5-1.8
(40-60s). No collision or target_lost the whole time -- the abstract sim
has no ground/physics failure mode, so it can't show a crash -- but that
20-30s worst window sits *directly inside* the 15-40s window the real
PX4/Gazebo deployment actually crashed in (see
`deployment/docs/PHASE2_HANDOFF.md`). This is real, load-bearing evidence
for a training-horizon-vs-real-flight-duration mismatch as (at least part
of) the deployment crash's cause, not just a plausible story. Worth
reading by whoever picks up the deployment crash investigation next.

**Currently running on Kaggle** (started 2026-08-20, ~1.5M steps each,
smaller/faster than the validation run above):
- `stage-marl-spread-fixed-seed1` (branch `xyz-spread-fixed`, N=4) --
  isolated test of the corrected 60 deg spread angle, to see whether the
  pre-fix bug actually cost anything.
- `stage-marl-phase2-n3-seed2` (branch `phase2-combined`, N=3) -- second
  seed, checking whether seed1's `target_lost` noise was real or
  seed-variance.

**Not yet done**: `phase2-combined` needs a manual
`git push origin phase2-combined:main` (blocked by the permission
classifier on a direct push from the agent, verified as a clean
fast-forward, no conflicts) -- `main` is still behind everything from
tonight until that's run.

**Brake instrumentation** (commit `ad4c483`): `_apply_brake` now exposes
`brake_passes`/`brake_violation`/`k_active` diagnostics (see its
docstring), threaded to 4 new `training_log_*.csv` columns
(`mean_brake_passes`, `max_brake_violation`, `mean_brake_solo`,
`mean_brake_multi`). Pure logging addition, doesn't change training
behavior -- safe to build on without re-validating.

## Phase 3: sustained-flight resilience (branch `phase3-resilience`, 2026-08-20)

Direct response to the horizon finding above. User's explicit call: bundle
five changes into one "leap of faith" test rather than isolate each first,
with an agreed fallback (isolate individually if the combined result isn't
good). Full detail: commit `85f0615` on `phase3-resilience`, and the plan
at the time (`/home/rayen/.claude/plans/abundant-plotting-crayon.md` as of
2026-08-20, may be overwritten by a later plan since).

1. `MAX_STEPS` 200 -> 1800 (90s), env-overridable. `ROLLOUT_LEN` now
   derived (`MAX_STEPS * ROLLOUT_EPISODES`), not independently hardcoded.
   Known risk: ~9x fewer episodes for the same 3M-step budget.
2. Actor hidden 64->128, Critic 128->256.
3. Ground awareness: `_apply_ground_clamp` (mirrors `_apply_brake`),
   `r_ground` reward, `ground_strike` termination + rate, now a 4th
   best-checkpoint selection criterion. A real bug was found and fixed
   while smoke-testing this -- see the commit message for the joint
   ground/overlap spawn-safety fix in `reset()`.
4. Per-axis `MAX_ACTION_SPEED_Z` (0.6x horizontal).
5. Dynamic target motion: redirects every `TARGET_REDIRECT_INTERVAL_STEPS`
   instead of one fixed direction for the whole episode (closes
   `KNOWN_ISSUES.md` item 9).

**Currently running on Kaggle** (started 2026-08-20, N=4, 3M steps,
branch `phase3-resilience`) -- note the actual slugs Kaggle assigned
differ from what was requested (title-derived, not the requested id):
`stage-marl-phase3-resilience-seed1`, `-seed2`, `-seed3`.

## Update 2026-08-20 (later still): altitude fix bundled in, docs refreshed, timeout sweep in progress

`phase3-resilience` (commit `62a685d`) now also includes a cruise-altitude reward preference
(`CRUISE_ALT_MIN`/`CRUISE_ALT_COEF`) and deterministic Z-velocity smoothing
(`Z_SMOOTHING_ALPHA`), fixing a confirmed vertical-jiggling issue (8-25% of steps per agent
flipping vz sign on a trained checkpoint) — bundled with the `LOST_TIMEOUT_SEC` fix below
under time pressure, not yet re-verified against a post-fix checkpoint.

`LOST_TIMEOUT_SEC` is now env-overridable (was a flat 2.0) — Phase 3's `target_lost_rate`
came back catastrophic (89-100%, all 3 seeds, flat from the first rollout) because this never
scaled when `MAX_STEPS` grew 9x. A 5-kernel sweep (6s x2 seeds, 10s x2 seeds, 18s x1 seed) is
running: `stage-marl-timeout6-seed1`, `-timeout6-seed2`, `-timeout10-seed1`, `-timeout10-seed2`,
`-timeout18-seed1` — **all 5 confirmed `RUNNING` as of 2026-08-21**. The 18s kernel was
blocked for ~4-4.5 hours by Kaggle's "Maximum batch CPU session count of 5" (persisted even
with only 4 kernels confirmed running and everything else on the account confirmed complete —
likely an orphaned session from an earlier interrupted push, invisible to `kernels status`);
it cleared on its own, no manual Kaggle UI intervention needed. See
`docs/ai_context/KNOWN_ISSUES.md` item 15 for the full record. A background watcher is polling
all 5 for completion/failure and will pull results down as each finishes.

`docs/ai_context/` (all 11 files) was fully rewritten this pass to reflect everything since
the last update: Phase 1/1b (brake fixes), the N-aware-margin/xyz-spread hypotheses and the
geometry-angle bug caught along the way, the Phase2-combined validation, Phase 3's bundle and
its `target_lost` failure, and the in-progress sweep above. Also corrected a stale claim in
`ARCHITECTURE.md`/`AI_CONTEXT.md`/`ENVIRONMENT.md` that the repo had no ROS2/Gazebo/PX4
integration — `deployment/` exists now (this thread's work) and is cross-referenced from
those files, though its own documentation stays in `deployment/docs/` as before.

**Kaggle account username, for whoever launches the next kernel**: `rayensboui`, not
`rayensb` (the GitHub org name) — easy to transpose, produces a permission-denied error that
reads like a slug typo rather than an owner typo.

## Update 2026-08-21/22: active search validated (properly, second try) — real win; no-termination ablation partial; timeout-bracket test running

Both the 3-seed no-termination ablation and the first active-search Kaggle batch (v1) were
re-run after discovering `13b9e38`/`b15f391` had never been pushed (see the git-push note
above this section) — v1 of both silently ran pre-active-search `62a685d` code. Corrected
versions (v2) are the results below.

**No-termination ablation (`DISABLE_TARGET_LOST_TERMINATION=1`, 3 seeds, correct code)**:
`contact_fraction` improved consistently across all 3 seeds (31-48%, up from ~16-21%
pre-ablation) — a real, reproducible effect. But `target_lost_rate` under the real 6s
timeout stayed catastrophic (79-100%) and **collision safety, previously ~0%, became
seed-dependent and unstable (0-21%)** — likely the brake's known two-agent-at-a-time gap
being stressed harder by longer average episodes + more divergent per-agent motion while
searching. Net: a genuine partial improvement that doesn't reach the finish line and adds a
new cost. Not adopted as-is.

**Active search, properly tested (2 seeds, correct code, 6s timeout)**: `contact_fraction`
85.0-90.7% (final), collision 0-1%, target_lost still 60-81% -- but **median loss-event
length was exactly 121 steps in both seeds, i.e. 1 step past the 120-step (6s) cutoff**.
Read: search is working -- the median failure is a near-miss on timing, not a
never-finds-it failure (contrast the ablation's median loss streak of ~900-1150 steps).
This reverses the earlier (invalid-code) conclusion that active search doesn't help --
it's the best single result of the whole investigation once actually tested.

**Currently running** (5 kernels, `phase3-resilience` @ `ddac78e`, active search + longer
timeout, bracketing the fix suggested by the near-miss finding above): `stage-marl-search-
t8-s{1,2}` (8s timeout, 2 seeds), `stage-marl-search-t10-s{1,2,3}` (10s timeout, 3 seeds).

## Open, not yet decided

- "Genetic/evolutionary" training as an alternative to PPO -- raised,
  not adopted; PPO is what's producing every result above.
- GPU was already measured slower than CPU for this network size
  (`DECISIONS.md`) -- not revisit unless the network architecture changes
  enough to change that math (e.g. the permutation-invariant encoder idea
  queued elsewhere). Phase 3's larger network doesn't change this
  conclusion on its own -- still small by GPU-dispatch-overhead standards.
