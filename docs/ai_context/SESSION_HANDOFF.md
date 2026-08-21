# Session Handoff

**Read this file every new session — it's the most likely to be stale, and the most
important for not repeating work or missing what just happened.** Update it whenever a
session ends mid-task or reaches a natural checkpoint.

## Last updated

2026-08-20. Prior entry (2026-08-19) covered the session that ran the first `NUM_AGENTS=4`
3-seed validation and found a mixed result (tracking generalized, collision didn't) — that
problem is now **resolved**. Since then: the two proposed fixes for it were implemented and
validated, two formation-quality hypotheses were tested and merged alongside them, a real
geometry bug was caught by review before it cost anything measurable, the combined result was
validated across `N=2/3/4`, a sustained-flight diagnostic surfaced a *new* problem (tracking
degrades past the training horizon, matching real deployment crash timing), a 5-change bundle
was built and tested to address it (explicit "leap of faith," user-directed), and that bundle's
one serious failure (`target_lost_rate` near 100%) was diagnosed and a fix is now being swept
on Kaggle. This doc update itself — "document all" — is the most recent action.

## What was happening (the short version)

Starting from the `N=4` collision problem confirmed at the end of the last session: two fixes
were implemented together (multi-pass POCS-style brake convergence; a brake-engagement reward
penalty) and validated on `N=4` — collision fixed, but a first, linear-from-zero version of the
penalty (Phase 1) pushed the formation wider and hurt tracking. A refined threshold-based
version (Phase 1b) fixed that side effect without losing the collision fix. In parallel, two
formation-quality hypotheses — an N-aware safety margin, and a true-3D (not horizontal-only)
`r_spread` — were each tested standalone, then merged with the Phase 1b brake fixes into
`phase2-combined` for a full 5-seed validation across `N=2/3/4`. Result: the best `N=4` numbers
of the project (0% collision in all 3 seeds, 0% `target_lost`, `tracking_rmse` averaging 1.91).
**A real bug was caught mid-stream by supervisor review**: the first `r_spread` implementation
used the wrong angle (a global, target-viewpoint quantity where a local, drone-viewpoint one
was needed) — caught, fixed, and given a permanent regression test (`test_geometry.py`) before
it was trusted, though an isolated re-test found the bug hadn't visibly cost anything in that
one run.

With `phase2-combined` validated, a sustained-flight diagnostic
(`training/diagnose_horizon.py`, built specifically for this check) ran the best checkpoint for
60 simulated seconds instead of its 10-second training horizon — tracking error nearly
quadrupled (0.95 → 3.66) in a 20-30s window that lines up *directly* with where the real
PX4/Gazebo deployment (a concurrent conversation's work, sharing this worktree) actually
crashed (15-40s). Given that evidence and limited time, the user made an explicit, informed
call to bundle five sustained-flight changes together (longer episodes, a larger network,
ground awareness, per-axis Z/XY speed limits, dynamic target motion) rather than isolate each
first — with an agreed fallback already in place if it didn't work. It partially didn't:
ground awareness and per-axis dynamics succeeded cleanly (0 ground strikes across every
validation rollout), collision stayed unaffected (~0%), but `target_lost_rate` came back at
89-100%, flat from the very first rollout through the entire 3M-step run in all 3 seeds — a
new failure at a severity matching the original collision problem.

The user asked for a plain-language explanation of what the "grace period" (`LOST_TIMEOUT_SEC`)
actually does, then, once it clicked, gave a precise, multi-part request: increase the grace
period and sweep several values to find a working one (6s/10s/18s, specific seed counts given);
add a coordinated "active search" behavior for when the swarm loses the target entirely
(queued, not yet built); and fix reported vertical jiggling, preferring a real altitude floor
or top-down tracking over letting the drone bob. Under an explicit time-pressure instruction
("include the altitude thing then test all we don't have much time"), the altitude/jiggling
fix (empirically confirmed first, then fixed with a graduated reward preference plus
deterministic Z-velocity smoothing) was bundled into the same branch and Kaggle sweep as the
`LOST_TIMEOUT_SEC` fix, rather than tested separately. 4 of the 5 sweep kernels launched
successfully; the 5th hit a Kaggle platform quirk (session-cap error despite only 4 kernels
confirmed running) that wasn't resolved before time pressure was lifted.

Once time pressure was lifted, the user asked for two things in order: bring `docs/ai_context/`
fully up to date (this document is part of that), then go back and resolve the stuck 5th
kernel. The stuck-kernel investigation confirmed the 4 real kernels are running and everything
else on the account is genuinely complete, narrowed the likely cause to an orphaned session
from an earlier interrupted push, and found no CLI/API way to list or cancel it directly (no
Chrome browser was connected to check Kaggle's web UI in this session) — see "Unresolved"
below.

## What changed this session (chronological, key commits on `phase3-resilience` and its
ancestor branches)

- `0b6278d`/`09e3038`/`dc1de73`/`b59c139`: pre-date this session's start — the original brake +
  vision-tracking work, already on `main`. Listed here only for orientation.
- `3f251b9`: Phase 1 — multi-pass brake convergence + linear brake-engagement penalty.
  **Fixed collision, widened the formation as a side effect** — see `EXPERIMENT_LOG.md`.
- `f042fb2`: Phase 1b — thresholded the brake-engagement penalty (only excess above a derived
  threshold is taxed). Fixed the side effect without losing the collision fix.
- `2c7c296` (branch `n-aware-margin`), `4c13f86` (branch `xyz-spread`): the two Phase 2
  formation-quality hypotheses, each tested standalone first.
- `ac98d67`: fixed `r_spread`'s ideal angle (60° local, not 109.47°/120° global) — a real bug
  caught by supervisor review, given a permanent regression test (`test_geometry.py`).
- `4bd4146`/`3f9069c`: merged `n-aware-margin` and `xyz-spread` into `phase2-combined`.
- `ad4c483`: brake instrumentation (`brake_passes`/`brake_violation`/solo-multi split) — pure
  logging, no behavior change.
- `8b724bb`: added `training/diagnose_horizon.py`; checkpointed Phase 2 as a reference point in
  `PHASE2_CHECKPOINT.md`.
- `85f0615`: Phase 3 — the 5-change sustained-flight bundle (longer episodes, larger network,
  ground awareness, per-axis dynamics, dynamic target motion), on branch `phase3-resilience`.
  **Ground/per-axis clean; `target_lost_rate` catastrophic** — see `EXPERIMENT_LOG.md`.
- `ad2de3a`: checkpoint doc update recording Phase 3's launch.
- `ed2f92d`: made `LOST_TIMEOUT_SEC` env-overridable, diagnosed and documented the episode-
  length/grace-period mismatch.
- `62a685d`: added `CRUISE_ALT_MIN`/`CRUISE_ALT_COEF` (altitude preference) and
  `Z_SMOOTHING_ALPHA` (deterministic Z-velocity smoothing), bundled with the timeout fix under
  time pressure. **Current `HEAD`.**
- `bebe232` and other commits interleaved in this same branch's history: the concurrent
  deployment conversation's PX4/Gazebo/ROS2 work — a separate workstream, not detailed here,
  see `ARCHITECTURE.md`'s `deployment/` section and `deployment/docs/`.
- This doc-update pass itself (uncommitted as of this writing) — full rewrite of all 11
  `docs/ai_context/` files to reflect everything above.

## Discoveries worth knowing

- **A real, ship-before-cost-was-measured bug was caught by review, not by testing.** The
  `r_spread` angle conflation (global vs. local) passed a full single-seed test with no
  visible harm — it was still wrong, and the fix was still correct to make. Don't take "the
  test looked fine" as proof a piece of reasoning was sound; `test_geometry.py` exists
  specifically because a plausible-looking test result isn't the same as a verified derivation.
- **Eval-time numbers keep understating real problems — this happened again.** The `N=4`
  collision problem last session was invisible at eval-time (0-2%) and only visible in the
  training-time rolling-window data. This session, the pattern repeated in a different form:
  Phase 3's `target_lost_rate` was so severe it was obvious even at eval-time, but the
  *diagnosis* (that it was flat from step 1, not a mid-training regression) came from the
  training-time data, not the eval CSV. Keep checking both.
- **The bundling-multiple-changes risk is real, but managed differently each time it comes
  up.** `phase2-combined` bundled two already-individually-tested pairs (brake fixes;
  formation-quality fixes) for a final validation — a lower-risk kind of bundling. Phase 3
  bundled five genuinely untested-together changes as an explicit, informed user decision under
  real time pressure, with an agreed fallback already in place. Both are documented as
  deliberate trade-offs in `DECISIONS.md`, not lapses — but Phase 3's partial failure (tracking
  broke while safety didn't) is exactly the attribution cost that kind of bundling risks, even
  though this particular failure turned out to be diagnosable without un-bundling.
- **Kaggle's per-kernel `status` check and its account-wide `list` command both have real,
  non-obvious gaps.** `status` appears to report only a kernel's latest-version session, so an
  interrupted-then-repushed kernel can leave an old version's session running invisibly. `list`
  sorts by a `lastRunTime` that isn't reliably newest-first across the whole account. Both are
  documented in detail in `ENVIRONMENT.md`/`KNOWN_ISSUES.md` item 15 — read those before
  troubleshooting a similar Kaggle issue rather than re-discovering these the slow way.
- **The Kaggle account username is `rayensboui`, not `rayensb`** (the GitHub org name, one
  letter off) — a wrong-owner guess produces a permission-denied error that reads like a slug
  typo, not an owner typo. Cost real time this session before being caught.
- **Two requested tracking behaviors turned out to already exist.** When the user asked for
  dead-reckoning to last-known position/velocity and instant swarm-wide sharing on contact,
  both were already fully implemented in `_update_target_track()` from the original
  vision-tracking work — confirmed by reading the code before building anything, avoiding
  redundant/conflicting reimplementation. The genuinely new ask (coordinated active search when
  the whole swarm loses contact) is queued in `TODO.md`, not built yet.

## Unresolved / pending as of this handoff

1. **The `LOST_TIMEOUT_SEC` sweep — the real open problem now.** All 5 kernels are running as
   of 2026-08-21 (`timeout6-seed1/2`, `timeout10-seed1/2`, and `timeout18-seed1` — the last one
   was blocked for roughly 4-4.5 hours by a Kaggle session-cap issue, which cleared on its own;
   see `KNOWN_ISSUES.md` item 15, now resolved). A background `Monitor` task is watching all 5
   and will surface a notification per kernel as it completes/fails. **Next step for whoever
   picks this up**: once all 5 finish, download each (`kaggle kernels output
   rayensboui/<slug> -p <dir>`) and analyze — collision/target_lost/tracking_rmse, both
   eval-time and the training-time rolling-window picture, per `TODO.md`'s Critical section —
   then pick a `LOST_TIMEOUT_SEC` value to adopt.
2. **Vertical-jiggling fix unverified.** Implemented, smoke-tested, not yet re-measured against
   a trained checkpoint. Do this once the sweep produces checkpoints — load one, run the same
   400-deterministic-step sign-flip measurement used to confirm the original problem, compare.
3. **The "active search when lost" feature — queued, not started.** See `TODO.md` High section
   for the scoping notes already worked out (what's already implemented vs. what's genuinely
   new). Deliberately not bundled into the current sweep.
4. **`phase2-combined` → `main` fast-forward still blocked** on the permission classifier;
   needs the user to run `git push origin phase2-combined:main` themselves.
5. **The brake's relative-vs-absolute-velocity gap** — now concretely relevant given
   `deployment/inference_node.py` calls `_apply_brake` against real telemetry. Not started. See
   `KNOWN_ISSUES.md` item 13.
6. **`readme.txt`** and **`envs/formation_env.py`'s stale "4-agent study" docstring
   paragraph** — both still not fixed, still low priority, flagged across several doc passes
   now without anyone getting to them.
7. **`PHASE2_CHECKPOINT.md`** (repo root, shared coordination doc with the deployment
   conversation) needs an update recording this doc pass and the sweep/stuck-kernel status —
   pending as the last step of the current "document all" instruction.

## Exact recommended next step for a new session

Check `git log`/`git status` on `phase3-resilience`, and check the 5 timeout-sweep kernels'
status directly (`kaggle kernels status rayensboui/stage-marl-timeout{6,10,18}-seed{N}` for
each — don't trust the account-wide `list` command's ordering, see `KNOWN_ISSUES.md` item 15).
If all 5 are complete, download and analyze them (collision/target_lost/tracking_rmse, both
eval-time and training-time rolling-window, per `TODO.md`'s Critical section) and pick a
`LOST_TIMEOUT_SEC` value to adopt — then re-measure the vertical-jiggling fix against one of
the resulting checkpoints before considering Phase 3 done. If the 18s kernel is still stuck,
try re-pushing it once, and if it still fails, ask the user to check Kaggle's web UI for a
stray session rather than continuing to guess-and-retry. If the user has instead moved on to
something else (the deployment thread, a new feature), read `PHASE2_CHECKPOINT.md` first to
check what the other concurrent session has in flight before touching shared files
(`envs/formation_env.py`, `config.py`, `training/*.py`).
