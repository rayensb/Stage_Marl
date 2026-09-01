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

## Update 2026-08-22: search-t10 seed1/seed2 identified as the underperforming pair; 5M-step resume attempt in progress (blocked on Kaggle dataset mount, not yet running)

Read the actual downloaded `stage-marl-search-t10-s{1,2,3}` results (not memory) to confirm
which seeds are "the two underperforming ones" referenced above: **seed1** (`target_lost_rate`
0.70-0.72, `contact_fraction` 0.84) and **seed2** (`target_lost_rate` 0.92-0.96,
`contact_fraction` 0.68-0.71) are underperforming; **seed3** (`target_lost_rate` 0.04-0.10,
`contact_fraction` 0.97-0.99, `success_rate` 0.85-0.90) is the good one and is *not* part of
this retest. (An earlier config.py comment guessing "seed2" was the good one referred to a
different, older experiment — don't trust that comment for this bracket.)

**Verified before touching anything else** (per explicit user instruction): no ground-truth
leakage into obs/reward (`self.pos_t`/`_target_dir`/`_target_speed` are only read in `reset()`,
the direct-contact sensor check, and diagnostic-only `infos` fields — grep-confirmed), and the
eval loop already respects the full 90s horizon (relies on `env.agents` emptying, which honors
`MAX_STEPS` and `DISABLE_TARGET_LOST_TERMINATION` correctly).

**`training/evaluate.py` extended** (commit `c4206eb`, pushed): added pooled (not per-episode-
averaged) `mean/p95/max_true_track_err`, and a proper reacquisition-**time** distribution
(`mean/median/p95/max_reacquisition_steps` + seconds, `n_reacquisitions_observed`) — previously
only reacquisition *counts/rates* existed, no duration. Diagnostic-only, doesn't touch brake or
tracking mechanism. Smoke-tested locally against the real seed1 checkpoint before use.

**5M-step resume attempt for seed1/seed2** (continuing from their existing ~3.01M checkpoints,
not retraining from scratch — `TOTAL_STEPS=5000000`, `LOST_TIMEOUT_SEC=10` preserved, no other
config change, per explicit user instruction not to touch the brake or tracking mechanism yet):
checkpoints `latest_1.pt`/`latest_2.pt` uploaded as Kaggle dataset
`rayensboui/stage-marl-t10-resume-ckpts`, kernels `stage-marl-search-t10-s{1,2}-resume-5m`
pushed referencing it. **Both failed twice** (`cp: cannot stat
'/kaggle/input/stage-marl-t10-resume-ckpts/latest_1.pt': No such file or directory`) even
though the kernel's own stored metadata correctly lists the dataset in `dataset_sources` and
`kaggle datasets files` confirms both files are really there — the dataset is just not showing
up under `/kaggle/input/` at all. Not a propagation-delay issue (failed identically on a retry
several minutes later). A tiny diagnostic kernel (`stage-marl-diag-input-path`, just
`os.listdir("/kaggle/input")`) is running now to find the actual mount path/cause before
retrying again. **If you're picking this up: do not assume the resume kernels above are
training anything — they are not, both errored in the first ~8 seconds.**

**Resolved**: the diagnostic kernel (`stage-marl-diag-input-path`, just `os.listdir`/`os.walk`
on `/kaggle/input`) showed datasets now mount at `/kaggle/input/datasets/<owner>/<dataset-
slug>/...`, not the classically-documented `/kaggle/input/<dataset-slug>/...` (confirmed
against the current kaggle-cli docs, which still describe the old path -- this looks like an
undocumented platform change). Fixed both `verify.py` scripts' `cp` source path and re-pushed
as v3; both confirmed `RUNNING` well past the ~8s mark where the old path failed twice, so the
checkpoint copy succeeded this time. A persistent background monitor is polling both every
2.5 min for completion/failure. **Worth remembering for any future Kaggle dataset_sources use
in this project**: use `/kaggle/input/datasets/<owner>/<dataset-slug>/`, not the shorter form.

## Update 2026-08-22: train-vs-eval collision-rate discrepancy explained (mostly)

Investigated why deterministic eval collision_rate has consistently run higher than the
trailing train-time (stochastic) rate across many runs project-wide (checked ~50 downloaded
runs, not just one) -- clearest example, `search-t8-s{1,2}`: train ~1-1.2% vs eval 8-14%.

**Direct test** (`/tmp/.../scratchpad/collision_discrepancy_test.py`, kept local/scratch, not
merged into `evaluate.py`): took the exact same frozen `actor_1.pt`/`actor_2.pt` weights from
`search-t8-s1`/`s2` and ran 100 fresh episodes twice each -- once deterministic (`tanh(mu)`,
matches `evaluate.py`), once stochastic (sampled, matches `train.py`'s own rollout collection)
-- same eval seeds both times, only the action-selection mode differs.

- **seed1**: deterministic 11% vs stochastic 2% -- alone reproduces the originally-observed
  gap almost exactly. Determinism vs stochasticity is a real, demonstrated cause here, not
  speculation.
- **seed2**: deterministic 13% vs stochastic 10% -- determinism only explains a small slice;
  stochastic-replayed-after-training (10%) is still far above what training itself logged
  near the end (~1.2%).

**Conclusion**: two separate, stacking effects, not one. (1) A policy shaped by entropy-driven
exploration can converge its *mean* action to a knife-edge equilibrium riskier than the noise-
perturbed behavior actually seen during training -- deterministic eval consistently exploits
that mean. Weak secondary signal (small-N, not confirmed): deterministic collisions involved
2+ simultaneously-close agents more often than stochastic ones (seed2: 54% vs 30%) -- consistent
with the brake's already-documented gap (multi-pass convergence stress-tested only for exactly-
symmetric cases, not proven for arbitrary 3+-agent conflicts) being easier to trigger without
noise breaking up a persistent near-symmetric closing pattern. (2) For less-converged policies
specifically (seed2's tracking was itself badly broken, 92-98% target_lost), train.py's logged
number is a rolling average over its *last* ~50 episodes of a still-unstable tail, which can
catch a locally lucky patch that a fresh, frozen 100-episode re-evaluation doesn't reproduce --
a measurement-window artifact layered on top of the real determinism effect, worse for noisier/
less-converged seeds (matches: seed1, better-converged, ~fully explained by determinism alone;
seed2, worse-converged, isn't).

**Practical implication**: deterministic eval is the number to trust for real deployment (a
flight controller runs the mean policy, not a sampled one) -- the higher eval collision rates
are the honest ones, not the more optimistic training-time rolling numbers. Also strengthens
the case for the already-queued brake relative-velocity reformulation (`TODO.md`) -- a
persistent, un-perturbed near-symmetric closing pattern is exactly what a deterministic policy
can produce and exactly what the brake's proof gap is about.

## Update 2026-08-22 (later): relative-velocity brake validation launched

Commit `94216fd` (this document also updated in the same commit). 3 fresh 3M-step kernels
launched to check the reformulation actually reduces `collision_rate` in training, not just in
the two hand-picked adversarial scenarios in `test_brake_relative_velocity.py`: `stage-marl-
brake-relvel-s{1,2,3}` (`NUM_AGENTS=4`, `LOST_TIMEOUT_SEC` left at its current default (6s, not
overridden) so this is a clean test of the brake change alone, no confound from the still-open
timeout question). All 3 confirmed `RUNNING`. Uses 3 of Kaggle's 5 session slots -- the other 2
are the `search-t10-s{1,2}-resume-5m` kernels from the update above, still training.

## Update 2026-08-23: `main` fast-forwarded; brake-relvel re-launched at 8s/10s (6s batch was inconclusive)

`main` fast-forwarded to `phase2-combined` (`8b724bb`) -- clean, no conflicts, per explicit user
instruction.

The 6s brake-relvel batch above (`stage-marl-brake-relvel-s{1,2,3}`) completed but was
**inconclusive**: all 3 seeds landed in the same bad-tracking regime the plain-6s sweep already
showed (88-97% `target_lost_rate`), never reaching the tight, confident convergence where the
8-14% collision problem this fix targets was ever observed -- collision came back 0-2% in all 3,
but with nothing to stress the brake against. A quick local check (re-running `evaluate.py`'s
*current* code -- i.e. the new relative-velocity brake -- against the existing, already-trained
`search-t8-s1`/`s2` checkpoints, no retraining) showed only a small improvement over the old
brake (8%->7%, 14%->13%) -- suggesting the fix alone doesn't rescue a policy trained under the
old, looser constraint; it likely needs to be present *during* training to actually change
learned behavior, same lesson as the original multi-pass brake fix.

**Relaunched at timeouts known to let tracking converge** (5 kernels, all `RUNNING`):
`stage-marl-brake-relvel-t8-s{1,2,3}` (8s) and `stage-marl-brake-relvel-t10-s{3,4}` (10s).
Seeds were chosen to pair directly with existing old-brake results where possible: t8-seed1/2
and t10-seed3 exactly match `search-t8-s1`/`s2`/`search-t10-s3` (the seeds that converged well
under the old brake), giving the cleanest available before/after comparison; t8-seed3 and
t10-seed4 are fresh coverage.

## Update 2026-08-23: checked deployment/ for signs of active work on the crash diagnostic trio

Not touching PX4/Gazebo/ROS2 directly (no established access to that stack in this thread) --
just checked what's visible in the shared worktree. Found `deployment/verify_obs_adapter.py`
(dated 2026-08-20), a complete, already-written Phase-7-style observation-adapter verification
script (6 scenarios: converged-formation sanity, asymmetric-state + velocity-clipping, NED/
spawn-offset round-trip, clock-skew quantification, neighbor ordering) -- this appears to
already answer the first item of `PHASE2_HANDOFF.md`'s "NEXT SINGLE TEST" list, contradicting
that document's own "skipped entirely" framing, which looks stale relative to this file.
`deployment/n4_seed1_run0.json`/`run1.json`/`last_trajectory.json` (also dated 2026-08-19/20)
look like real captured trajectory data, possibly toward Phase 9's distribution comparison --
contents not inspected in detail. No sign of Phase 17 (control-loop Hz under load) having been
done. **This doc itself has had no updates from the deployment side since 2026-08-20** -- no way
to confirm from here whether that workstream is still active. Whoever picks this up next should
first re-run `verify_obs_adapter.py` (per its own docstring, needs ROS2 sourced but no daemon/
simulator running) to get a current pass/fail before assuming anything about Phase 7's status.

## Update 2026-08-23 (later): NUM_AGENTS=3 deployment-compatible training launched; first brake-relvel-t8/t10 results in, seed-reuse assumption didn't hold

**Why**: `deployment/inference_node.py` hard-asserts `NUM_AGENTS == 3` -- confirmed by reading
the code, not assumed. Every checkpoint produced this whole session (timeout sweep, active
search, the t8/t10 bracket, both brake-relvel batches) was trained at `NUM_AGENTS=4` and is
therefore **not loadable** by the real 3-drone deployment (different `OBS_DIM`, no weight
transfer -- see `docs/ai_context/DECISIONS.md`). None of Phase 3's fixes (ground awareness --
the crash's #1 hypothesized cause in `deployment/docs/PHASE2_HANDOFF.md` -- active search, or
the brake reformulation) have ever been trained at N=3. Adapting the deployment side to 4
instead was considered and rejected: `SPAWN_XY`/`AGENT_NAMESPACE` are hardcoded 3-entry dicts
tuned to N=3's exact 120-degree planar geometry, N=4's ideal formation is a non-planar regular
tetrahedron (a real, previously-confirmed project finding), and it would add a new variable to
a deployment stack that's currently mid-crash-diagnosis -- worse methodology, not just more work.

**Launched**: `stage-marl-deploy-n3-s{1,2}` (`NUM_AGENTS=3`, `TOTAL_STEPS=3000000`,
`LOST_TIMEOUT_SEC=8` -- the only timeout value with a clean N=4 convergence record so far, current
`phase3-resilience` code). Both confirmed `RUNNING`, launched one at a time as brake-relvel
slots freed up (Kaggle's 5-session cap). **Whoever picks up the deployment side next**: these
will be the first Phase-3-era, ground-aware, N=3-compatible checkpoints once complete -- check
`kaggle kernels status rayensboui/stage-marl-deploy-n3-s{1,2}` before assuming either is ready.

**First brake-relvel-t8/t10 results, and an important negative finding about the seed-reuse
design**: `t8-s3` (fresh seed, no old-brake counterpart): 10% collision, 6% `target_lost`, 1.97
tracking_rmse -- excellent tracking, but collision still double-digit even training with the new
brake from scratch. `t8-s1` (same seed as the original `search-t8-s1`, which had 8% collision /
2-3% `target_lost` under the *old* brake): **90% `target_lost_rate`** under the new brake -- a
completely different outcome, not just a different collision number. Reusing a seed does not
give a clean "only collision changes" comparison the way it was expected to: the brake's
behavior differs the moment two agents are close enough for `v_closing` to differ between
formulas, and since training is on-policy, everything downstream of that first divergence can
send the whole run to a different outcome entirely. **Don't treat the seed-1/seed-3(t10) pairings
as clean before/after comparisons** -- read each new result on its own terms, not as a paired
diff against its old-brake counterpart. Still waiting on `t8-s2`, `t10-s3`, `t10-s4`.

## Update 2026-08-23 (later still): relative-velocity brake looks net-harmful to training convergence -- 6 of 7 seeds failed, root mechanism unclear, no deployment-ready checkpoint exists

**Full result, all 7 seeds trained with the new (relative-velocity) brake code, this update
session**:

| seed | config | old-brake baseline | new-brake result |
|---|---|---|---|
| t8-s1 | N=4, 8s | 2-3% target_lost (converged) | **90% target_lost (failed)** |
| t8-s2 | N=4, 8s | 2% target_lost (converged) | **86% target_lost (failed)** |
| t8-s3 | N=4, 8s | (fresh, no baseline) | 6% target_lost, 10% collision (converged) |
| t10-s3 | N=4, 10s | 4% target_lost (converged) | **95% target_lost (failed)** |
| t10-s4 | N=4, 10s | (fresh, no baseline) | **90% target_lost (failed)** |
| deploy-n3-s1 | N=3, 8s | (no N=3 Phase-3 baseline exists) | **99% target_lost (failed)** |
| deploy-n3-s2 | N=3, 8s | (no N=3 Phase-3 baseline exists) | **96% target_lost (failed)** |

**6 of 7 failed to converge on tracking at all -- including both `NUM_AGENTS=3` deployment-
target seeds.** The 3 seeds with a direct old-brake baseline (t8-s1, t8-s2, t10-s3) all had a
**clean 3/3 success record under the old brake and a clean 0/3 record under the new one, same
seeds, deterministic training** -- not the usual bimodal seed-variance pattern this project has
seen elsewhere, a genuine reversal on repeat, strong evidence the brake change itself is
causally responsible, not incidental.

**Leading hypothesis (fires harder per unit of tight tracking, since the new formula reads
~2x the closing speed under symmetric approach) was tested directly and REFUTED**: compared
`search-t8-s1` (old brake, converged) against `brake-relvel-t8-s1` (new brake, same seed,
failed) at the final training row. `r_brake` was only marginally more negative (-0.091 vs
-0.129, not dramatically larger), and **`mean_brake_reduction` was actually *smaller* in the
failed run** (0.0025 vs 0.0008) -- the brake is barely engaging in the failed run, most likely
because the swarm never gets close enough to need it (`avg_min_dist` 6.57 vs 8.14, wider not
tighter). This rules out "harsher direct penalty" as the mechanism. **The actual cause is not
yet understood** -- would need reward-component *trajectories* over the course of training
(not just the final snapshot) to see where/when the two runs' paths actually diverge, not yet
done.

**Bottom line**: the relative-velocity brake fix, in its current form, looks like a net
negative for training convergence, not a clean safety improvement -- and there is currently
**no deployment-ready (`NUM_AGENTS=3`, Phase-3-era) checkpoint**, since both attempts at that
config also failed to converge under the new brake. **Open question posed to the user, not yet
answered**: dig further into the mechanism, or revert the brake to the old (one-sided-velocity)
formula and treat this as a dead end for now. Whoever picks this up next should check for that
answer before doing either.

## Update 2026-08-24: brake reverted, network-capacity sweep launched

Per explicit user decision: reverted the relative-velocity brake to the old, proven one-sided
formula (commit `55e7232`) rather than keep chasing the unexplained convergence-breaking
mechanism -- deployment-readiness took priority, and the new brake had no confirmed safety
benefit to weigh against the cost anyway (its one converged run still showed 10% collision,
no better than before). `test_brake_relative_velocity.py` kept as a general regression test --
still passes, since both its scenarios are symmetric and the reverted formula already handles
those cleanly. `_apply_brake`'s CAUTION comment records the full tried-and-reverted story for
whoever revisits this later.

Also added `ACTOR_HIDDEN`/`CRITIC_HIDDEN` env-var overrides (`config.py`, `training/networks.py`,
threaded through `train.py`/`evaluate.py`/`diagnose_horizon.py`) -- the 128/256 hidden width has
been Phase 3's default since it was bundled in as "a starting guess... not measured," never
actually isolated. `deployment/sim_demo.py`/`inference_node.py` were deliberately **not** touched
(still hardcode `Actor(OBS_DIM, ACT_DIM)`, i.e. assume the 128 default) -- fine for now since
nothing has been deployment-tested yet, but whoever wires up the eventual winning checkpoint
needs to either match its hidden width there or update those two files the same way.

**Launched** (5 kernels, all `RUNNING`, fills Kaggle's cap): a network-capacity sweep on the
reverted brake, `LOST_TIMEOUT_SEC=8` (the best-evidenced timeout so far), 3M steps each --
`stage-marl-netsweep-n4-{small,medium,large}` (Actor/Critic hidden 64/128, 128/256, 256/512;
`NUM_AGENTS=4`) and `stage-marl-netsweep-n3-{medium,large}` (128/256, 256/512; `NUM_AGENTS=3`,
this project's first Phase-3-era N=3 attempt on a *working* brake -- the two prior N=3 attempts
both used the since-reverted broken brake and both failed). `netsweep-n4-medium` deliberately
reuses `SEED=1` with the exact same config as the original `search-t8-s1` (which converged
cleanly under the old brake before the relative-velocity detour) -- a free sanity check that
the revert actually reproduces that result; if it doesn't, something is more wrong than just
the brake formula, worth flagging as a red flag if it comes back non-converged.

## Update 2026-08-25: critic-collapse investigation -> CRITIC_LR added -> matched-seed Kaggle experiment launched

Deep supervisor-guided investigation into why active-search checkpoints fail to recover from
`target_lost` (79-100% failure rate) despite a scripted (non-learned) controller solving the
identical forced-loss task 96-100% of the time using the same search mechanism. Chain of local
diagnostics (all in `/tmp/.../scratchpad/`, not merged into the repo): forced-loss masking test
(confirmed the search mechanism itself is usable), PPO-vs-scripted action/reward instrumentation
on a real trained checkpoint (found PPO's actions diverge substantially from scripted but
one-step rewards are nearly identical -- pointed at credit assignment, not reward shaping), then
a value/critic diagnostic that found the production `CentralCritic`'s **value predictions are
essentially constant regardless of state** (`std=0.000000` across 90 real loss-states spanning
the full 1-120 step horizon). Traced the mechanism: the critic's **second Tanh layer saturates
100%** within the very first rollout of training (confirmed via a from-scratch reproduction,
reproducing exactly at pure random init: 0% saturated; after ~14,400 steps: 100%), and this is
**not specific to layer 2** -- removing it just relocates the same saturation onto whichever
layer is last before the value head. A critic-loss-function sweep (MSE/Huber/normalized-MSE)
found the loss function's outlier-sensitivity does **not** control this (all three saturate
identically despite gradient norms differing by ~50,000x) -- instead, a per-update trajectory +
LR sweep found **critic learning rate** is the controlling variable: `CRITIC_LR=1e-5` (30x
smaller than the shared default) fully prevents saturation over a 2250-update reproduction,
where the production default (shared with actor LR, `3e-4`) reaches 100%. A corrected, properly
stratified (not temporally-contiguous) diagnostic confirmed the critic's earlier-measured
anti-correlation with real returns was substantially a sampling artifact, and shrinks toward
statistical noise as LR drops -- real signal, but short of confirmed-positive at this tiny
diagnostic budget.

**Production change** (commit `71327be`, pushed): `training/train.py` now has a separate,
env-var-overridable `CRITIC_LR` (defaults to `LR`, i.e. unchanged behavior unless set), and
logs critic-health diagnostics every rollout to `training_log.csv` (`critic_v_mean/std/min/max`,
`critic_target_mean/std`, `critic_pearson`/`critic_spearman` vs real returns, and the final
hidden layer's saturation fraction/pre-activation std) using a random subsample of that
rollout's own pooled data. **Caught and fixed a real bug before any Kaggle spend**: the
pre-existing LR-anneal block unconditionally set both `opt_actor` and `opt_critic` to the same
actor-derived `lr_now` every rollout, silently overwriting `CRITIC_LR` back to a `LR`-derived
value on the very first update regardless of what it was constructed with -- caught by a local
smoke test producing bit-identical output under `CRITIC_LR=3e-4` vs `CRITIC_LR=1e-5`; each
optimizer now anneals from its own base LR. Verified fixed (divergent `final_sat=0%` vs `100%`
output) before pushing.

**Launched, completed, and analyzed** -- `stage-marl-criticlr-{control,1e5}-s{1,2,3}` (5 on
Kaggle, treatment-seed3 run locally instead -- local throughput measured faster than Kaggle's,
~700 vs ~330 steps/sec). **Result: rejected as a general recovery fix.** Full writeup in
`docs/ai_context/EXPERIMENT_LOG.md` ("Critic-LR ablation" entry) -- short version: saturation is
delayed, not prevented (all 3 treatment seeds reach 100% final-layer saturation again by the end
of the 3M-step run), and behavior was a mixed bag across the 3 matched seed-pairs -- one seed
traded a real `target_lost`/`contact_fraction` improvement for a new collision cost, two seeds
showed no improvement or regression, and `mean_reacquisition_steps` was worse under `CRITIC_LR=
1e-5` in all three pairs (the one fully consistent effect, pointing against the treatment).
`CRITIC_LR`'s default is unchanged (still equals `LR`) -- do not set it to `1e-5` based on this
result. The critic-health columns added to `training_log.csv` (this commit, `71327be`) stay --
useful instrumentation regardless of this specific ablation's outcome; older logs won't have them.

**Follow-on, in progress**: since critic saturation is now a confirmed-but-secondary pathology,
investigation moved to the actor's own action dynamics during search (natural loss events, PPO
vs scripted at the per-step action level, not just reward/value). See EXPERIMENT_LOG.md's "Actor
search-action dynamics" entry -- found a concrete near-miss mechanism (one agent closing steadily
and correctly, ~30 steps from success, while 2-3 teammates drift consistently the wrong way for
the whole event) rather than a uniform "PPO can't do this" failure. Two candidate mechanisms
flagged (uncorrected bad headings persisting all event; the 120-step budget being a tight margin
even for a working trajectory, not just a training-quality gap) -- neither yet tested, no
direction chosen.

## Update 2026-08-26: decisive result -- localized to the actor, not the environment/search mechanism

Follow-on to the actor-dynamics work above. Best-agent/productive-agent re-analysis of the same
30 natural PPO loss events gave a clean signature: reacquired events average 2.25 productive
agents (0% have zero); target_lost events average 0.41 (70.6% have zero). A scripted-controller
A/B/C search-strategy comparison and a 6/8/10/12s timeout sweep (both forced-loss, starting from
a well-formed scripted warmup) both hit a 100% ceiling regardless of condition -- informative in
itself (neither search heading choice nor the 6s window is the bottleneck when execution is
already good), but structurally unable to test PPO's own, worse-formed states.

**The decisive experiment**: branched PPO / scripted / adaptive-reassignment controllers from
the *exact* env states (positions, velocities, target state, RNG) PPO's own policy created at
28 real loss-onset moments -- not an artificial mask on a good formation. Cross-tabulated against
what PPO's real trajectory actually did from each state. Result: from the 16 states where PPO's
real trajectory failed (`target_lost`), **PPO replayed from that state fails again (0%), but
both scripted and adaptive controllers reacquire from every single one (100%)**, typically
within 1-2 steps. `min_true_dist` for those recoveries averages 7.88m, right at the 7.91m
sensor boundary -- these are not hard states, PPO just doesn't execute the (comparatively easy)
correction. This resolves the causal fork the whole active-search investigation has been
building toward: **the problem is the learned actor's own search execution, not the
environment, not the search geometry/heading strategy, not the 6s timeout, and (per the
critic-LR ablation above) not primarily the critic.**

**Update 2026-08-26 (later)**: now fully written up in `docs/ai_context/EXPERIMENT_LOG.md`
(4 new entries: network-capacity sweep, best-agent/productive-agent/confidence-response
reanalysis, search-strategy A/B/C + timeout-margin sweep, and the decisive counterfactual above),
plus corresponding updates to `DECISIONS.md`, `KNOWN_ISSUES.md` (items 12/13/18/19),
`ARCHITECTURE.md`, `ENVIRONMENT.md`, `COMMANDS.md`, `CURRENT_STATE.md`, `SESSION_HANDOFF.md`,
`TODO.md`, `AI_CONTEXT.md`, and `INDEX.md` -- a full "document all" pass triggered by explicit
user request. Also corrected a stale claim that had persisted across three days of doc passes:
`origin/main` was actually fast-forwarded to `phase2-combined` (`8b724bb`) on 2026-08-23, not
"still behind" as several files continued to say.

## Update 2026-08-26 (later still): credit-assignment diagnostic -- confirms a long-horizon signal problem, not yet a fix

Direct follow-on to the decisive counterfactual above, per explicit supervisor-style request:
"why does PPO learn such an ineffective search policy, given the states are demonstrably
recoverable." Used the identical 16 real PPO-failure loss-onset states, the real trained critic
(`checkpoints/latest_1.pt` for `search_v2_seed1`, not a substitute), and training's own exact
GAE formula (`gamma=0.99`, `lambda=0.95`, copied from `training/buffer.py`) -- all local,
read-only, no production changes (`/tmp/.../scratchpad/credit_assignment_diagnostic.py`).

**Bug caught and fixed before trusting any number**: `counterfactual_from_ppo_states.py`'s
`BRANCH_CAP = LOST_TIMEOUT_STEPS` (120) is one step short of where `_target_lost` actually
fires (`steps_since_contact > LOST_TIMEOUT_STEPS`, i.e. step 121) -- doesn't change that
experiment's 0%-vs-100% reacquisition conclusion (nothing reacquires on one extra ramp-maxed
no-op step), but this diagnostic specifically needed the true terminal step, so `BRANCH_CAP` is
`+1` here. Confirmed fixed: terminal-step `contact` component now reads exactly -230.00 for
still-lost PPO branches (-30 ramp + -200 `TARGET_LOST_PENALTY`, matches `config.py` exactly).

**Result**: one-step reward difference (scripted - PPO) is small but real, mean +0.36 (vs. mean
rewards around -11 to -12) -- not literally indistinguishable, but roughly 3-4 orders of
magnitude smaller than the multi-step gap: mean discounted return difference +1716, undiscounted
+3460. Component breakdown of the multi-step gap: `contact` (+2045, the terminal penalty plus
~120 steps of ramp difference) and `track` (+1513, PPO's swarm keeps accumulating tracking-error
cost while still lost) are the two dominant drivers; `safety` actually runs slightly the other
way (-112, a secondary effect, not investigated further). GAE advantage of the actual first
action taken: PPO's own first action scores -110.7 on average; scripted's counterfactual first
action from the identical state scores -4.2 -- scripted's advantage is higher in 16/16 states.
**V(s) at the loss-onset state is -876.51 with std=0.0000 across all 16 -- a fresh, independent
reconfirmation of the critic-saturation finding** (the critic assigns literally the same value
regardless of which of these 16 different states it's looking at).

Separately, across a fresh natural-event sample (1608 reacquired-event steps, 8092 target_lost-
event steps), per-step reward barely correlates with a simple progress proxy (reduction in
distance to the sensor boundary) at all: pearson ~-0.04 to -0.08 in both outcome groups. The
reward doesn't reward the one thing that actually matters for search, step by step.

**Conclusion (diagnosis only, no fix proposed or attempted, per explicit instruction)**: this is
real, converging evidence for a long-horizon credit-assignment problem, not (or not only) a
reward-shaping gap in the ordinary sense -- the per-step reward doesn't discriminate a good
search step from a bad one, the only real signal is a huge, ~120-step-delayed, all-or-nothing
outcome, and the critic that's supposed to help propagate that signal has collapsed to a
near-constant function of state. All three pieces point the same direction independently. Full
numbers reported to the user directly; not yet written into `docs/ai_context/EXPERIMENT_LOG.md`
(gated to explicit "document all" requests) or acted on.

## Update 2026-08-26 (later still): three more diagnostics narrow the actor-execution problem to "weak but real directional judgment," not heading choice or entrenchment

Before proposing the estimate-progress reward the credit-assignment diagnostic motivated, a design
catch stopped it: that reward would have to use ground truth (`env.pos_t`) to compute boundary
progress, which the project's established rule forbids (`_get_reward`/`_get_obs` must never read
`self.pos_t` -- the same class of bug a supervisor review caught once for `r_safety`). The only
legitimate (agent-observable) proxy -- progress toward the agent's own synthetic waypoint -- turned
out much weaker (r~0.09-0.11 vs ground-truth's r~0.29-0.31), and a sign-mismatch check found an
estimate-progress reward would reward a step actually moving away from the true target ~44.8% of
the time. Reward shaping was shelved pending further diagnosis, not implemented.

**Search-heading quality** (`search_heading_quality_diagnostic.py`): the assigned search heading's
own alignment with the true target is statistically indistinguishable between successful and
failed events (-0.040 vs -0.045 pooled; -0.007 vs -0.006 on the 16/12 exact matched states) --
heading *choice* isn't the differentiator. A last-known-velocity-vs-random-heading counterfactual
from the same 16 real failure states confirmed this directly: both reach 100% reacquisition in a
mean 1.4 steps, identically -- heading source doesn't matter at all this close to the sensor
boundary (min true dist ~7.88-7.90m either way). What *is* real and consistent across two datasets:
PPO's actual action aligns more with true target velocity and with last-known-velocity in
successful events than failed ones (last-known-vel: 0.292 vs 0.175 pooled, 0.410 vs 0.256 on the
matched states) -- confidence-bucketed, this alignment *increases* as confidence falls in
target_lost events (0.163->0.180->0.197), the opposite of losing access to the cue.

**Directional entrenchment, tested and rejected** (`directional_entrenchment_diagnostic.py`): the
rising alignment above looked like escalating commitment to a (mediocre) heading. Directly
measured instead: P(a >45deg direction change at step t | consecutive bad streak ending at t-1)
spikes right after progress turns negative (14.4% at streak 1-5) then flattens at a low, roughly
constant 5.5-6.8% for streak 6-10/11-20/21+ -- no escalating refusal to reconsider. Per-agent,
persistently-unproductive agents actually correct *more* than eventually-productive ones (42.3% vs
31.1% ever-corrected during a sustained bad streak; 5.77 vs 1.76 mean corrections each) -- the
opposite of the entrenchment hypothesis. Retired.

**Correction quality -- the decisive result** (`correction_quality_diagnostic.py`, n=2006 real
corrections + a 150-correction branched null test against random-direction rotations from the
identical state): corrections made by eventually-productive agents lead to sustained positive
progress 62-70% of the time (5/10/20-step windows); corrections made by persistently-unproductive
agents lead to sustained positive progress only 3.3-4.5% of the time -- a ~15-20x gap, the cleanest
split in this whole investigation. The null test shows PPO's actual chosen direction beats a
same-state random rotation 61-62% of the time, consistently across window lengths -- real,
non-random directional judgment, not pure noise -- but absolute success remains low even for real
corrections (pooled P(sustained+) ~12-13%, matching a weighted blend of the two agent classes
above). **Synthesis: PPO exercises genuine but weak directional judgment during search --
meaningfully better than chance, but not good enough, especially exactly for the agents already in
trouble.** Not "wandering" (beats random) and not "purposeful and effective" (mostly fails anyway)
-- a real middle case.

**Status**: no fix proposed or attempted at any point in this chain, per explicit instruction each
time. The investigation has now localized the problem about as precisely as diagnosis alone can:
target_lost during active search traces to the learned actor's own directional judgment during
correction being real-but-weak, not to heading initialization, timeout, critic, or entrenchment/
persistence. What (if anything) to do about it is an open question for the user -- no direction
chosen yet.

## Update 2026-08-26 (later still): first actor-side intervention implemented, smoke-tested, awaiting launch go-ahead

Per explicit user decision to leave diagnosis mode: one more small offline check first, then the
first real intervention. **Separability check** (`separability_check.py`): logistic regression on
the raw 39-dim pre-correction observation predicting sustained-positive-progress-after-correction
-- test AUC=0.834 vs a 0.786 majority-class baseline / 0.5 chance. Confirms the agent-observable
state already contains substantial exploitable information the policy isn't fully using, ruling
out "not enough information" and justifying an actor-side (not observation-side) intervention.

**Implemented** (commit `a2322ca`): `AUX_DIR_COEF`, a new env-overridable constant in
`training/train.py` (default `0.0`, verified no-op), adds a small auxiliary actor loss term --
active only during search (`steps_since_contact > 0`), encouraging the actor's current mean action
direction toward `unit(_last_known_vel)` -- a cue already shown (this investigation) to correlate
with PPO's own successful corrections, agent-observable, no ground truth involved. Not reward
shaping (the credit-assignment diagnostic found the ground-truth-free reward-proxy equivalent has
~44.8% sign mismatch with true progress) and not imitation of the scripted controller (a directional
nudge the policy can override via its own gradient, not a hard-coded target). Required threading
`aux_dir`/`in_search` through `RolloutBuffer` (`training/buffer.py`) and extending
`Actor.evaluate()` to also return the current mean action direction (`training/networks.py`, the
only call site, in `train.py`'s minibatch loop, was updated to match) so the auxiliary loss uses a
gradient-connected quantity, not the buffer's already-sampled (and by then stale) action.

**Smoke-tested, three configurations, all local**: `AUX_DIR_COEF` unset -- clean, `coef=0.000` in
every logged row, no crash. `AUX_DIR_COEF=1.0` (deliberately exaggerated) -- `aux_cos` climbs
0.363→0.468→0.969→0.909 across 4 rollouts, confirming the mechanism is correctly wired and
functionally effective end-to-end, not just non-crashing. `AUX_DIR_COEF=0.01` (the actual intended
treatment value, sized by comparing to this project's own logged `actor_loss` magnitude -- median
0.04, p90 0.11 -- rather than guessed outright) -- produces a modest, non-dominating nudge as
intended. `test_env.py` (N=2/3/4) and `test_geometry.py` both still pass.

**Also added** (commit `3b9a73d`): `productive_agent_fraction`/`mean_productive_agents`/
`frac_events_zero_productive` to `training/evaluate.py`'s summary output -- per loss event, how
many of `NUM_AGENTS` ever closed on the true target during that event, ground-truth-based and
eval-only exactly like `tracking_rmse`, needed to evaluate this intervention against the same
metric this whole investigation has used to characterize search quality. Smoke-tested against an
already-trained checkpoint: internally consistent (`mean_productive_agents / NUM_AGENTS ==
productive_agent_fraction` exactly), sane values for a near-random policy.

**Status**: implementation complete and smoke-tested; **the actual Kaggle launch (matched 3-seed
baseline-vs-treatment, 3M steps each, current `phase3-resilience` HEAD config) has not been
started** -- awaiting a final go-ahead from the user before spending Kaggle compute, per this
project's standing practice of treating "implement/commit" and "spend real compute" as two
separate approval gates.

## Update 2026-08-26 (later still): matched 3-seed baseline-vs-treatment launched (5 Kaggle + 1 local)

Explicit user go-ahead received ("5 to kaggle and 1 locally"). Pushed the 7 pending local commits
to `origin/phase3-resilience` first (standing pre-launch rule). Launched, same structure as the
CRITIC_LR ablation: `stage-marl-auxdir-control-s{1,2,3}` (AUX_DIR_COEF unset/default) and
`stage-marl-auxdir-treat-s{1,2}` (AUX_DIR_COEF=0.01) on Kaggle -- all 5 confirmed `RUNNING`.
Treatment seed 3 run locally instead (Kaggle's 5-session cap, matching the CRITIC_LR ablation's
exact precedent).

**Caught before it could silently corrupt the experiment**: the local `SEED=3`/`run_id=3` launch
first tried to resume from `checkpoints/latest_3.pt` -- a **leftover checkpoint from the earlier,
unrelated CRITIC_LR=1e-5 local treatment-seed-3 run** (same run_id, same worktree, already fully
consumed at 3,009,600 steps), and since that's already past this run's `TOTAL_STEPS=3,000,000`,
`train.py` exited instantly without training anything. Caught by checking the log/process, not
assumed. Fixed by moving the old `checkpoints/latest_3.pt`, `models/actor_3.pt`,
`models/actor_best_3.pt`, `logs/training_log_3.csv`, `logs/eval_3.csv` aside (preserved, not
deleted) before relaunching cleanly -- confirmed via `ps` that the process is now actually
training (real CPU usage, elapsed time advancing). All 6 arms (5 Kaggle + 1 local) confirmed
running as of this update.

**Status**: launched, in progress. Next step once complete: download all 6, run
`training/evaluate.py --episodes 100` (both final and `--best`) already happens inside each
Kaggle kernel and will be run locally for the 6th; compare `target_lost_rate`, `contact_fraction`,
`reacquire_rate_eventual`, `mean_reacquisition_steps`, `collision_rate`, `productive_agent_fraction`
per the pre-agreed decision rule (≥2/3 seeds improve without unacceptable safety regression -> keep;
otherwise abandon and move to the next intervention).

## Update 2026-08-27: result — passes decisively at both final and best checkpoints; AUX_DIR_COEF=0.01 adopted as the new default

All 5 Kaggle kernels completed; downloaded and parsed alongside the local seed-3 result. Every
core metric improved in **3/3 seeds** (not just the ≥2/3 threshold), at **both** final and best
checkpoints: `target_lost_rate` 0.83/0.86/0.98 → 0.26/0.14/0.38 (final), 0.79/0.95/0.98 →
0.26/0.14/0.56 (best); `success_rate` 0.17/0.13/0.02 → 0.72/0.83/0.62 (final); `contact_fraction`
and `productive_agent_fraction` up in every seed at both checkpoints (`mean_productive_agents`
nearly doubles). `collision_rate` rises slightly in 2/3 seeds (1-3 points) — real but small next
to the 8-14% costs seen elsewhere in this project, not treated as disqualifying.

Per explicit user instruction, ran exactly the one requested validation (best-checkpoint
comparison, same protocol) rather than a coefficient sweep or further diagnosis, confirmed the
result survives it, then stopped. **`AUX_DIR_COEF`'s default changed from `0.0` to `0.01`**
(commit `84232ab`) — this is now the standing default for any future `training/train.py` run at
`NUM_AGENTS=4` unless explicitly overridden to `0.0`. Recorded in full in `EXPERIMENT_LOG.md`
("Search-direction auxiliary objective" entry) and `DECISIONS.md`.

**This is the first change in the entire active-search/`target_lost` investigation (spanning the
critic-collapse chain, the actor-search-dynamics chain, and this intervention) that measurably
improved the primary failure metric, not just localized where it lives.** Per explicit decision,
no coefficient sweep, no further diagnosis, no other mechanism change planned right now — this is
the new experimental baseline to build on. Whoever picks this up next (including the deployment
thread, if a fresh `N=3` training attempt is tried) should know the default has changed and that
`checkpoints/`/`models/` produced before commit `84232ab` were trained under the old
(`AUX_DIR_COEF=0.0`) behavior, not this one.

## Update 2026-08-27 (later): new-baseline failure-mode audit -> AUX_DIVERSIFY implemented, launch pending

Per explicit instruction, did **not** launch another 3M-step run just to check whether more
training closes the remaining gap. Instead re-used the already-trained `AUX_DIR_COEF=0.01`
checkpoints as the new Phase 3 baseline (zero new training cost) and audited remaining failures
from data already downloaded: pooled 300 eval episodes (3 seeds) split **72.3% success / 26.0%
target_lost / 1.7% collision / 0.0% ground_strike** -- target_lost remains ~15x more common than
collision, still the dominant failure mode. Failed episodes average `episode_len=1176.5` (of 1800
max) and `contact_fraction=0.865` even at failure -- these are late, genuine recovery failures,
not early collapse. Of the 78 pooled target_lost episodes, 67.9% fail on the swarm's very first
loss event of that episode (`longest_loss_streak` mean=121, a real 6s-timeout exhaustion, not
something else).

**`AUX_DIVERSIFY` implemented** (commit `bb2b3dd`), specified entirely from already-established
evidence, no new diagnostic run: `AUX_DIR_COEF` as built pulls every agent toward the identical
`unit(_last_known_vel)` direction every search step -- in tension with active search's own design
(deliberately spread headings for coverage) at exactly the point this project's most robust,
repeatedly-reconfirmed finding (productive-agent count/diversity separates success from failure,
both pre- and post-AUX) says diversity still matters. When enabled, blends
`unit(_last_known_vel)` with each agent's own already-diverse `_search_dirs[agent]` instead of
using one shared direction for the whole swarm -- `AUX_DIR_COEF`'s magnitude and everything else
unchanged, isolating exactly one variable against the *current* baseline. Smoke-tested both paths
(`AUX_DIVERSIFY` unset vs `=1`), no crash, sane values, existing test suite still passes.

**Status**: implemented and smoke-tested only. **No Kaggle launch yet** -- awaiting the user's
go-ahead on the matched 3-seed comparison (control = current baseline, `AUX_DIVERSIFY` unset;
treatment = `AUX_DIVERSIFY=1`), same protocol as every prior ablation this investigation.

## Update 2026-08-27 (later still): AUX_DIVERSIFY matched comparison launched (5 Kaggle + 1 local)

Explicit user go-ahead received. Checked `git status` was already up to date with `origin`
(nothing new to push). Launched, identical structure to the AUX_DIR_COEF launch:
`stage-marl-diversify-control-s{1,2,3}` (`AUX_DIVERSIFY` unset -- current baseline,
`AUX_DIR_COEF=0.01` only) and `stage-marl-diversify-treat-s{1,2}` (`AUX_DIVERSIFY=1`) on Kaggle
-- all 5 confirmed `RUNNING`. Treatment seed 3 run locally (Kaggle's 5-session cap).

**Same checkpoint-collision issue as last time, caught proactively this round**: `run_id=3`
locally still held the *previous* local run's artifacts (the `AUX_DIR_COEF=0.01`-only
treatment-seed-3 run from the last ablation) at 3,009,600 steps -- checked for this specifically
before launching (having been bitten by it once already) and moved
`checkpoints/latest_3.pt`/`models/actor_3.pt`/`models/actor_best_3.pt`/`logs/training_log_3.csv`/
`logs/eval_3.csv`/`logs/eval_best_3.csv` aside (preserved, not deleted) before relaunching --
confirmed via `ps` the process is actually training this time, not exiting instantly. All 6 arms
confirmed running as of this update.

**Status**: launched, in progress. Same evaluation plan as before: download all 6 once complete,
compare `target_lost_rate`, `contact_fraction`, `success_rate`, `productive_agent_fraction`/count,
`collision_rate`, and the failure-category breakdown, at both final and best checkpoints, against
the acceptance criterion (meaningful `target_lost_rate` reduction in ≥2/3 seeds without
unacceptable collision regression).

## Update 2026-08-27 (later still): AUX_DIVERSIFY rejected (clean negative result); AUX_DIR_RAMP implemented and smoke-tested, awaiting launch go-ahead

**AUX_DIVERSIFY result**: all 6 arms (5 Kaggle + 1 local) completed. Matched 3-seed comparison
against the `AUX_DIR_COEF=0.01`-only baseline: 1 seed improved, 1 seed regressed clearly
(`target_lost_rate` +0.18, the largest single movement in the table), 1 was essentially flat.
`productive_agent_fraction` did not improve consistently either (flat-to-down in 2/3 seeds) --
the specific hypothesis motivating this (uniform direction trades away useful coverage
diversity) is not supported by this implementation. **Fails the pre-agreed ≥2/3-seeds bar --
rejected.** Per explicit user instruction: this is a clean negative result about this specific
diversification formulation, not evidence against the AUX approach generally (`AUX_DIR_COEF=0.01`
was a clean 3/3 win and stays the baseline). `AUX_DIVERSIFY` stays off by default; no further
diagnosis planned on this specific hypothesis. One methodological caveat noted but not
disqualifying: seed 3's control ran on Kaggle while its treatment ran locally (an
execution-environment confound), but seeds 1/2 (both fully on Kaggle) already independently show
the same mixed 1-up-1-down split, so it doesn't change the conclusion.

**`AUX_DIR_RAMP` implemented** (commit `dee4016`), a different kind of follow-on to
`AUX_DIR_COEF` -- not a different target direction (that's what `AUX_DIVERSIFY` tried, and it
didn't work), but a different *strength schedule* for the same, already-validated direction
(`unit(_last_known_vel)`). Specified entirely from existing evidence, no new diagnostic run:
(1) `r_contact`'s own urgency ramp (`CONTACT_URGENT_COEF * min(1, steps_since_contact/
LOST_TIMEOUT_STEPS)`) is an already-validated pattern in this exact codebase for "increase
pressure as the timeout approaches"; (2) the directional-entrenchment diagnostic found
correction probability does *not* rise on its own as a bad streak lengthens (flat ~5.5-6.8% for
streaks of 6-10/11-20/21+ steps) -- the actor doesn't self-generate escalating urgency, so an
explicit signal that does could compensate; (3) the post-AUX_DIR_COEF failure audit found 67.9%
of remaining `target_lost` failures are fatal on the swarm's very first loss event of the
episode, with `contact_fraction` still ~0.865 even in failed episodes -- these are "ran out of
time on the one search that mattered" failures, exactly where a pull that gets more assertive as
time runs low has the most leverage. When set > 0, the per-transition auxiliary coefficient
becomes `AUX_DIR_COEF * (1 + AUX_DIR_RAMP * min(1, steps_since_contact/LOST_TIMEOUT_STEPS))`
instead of a flat `AUX_DIR_COEF` for the whole search window. Required extending
`RolloutBuffer` (`training/buffer.py`) with a new `aux_ramp_frac` field threaded through
`store()`/`get_tensors()`, and rewriting the minibatch loop's auxiliary-loss computation to use a
per-row coefficient instead of one scalar.

**Smoke-tested, three configurations, all local, all exercising real in-search rows (not just
the trivial all-False case)**: `AUX_DIR_RAMP=0` -- `coef_mean` logged exactly `0.0100`
(== `AUX_DIR_COEF`) in both rollouts despite real search rows present (`frac_insearch` 0.273/
0.203), confirming the ramp is an exact no-op, not just algebraically but under live rollout
conditions. `AUX_DIR_RAMP=50` (deliberately exaggerated) -- `coef_mean` jumped to 0.258/0.278,
no NaN/crash, training proceeded normally. `AUX_DIR_RAMP=1.0` (the actual intended treatment
value, doubling the coefficient by the time the timeout is reached) -- `coef_mean` landed at
0.0156/0.0141, inside the expected `[0.01, 0.02]` band. `test_env.py` (N=2/3/4) and
`test_geometry.py` both still pass (unaffected -- this change doesn't touch `envs/
formation_env.py`).

**Status**: implemented, smoke-tested, committed. **No Kaggle launch yet** -- awaiting the
user's go-ahead on the matched 3-seed comparison (control = current baseline unchanged,
`AUX_DIR_RAMP` unset; treatment = `AUX_DIR_RAMP=1.0`), same protocol and acceptance criterion
(≥2/3 seeds improve `target_lost_rate` without unacceptable collision regression, plus a
secondary check that `productive_agent_fraction` doesn't regress) as every prior ablation this
investigation.

## Update 2026-08-27 (later still): AUX_DIR_RAMP matched comparison launched (5 Kaggle + 1 local)

Explicit user go-ahead received. Confirmed `git status` up to date with `origin` (nothing
pending), pushed the 2 local commits above first. Launched, identical structure to the two prior
ablations: `stage-marl-auxramp-control-s{1,2,3}` (`AUX_DIR_RAMP` unset -- current baseline,
`AUX_DIR_COEF=0.01` only) and `stage-marl-auxramp-treat-s{1,2}` (`AUX_DIR_RAMP=1.0`) on Kaggle --
all 5 confirmed `RUNNING`. Treatment seed 3 run locally (Kaggle's 5-session cap).

**Same checkpoint-reuse hazard as the last two launches, checked proactively again**: `run_id=3`
locally still held the *previous* local run's artifacts (the `AUX_DIVERSIFY=1` treatment-seed-3
run from the last ablation, fully consumed at 3,009,600 steps -- confirmed by reading the
checkpoint before touching anything) -- moved `checkpoints/latest_3.pt`/`models/actor_3.pt`/
`models/actor_best_3.pt`/`logs/training_log_3.csv`/`logs/eval_3.csv`/`logs/eval_best_3.csv` aside
(preserved, not deleted) before relaunching. Confirmed via `ps` (real CPU usage, elapsed time
advancing) that training is genuinely progressing, not exiting instantly. All 6 arms confirmed
running as of this update.

**Status**: launched, in progress. Same evaluation plan and acceptance criterion as the launch
plan above: download all 6 once complete, compare `target_lost_rate`, `contact_fraction`,
`success_rate`, `productive_agent_fraction`/count, `collision_rate` at both final and best
checkpoints (≥2/3 seeds improve `target_lost_rate` without unacceptable collision regression,
`productive_agent_fraction` should not regress -> keep; otherwise reject and move to the next
intervention).

## Open, not yet decided

- "Genetic/evolutionary" training as an alternative to PPO -- raised,
  not adopted; PPO is what's producing every result above.
- GPU was already measured slower than CPU for this network size
  (`DECISIONS.md`) -- not revisit unless the network architecture changes
  enough to change that math (e.g. the permutation-invariant encoder idea
  queued elsewhere). Phase 3's larger network doesn't change this
  conclusion on its own -- still small by GPU-dispatch-overhead standards.
