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

## Open, not yet decided

- "Genetic/evolutionary" training as an alternative to PPO -- raised,
  not adopted; PPO is what's producing every result above.
- GPU was already measured slower than CPU for this network size
  (`DECISIONS.md`) -- not revisit unless the network architecture changes
  enough to change that math (e.g. the permutation-invariant encoder idea
  queued elsewhere). Phase 3's larger network doesn't change this
  conclusion on its own -- still small by GPU-dispatch-overhead standards.
