# Current State

As of commit `77b79b3` on `phase3-resilience`, 2026-08-27. `origin/main` is at `8b724bb`
(`phase2-combined`'s reference point) — **fast-forwarded 2026-08-23**, no longer behind; this
corrects a stale claim ("`main` still at `b59c139`") that persisted across several doc passes
after the push actually happened. Local `main` in this worktree may still show an older commit
until someone checks it out and pulls — that's normal local-ref staleness, not a real
discrepancy (confirmed: local `main` is an ancestor of `origin/main`). Working tree has a few
untracked `deployment/`-side files (not this doc suite's concern — see `ARCHITECTURE.md`).

## Confirmed working (verified against completed, multi-seed-replicated Kaggle runs)

- **Collision avoidance is solid at `NUM_AGENTS=4`.** The closing-speed brake's multi-pass
  POCS-style convergence plus a thresholded brake-engagement reward penalty remain the validated
  fix for the original simultaneous-multi-threat gap — 3/3 `N=4` seeds clean at 0% `collision_rate`
  in the Phase2-combined validation, both at eval-time and in the training-time rolling-window
  picture. **The brake's relative-velocity reformulation was tried and reverted** — see "Known
  broken" below.
- **3D (XYZ) formation spread** and **N-aware safety margin** — unchanged, still validated, no
  new evidence either way this pass.
- **Vision-based cooperative target tracking**, mechanism unchanged since the original redesign:
  sensor-range-limited detection, swarm-shared contact, dead-reckoning, `target_lost`
  termination. `LOST_TIMEOUT_SEC` default is `6.0` (resolved after a 6/10/18s sweep found no
  clean dose-response).
- **Active search**, validated as a real, major improvement over the plain-timeout approach:
  `contact_fraction` ~16-21% → 85-90.7% at 6s, and up to 99%+ contact / ~2-4% `target_lost_rate`
  for seeds that converge well at 8-10s.
- **Search-direction auxiliary objective (`AUX_DIR_COEF=0.01`), the new default as of
  2026-08-27.** A small actor-loss term nudging search direction toward `unit(_last_known_vel)` —
  the first intervention in the whole `target_lost` investigation to measurably move the primary
  failure metric. 3/3 seeds improved on `target_lost_rate` at both final and best checkpoints
  (e.g. 0.83/0.86/0.98 → 0.26/0.14/0.38 final), with `success_rate` and `productive_agent_fraction`
  up in every seed and only a small collision-rate cost (0-3 points). Pooled new-baseline audit:
  26.0% `target_lost` remains — late, first-loss-event-fatal failures, not early collapse. Two
  follow-on ideas (`AUX_DIVERSIFY`, `AUX_DIR_RAMP`) were tested and **both rejected** — see "Known
  broken" below.
- **Ground awareness, per-axis (Z vs XY) action scaling, cruise-altitude preference, and
  Z-velocity smoothing** — all unchanged, still validated (0 ground strikes across every Phase 3
  validation rollout; jiggling fix still not re-measured, see "Unverified").
- **Network capacity at `NUM_AGENTS=4`, now actually measured, not just a starting guess.** A
  5-way hidden-width sweep (`ACTOR_HIDDEN`/`CRITIC_HIDDEN`, env-overridable since 2026-08-24)
  confirmed the Phase 3 default (128/256) is a genuine sweet spot: a smaller network (64/128)
  partially degrades tracking (45% `target_lost`), a larger one (256/512) collapses learning
  almost entirely (97% `target_lost`, flat from the first rollout). Defaults unchanged. See
  `EXPERIMENT_LOG.md`.
- **Critic-health instrumentation** (`CRITIC_LR` env override + per-rollout diagnostic columns in
  `training_log.csv`, commit `71327be`) — works as built, logs correctly, and was the tool that
  let the critic-saturation investigation (below) actually measure what it needed to.
- **Shared-actor CTDE-PPO training loop**, **curriculum infra**, and **evaluation/diagnostics**
  (contact/loss-streak tracking, reacquisition-time distribution, pooled tracking-error
  percentiles, `diagnose_horizon.py`) — all unchanged, still working as designed.

## Known broken / open problems

**Headline: `target_lost_rate` has its first confirmed fix (`AUX_DIR_COEF`), but 26% remains and
the two most-evidence-motivated follow-ons both failed.** The root cause (localized last pass to
the learned actor's own search execution — a scripted controller reacquires 100% of the time from
states where PPO's own trajectory failed 100% of the time, holding environment/geometry/timeout/
starting-state exactly fixed) is unchanged; what's new this pass is the first working intervention
against it. `AUX_DIR_COEF=0.01` — a small actor-loss term nudging search direction toward
`unit(_last_known_vel)` — improved `target_lost_rate` in 3/3 seeds at both checkpoints (e.g.
0.83/0.86/0.98 → 0.26/0.14/0.38 final) and is now the default. A pooled audit of the new baseline
found 26.0% `target_lost` remains, concentrated in late, first-loss-event-fatal failures (67.9% of
failures die on the swarm's very first loss event, `contact_fraction` still ~0.865 at failure) —
not early collapse. Two follow-on ideas built directly from that audit were tested and **both
rejected**: `AUX_DIVERSIFY` (per-agent heading-diversity blend) came back mixed (1 up, 1 down
+0.18, 1 flat); `AUX_DIR_RAMP` (urgency-scaled coefficient) came back as a clean **3/3-seed
regression** (+0.37/+0.08/+0.23 percentage points), worse than the mixed result. Neither is
adopted; `AUX_DIR_COEF=0.01` (flat) remains the baseline. **No further fix direction has been
chosen** — closing the remaining 26% is still the single most important open problem in the
project, and the two most-motivated next ideas from existing evidence have both been tried and
failed. See `EXPERIMENT_LOG.md`'s decisive counterfactual and AUX-intervention entries, and
`KNOWN_ISSUES.md` item 12.

**The relative-velocity brake fix was tested at full scale and reverted — net-harmful, cause
unknown.** The collision-safety cost that motivated it (8-14% under deterministic execution on
well-converged checkpoints) is still real and still unaddressed by any current mechanism, but the
fix built for it broke training convergence itself on 6 of 7 tested seeds (including a clean
3/3→0/3 flip on 3 seeds with direct old-brake baselines). The leading hypothesis for why was
tested directly and refuted; the actual mechanism was never found. Reverted to the original
one-sided formula (commit `55e7232`) given deployment-readiness was the more urgent priority. See
`DECISIONS.md`, `KNOWN_ISSUES.md` item 13.

**New: `NUM_AGENTS=3` cannot learn tracking at all under Phase 3's mechanism set, at any tested
network size — a real deployment blocker.** The network-capacity sweep's two `N=3` kernels (the
first attempt at `N=3` with a *working*, reverted brake) both came back flat at ~100%
`target_lost_rate` from the very first rollout, at both tested hidden widths. `deployment/
inference_node.py` hard-requires `NUM_AGENTS=3`, so no checkpoint from anywhere in the Phase 3
era is currently deployable. Not yet under active investigation — see `KNOWN_ISSUES.md` item 18.

**Critic saturation: confirmed real, demoted to a secondary pathology.** The central critic's
final Tanh layer saturates 100% within the first rollout of training, collapsing value
predictions to a near-constant — real, reproducible, and delayable (not preventable) with a
30x-smaller critic learning rate. But a matched, full-scale ablation of that fix found saturation
returns by the end of a full run regardless, and behavioral results were a mixed bag, not a clean
win. Rejected as a general fix; `CRITIC_LR`'s default is unchanged. See `DECISIONS.md`,
`KNOWN_ISSUES.md` item 19.

**Two seeds resumed to 5M steps to test "does more training help" (unchanged from last pass) —
partially yes.** The two `t10`-bracket seeds that got stuck in a bad tracking regime improved
substantially when resumed to 5M steps (70-72%→41% and 92-96%→83% `target_lost_rate`) but neither
closed the gap to well-converged seeds' ~2-4%. Superseded in priority by the actor-localization
finding above — more training was never going to fix a specific execution problem in the learned
policy.

**Secondary: vertical jiggling — fixed in code, still not re-verified.** Unchanged from every
prior pass — nobody has yet loaded a post-fix checkpoint and re-measured the sign-flip rate. See
`KNOWN_ISSUES.md` item 14.

## Unverified (implemented, not yet confirmed by a completed run)

- The vertical-jiggling fix (still not re-measured against a trained checkpoint, unchanged from
  every prior pass).
- Whether any fix exists for the actor-search-execution problem — none has been attempted yet,
  so there's nothing to verify.
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
comparing across runs, given how many distinct configurations now exist in this project's
history (pre-brake, brake-only, vision-tracking with/without floor, Phase 1, Phase 1b,
N-aware-margin, xyz-spread, Phase2-combined, Phase 3, the timeout/active-search sweeps, both
brake-relvel batches, the network-capacity sweep, the critic-LR ablation). Phase2-combined's best
`N=4`/`N=3`/`N=2` checkpoints are saved as `actor_best_{1,2,3}_20260820-1255_3f9069c-n4-5m-phase2-
seed{1,2,3}.pt`, `actor_best_n3_1_...`, `actor_best_n2_1_...`. **No deployment-ready
(`NUM_AGENTS=3`, Phase-3-era) checkpoint exists as of this writing** — see "Known broken" above.

## Local environment caveat

Unchanged: this sandbox's CUDA is unavailable, `DEVICE` always falls back to `cpu` here,
matching the project's measured-optimal default anyway. Kaggle API access from this machine
(`~/.kaggle-venv`) continues to work — see `ENVIRONMENT.md` for the account-username and
dataset-mount-path gotchas.
