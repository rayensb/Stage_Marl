# Experiment Log

Chronological, including failures — the point of this file is to stop future sessions from
re-trying something already measured to fail. Reconstructed from git commit messages, inline
code comments (which record specific measured results), and files present in this session.
Anything not backed by a specific number/file is marked accordingly.

## Entropy floor raise (`ENT_COEF_END` 0.001 → 0.004) + extended schedule (`TOTAL_STEPS` → 3M)

**Objective**: stop entropy collapse (policy std shrinking too aggressively) by giving the
policy a higher entropy-coefficient floor and more steps to use it.
**Configuration**: `ENT_COEF_END=0.004`, `TOTAL_STEPS=3_000_000` (vs. baseline `0.001` /
`600_000`).
**Result** (recorded verbatim in the inline comment at `training/train.py:25-36`):
`collision_rate` plateaued at 0.78 for the last 2M steps — worse than any prior run — and
entropy still collapsed to -3.92 anyway.
**Conclusion**: a fixed higher floor coefficient does not stop entropy collapse; it just
gives the policy more time to commit harder to trading safety for precise tracking, because
the reward at the time genuinely permitted that trade. **Fixed-calendar entropy schedules
(of any shape) can't fix this class of problem alone** — this is what motivated the
plateau-triggered entropy-recovery mechanism instead (see `DECISIONS.md`).
**Status**: reverted — current code is back to `ENT_COEF_END=0.001` / `TOTAL_STEPS=600_000`.

## Entropy-recovery trigger v1: `entropy < 0`

**Objective**: automatically boost exploration when the policy stops improving.
**Configuration**: recovery triggered whenever logged policy entropy went negative.
**Result**: an `NUM_AGENTS=3` shared-actor run relapsed — `collision_rate` went from 0.02
back up to 0.25+ — **without entropy ever going negative**, so the trigger never fired
during the regression it was meant to catch.
**Conclusion**: entropy sign alone is not a reliable proxy for "the policy needs more
exploration" — a policy can relapse in collision rate while its entropy stays in a range
that looks fine. Replaced with a trigger based on the same rolling `collision_rate` window
everything else already uses (see `DECISIONS.md` for the v2 design: patience/cooldown/
max-triggers).
**Status**: superseded by the `collision_rate`-plateau version currently in `train.py`.

## Best-checkpoint-by-collision-rate-only produced a "safe by spreading out" policy

**Objective**: n/a — this was an observation made while reviewing a completed run's
best-vs-final checkpoint, not a designed experiment.
**Result**: the checkpoint selected purely by lowest `collision_rate` had meaningfully worse
`tracking_rmse` and `swarm_diameter` than that same run's own final-step model — i.e. it
achieved safety partly by spreading the formation out excessively rather than by better
collision avoidance at the intended spacing.
**Conclusion**: collision rate alone is an incomplete selection criterion. Replaced with the
lexicographic `(round(collision_rate, 2), -track)` criterion (see `DECISIONS.md`).
**Status**: fixed — `best_score` in `train.py` now uses the lexicographic criterion.

## `log_std` runaway → entropy explosion (commit `fc4c46f`)

**Objective**: n/a — bug fix, not a designed experiment.
**Symptom** (from commit message "Clamp log_std (fix entropy runaway)"): unbounded log_std
growth during training.
**Fix**: clamp `log_std` to `[LOG_STD_MIN=-2.0, LOG_STD_MAX=0.5]` in `Actor`.
**Status**: fixed, confirmed present in current `training/networks.py`. Full pre-fix
numeric evidence not available in this session (git history has the commit but this session
didn't inspect the diff in detail) — if precise before/after entropy numbers are needed,
check `git show fc4c46f`.

## Per-neighbor "diverge" penalty → global cohesion penalty (commit `78e1feb`)

**Objective**: fix a "relock loophole" where an agent could satisfy its per-neighbor
divergence constraint locally while the swarm's overall diameter drifted.
**Configuration change**: replaced per-neighbor diverge penalty with a single global
cohesion penalty based on swarm-wide spread; also rebalanced reward weights (track x4,
safety magnitude down) in the same commit.
**Result**: commit message states this fixes the relock loophole and swarm-diameter drift;
whether it introduced the cohesion/safety conflict fixed two commits later in `63274b1` is
not explicitly stated but is a plausible link worth checking if investigating that later fix
in depth.
**Status**: landed; the resulting cohesion/safety interaction was further adjusted by
`63274b1` (see `KNOWN_ISSUES.md` item 1 — that follow-up fix's effect on a real run is
unverified as of this writing).

## `NUM_AGENTS=3` shared-actor curriculum run(s) — evaluation data present, full training log not reviewed this session

**Objective**: validate the shared-actor CTDE-PPO architecture and reward config at
`NUM_AGENTS=3`, as the middle stage of the 2→3→4 curriculum.
**Evidence available this session**: two deterministic-evaluation CSVs were present in
`~/Downloads/` on the local machine — `eval_best_n3_2.csv` and `eval_best_n3_3.csv` (100
episodes each, columns: `episode, collided, episode_len, min_dist, tracking_rmse,
avg_spacing_std, avg_diameter, avg_speed`), evidently from `evaluate.py --best` for two
different seeds (`_2`, `_3`) of an `NUM_AGENTS=3` run.
**Quick skim (not a full analysis)**:
- `eval_best_n3_2.csv`: 4 collisions out of 100 episodes (episodes 47, 82, 83, 97), each
  terminating well before `MAX_STEPS=200` (lengths 18-37) — genuine early collisions.
- `eval_best_n3_3.csv`: 2 collisions out of 100 episodes (episodes 83, 89), lengths 25 each.
- Non-collision episodes run the full 200 steps in both files; `min_dist` values on
  non-collision episodes stay comfortably above `COLLISION_DIST=4.20` in the rows skimmed.
**Status**: **UNVERIFIED / informal** — this is a skim of two CSVs present in this session,
not a rigorous statistical comparison, and the corresponding *training* logs (not just
these evaluation CSVs) were not located or reviewed this session. If asked to properly
characterize the `NUM_AGENTS=3` results, locate the actual training CSV log(s)
(`logs/training_log_n3_*.csv` naming convention per `logger.py`) and the full evaluation
data, and compute proper aggregate statistics (mean/median `tracking_rmse`, collision rate
with confidence interval, etc.) rather than relying on this skim.

## `NUM_AGENTS=4`, 3-seed validation run — status pending

**Objective**: validate the full current fix chain (shared actor + best-checkpoint +
entropy-recovery + credit-fix + gradient-clipping + lexicographic-best-criterion +
the `a40db9b`..`63274b1` reward/stability chain) at the target `NUM_AGENTS=4`, across 3
seeds for robustness.
**Configuration**: reportedly launched on Kaggle using parallel `subprocess.Popen` per seed
with `OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`/`NUMEXPR_NUM_THREADS=1` pinning to avoid CPU
contention between the parallel seed runs.
**Result**: **PENDING** — not yet received as of the last known state of this project. This
is the single most important open item — see `SESSION_HANDOFF.md`.
**Status**: in progress / awaiting results.
