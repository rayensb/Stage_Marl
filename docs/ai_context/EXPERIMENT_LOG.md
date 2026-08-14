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

## `NUM_AGENTS=3` shared-actor curriculum run, 3 seeds — full analysis (supersedes the earlier skim below)

**Objective**: validate the shared-actor CTDE-PPO architecture and reward config at
`NUM_AGENTS=3`, as the middle stage of the 2→3→4 curriculum, using the full
`a40db9b`..`63274b1` fix chain.
**Evidence**: complete 3-seed dataset (`training_log_n3_{1,2,3}.csv`,
`training_curves_n3_{1,2,3}.png`, `eval_n3_{1,2,3}.csv`, `eval_best_n3_{1,2,3}.csv`, all 100
episodes) — a fuller set than what an earlier version of this entry recorded (that version
only found 2 of the 6 eval CSVs; all 6 plus all 3 training logs turned out to be present).
**Final-model results (deterministic eval, 100 ep, `evaluate.py` without `--best`)**:

| seed | collision_rate | avg collision step | tracking_rmse | swarm_diameter |
|---|---|---|---|---|
| 1 | 30% (30/100) | 39.5 | 2.36 | 10.31 |
| 2 | 26% (26/100) | 51.4 | 2.00 | 9.65 |
| 3 | 46% (46/100) | 67.2 | 2.00 | 9.57 |

**Best-checkpoint results (`evaluate.py --best`)**:

| seed | collision_rate | tracking_rmse | swarm_diameter |
|---|---|---|---|
| 1 | 8% (8/100) | 3.81 | 12.48 |
| 2 | 4% (4/100) | 3.69 | 12.70 |
| 3 | 2% (2/100) | 3.74 | 12.15 |

**Training-curve pattern (identical across all 3 seeds)**: `collision_rate` drops near 0 by
~step 100-150k, holds through ~step 220-250k, then climbs steadily back to 26-46% by
`TOTAL_STEPS=600k`. This is a **confirmed relapse of the exact failure `63274b1` targeted**,
not noise — see `KNOWN_ISSUES.md` item 1 for the root-cause diagnosis (entropy-recovery
budget exhaustion, `RECOVERY_MAX_TRIGGERS=5` spent by rollout ~110-125 in every seed,
matching precisely where each seed's recovery-active window ends and collision_rate starts
climbing) and the fix applied (`RECOVERY_MAX_TRIGGERS` 5 → 20 in `training/train.py`).
**Conclusion**: the `a40db9b`..`63274b1` chain did not fix the collapse; it's an
entropy-recovery scheduling bug, not a reward-balance problem. `--best` checkpointing works
exactly as designed as a safety net (rescues collision_rate to 2-8%) but at a real
tracking/diameter cost, which is what motivated fixing the underlying budget instead of
relying on `--best` alone.
**Status**: analysis complete; the `RECOVERY_MAX_TRIGGERS` fix is applied but itself
**unverified** — needs a fresh single-seed `NUM_AGENTS=3` run to confirm before spending a
3-seed `NUM_AGENTS=4` Kaggle batch on it.

## `NUM_AGENTS=3` shared-actor curriculum run(s) — earlier informal skim (superseded above)

**Evidence available in an earlier session**: two deterministic-evaluation CSVs —
`eval_best_n3_2.csv` and `eval_best_n3_3.csv`.
**Quick skim (not a full analysis)**:
- `eval_best_n3_2.csv`: 4 collisions out of 100 episodes (episodes 47, 82, 83, 97), each
  terminating well before `MAX_STEPS=200` (lengths 18-37) — genuine early collisions.
- `eval_best_n3_3.csv`: 2 collisions out of 100 episodes (episodes 83, 89), lengths 25 each.
**Status**: superseded by the full 3-seed analysis above — kept here only as a record that
the earlier skim's numbers (4/100 and 2/100 for seeds 2/3) match the rigorous best-checkpoint
results above (4% and 2%), i.e. the skim wasn't wrong, just incomplete (missing seed 1, the
non-best evals, and the training logs).

## `NUM_AGENTS=4`, 3-seed validation run — deliberately deferred, not launched

**Objective**: validate the full current fix chain at the target `NUM_AGENTS=4`, across 3
seeds for robustness.
**Configuration**: a Kaggle launcher script (parallel `subprocess.Popen` per seed,
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`/`NUMEXPR_NUM_THREADS=1` pinning, `NUM_AGENTS` left
unset so `config.py`'s default of 4 applies) was prepared and ready to run.
**Decision**: **not launched** on the pre-fix code. The `NUM_AGENTS=3` 3-seed analysis above
found the collision-rate collapse is caused by `RECOVERY_MAX_TRIGGERS` exhaustion, a bug that
would very likely reproduce (probably worse, given `NUM_AGENTS=4`'s tighter Tammes-packing
margin and noisier collision-credit attribution per `DECISIONS.md`) at `NUM_AGENTS=4` too.
Launching the 3-seed batch pre-fix would have spent Kaggle compute reconfirming a known
issue instead of testing something new.
**Status**: queued behind a single-seed `NUM_AGENTS=3` verification of the
`RECOVERY_MAX_TRIGGERS` fix (see entry above).
