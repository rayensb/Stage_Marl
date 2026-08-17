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

## `NUM_AGENTS=4`, 3-seed validation run — deferred (at the time), see final entries below for how this resolved

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
**Status**: superseded — the `RECOVERY_MAX_TRIGGERS` fix itself turned out to be wrong (see
below); N=4 stayed deferred through several more iterations until the collision problem was
actually solved via a completely different mechanism (the closing-speed brake). See
`TODO.md`/`SESSION_HANDOFF.md` for current status — this is likely the next real work item.

## `RECOVERY_MAX_TRIGGERS` 5 → 20 — tested, falsified

**Objective**: test whether the collision-rate collapse above was caused by the
entropy-recovery mechanism's trigger budget running out mid-training (~rollout 110-125 at the
old cap of 5), leaving entropy unprotected for the remaining ~60% of a 600k-step run.
**Configuration**: `RECOVERY_MAX_TRIGGERS` raised 5 → 20, `NUM_AGENTS=3`, 3 seeds, 600k steps.
**Result**: `collision_rate` at `TOTAL_STEPS` was 27%/25%/66% — essentially unchanged or worse
than the pre-fix 30%/26%/46%, even though the entropy-coefficient trace confirmed recovery
was now firing across the whole run, not just the first ~250k. The raw policy entropy trace
barely moved despite far more frequent coefficient boosts.
**Conclusion**: the entropy coefficient isn't the binding lever on this failure mode —
matches the earlier-rejected `ENT_COEF_END=0.004` experiment reaching the same dead end a
different way (see above). **This falsified the "recovery budget" hypothesis.**
**Status**: reverted to `RECOVERY_MAX_TRIGGERS=5`.

## `r_safety` bonus-zone reshape (peak at `EDGE_TARGET` instead of below it) — tested, falsified

**Objective**: `r_safety`'s bonus previously peaked in `[SAFE_DIST_ENTER, SAFE_DIST_EXIT)`
(5.40-6.60), below the actual designed-safe formation edge (`EDGE_TARGET=7.80`) — nothing
rewarded sitting at the intended equilibrium, only a tighter, riskier band. Reshaped to a
triangular ramp peaking at `EDGE_TARGET`, decaying back to neutral one more `REACTION_DIST`
beyond it.
**Configuration**: `NUM_AGENTS=3`, single seed, 600k steps, alongside `K_NEIGHBORS =
NUM_AGENTS-1` (closes an observation/reward mismatch a supervisor review flagged, though a
no-op at N≤3) and the `RECOVERY_MAX_TRIGGERS` revert above, in the same commit (`e4af171` —
flagged later by an external review as three changes tested together, a fair critique of
this session's discipline at that point).
**Result**: final-model `collision_rate=40%`, no better than the pre-fix 30%.
**Conclusion**: where the safety bonus peaks wasn't the binding constraint either. Two
hypotheses now falsified in a row.
**Status**: kept in the codebase (it's not wrong, just not sufficient) — the actual fix ended
up being a completely different mechanism (see the closing-speed brake, below).

## Diameter floor v1 (`MIN_DIAMETER=10`, `DIAMETER_FLOOR_WEIGHT=-1.0`) + entropy-recovery retiming — tested, insufficient

**Objective**: `r_cohesion` only ever penalized the swarm being too *loose* — nothing pushed
back against diameter shrinking as `r_track` pulled harder over training. Added a symmetric
floor. Bundled in the same commit (`67ccfd7`): retimed `RECOVERY_PATIENCE` 15→29 /
`RECOVERY_MAX_TRIGGERS` 5→10 for ~10 uniformly-spread triggers instead of 5 clustered early,
and made `TOTAL_STEPS` env-var overridable.
**Configuration**: `NUM_AGENTS=3`, single seed, tested at both 600k and 1200k steps.
**Result**: diameter floor visibly engaged (nonzero `r_cohesion` contribution whenever
diameter dipped under 10) but was overwhelmed — diameter still collapsed to ~9.1-9.6 by the
end of both runs, `collision_rate` stayed in the same broken range (45%/34%), and — critically
— `entropy` went negative in **both** runs regardless of duration, with the 1200k run's final
entropy five times more negative than the 600k run's (more time = more time to decline
further, not evidence of a fix).
**Conclusion**: this is what motivated actually instrumenting `log_std` directly instead of
continuing to guess at reward weights (see next entry) — every intervention so far assumed
entropy collapse meant *exploration noise* collapsing, and none of them had ever actually
checked.
**Status**: `DIAMETER_FLOOR_WEIGHT` was later disabled entirely once the real cause was found
(see below) — not because this hypothesis was fully wrong, but because it turned out to be
fighting a *different* later requirement (sensor range) once vision-based tracking existed.

## `log_std_mean` / `mean_action_abs` instrumentation — the diagnostic that actually explained the collapse

**Objective**: distinguish two different possible causes of `entropy` (a derived `-logp`
Monte Carlo estimate) declining: `log_std` itself shrinking (real exploration-noise loss,
until it hits `LOG_STD_MIN=-2.0`) vs. the mean action increasingly saturating near tanh's
±1 boundary (which independently drives `-logp` more negative via the squashing-Jacobian
term, with `log_std` barely moving).
**Configuration**: `NUM_AGENTS=3`, single seed, 600k steps, diameter floor still active.
**Result**: `log_std` moved from -0.505 to -0.840 over the *entire* run — nowhere near its
floor of -2.0. `mean_action_abs` climbed steadily from 0.41 to 0.52. The raw closed-form
Gaussian entropy implied by that `log_std` change would only have dropped ~1.0; the logged
`entropy` metric dropped ~1.9 — the gap is the tanh-squashing correction responding to
increasing action saturation, not exploration noise.
**Conclusion**: **every entropy-side intervention tried so far (recovery budget, recovery
timing, entropy floor) was aimed at the wrong variable.** The policy was learning to commit
to larger-magnitude actions over training — a drone closing at high commanded speed has less
room to back out of a close encounter than one making smaller corrections, independent of
stochastic noise. This reframed the whole problem from "exploration schedule" to "action
magnitude near danger."
**Status**: directly motivated the closing-speed brake (next entry) — the first fix in this
whole chain grounded in an actual measurement of what was changing, not another guess.

## Closing-speed brake — deterministic action-space safety layer — **succeeded**

**Objective**: test whether capping action magnitude specifically *near other agents*, rather
than trying to teach the policy to self-moderate via reward, would stop the collision-rate
relapse that five reward-shape/schedule attempts in a row had failed to fix.
**Configuration**: in `envs/formation_env.py`'s `step()`, cap only the *closing* velocity
component (the part of commanded velocity actually shrinking the gap to a specific other
agent) once already inside `SAFE_DIST_ENTER`, ramping linearly to zero exactly at
`COLLISION_DIST`. Lateral/evasive motion untouched; a no-op outside `SAFE_DIST_ENTER`. Reuses
only already-derived constants (`SAFE_DIST_ENTER`, `COLLISION_DIST`, `MAX_ACTION_SPEED`), no
new ones. Doesn't touch the reward at all — `r_safety`'s urgent penalty still applies in
full, so the policy retains every incentive to avoid the zone; this is a safety net for once
it's already there.
**Result**: **`collision_rate=0.000` for the entire 600k-step training run** (not just at
eval — every single one of 293 logged rollout checkpoints), replicated across seed 2, seed 3,
and a 1200k-step variant — 6/6 configurations at exactly 0%. `mean_action_abs` still climbed
over training (same as every prior run — that was never the thing to fix), but with the brake
in place it stopped mattering. `tracking_rmse` also improved (3.06-4.25 at 600k → 2.31-2.46
at 1200k) since duration was no longer coupled to collision risk.
**Conclusion**: the collision-rate collapse that persisted through commits `78e1feb` through
`67ccfd7` (six prior fix attempts) is resolved. It was never a reward-balance problem.
**Status**: landed (`0b6278d`), then a genuine `r_collision` logging bug was found and fixed
separately (`09e3038` — every episode's terminal step, the only step `r_collision` is ever
nonzero, was being silently excluded from the logged component due to a stale `env.agents`
check; training itself was unaffected, only that one logged column was blind). One rare
caveat found later at longer training durations: a single isolated collision event occurred
in a 3M-step run (1/100 eval episodes elsewhere) — the brake's no-crossing proof was only
worked out rigorously for two agents at a time; simultaneous multi-threat braking (an agent
close to two others at once) doesn't have the same guarantee. Rare (~1%), not systemic, but
real — see `KNOWN_ISSUES.md`.

## Vision-based cooperative target tracking — replacing ground-truth telemetry

**Objective**: every drone previously knew the target's exact position/velocity every step
regardless of distance — an explicit, deliberate simplification the project had documented
but never actually revisited. Replaced with: a direct reading only when the target is within
`SENSOR_RANGE` (omnidirectional — no heading/FOV-cone state exists in this sim), shared
instantly across the swarm whenever any drone has contact, dead-reckoned for up to
`LOST_TIMEOUT_STEPS` (2s) if no one currently does, with episode termination (`target_lost`)
beyond that. New observation features: `has_direct_contact`, `confidence`, track `age`,
`observer_count`, relative direction to the centroid of currently in-contact teammates (for
cooperative recovery). New `r_contact` reward component (same shape as `r_safety`'s urgent
ramp). `r_track`/`r_velocity` switched to scoring against the tracked *estimate*, never
ground truth directly — avoiding a second instance of the exact observation/reward mismatch
class a supervisor review caught for `r_safety` earlier. Deliberately deferred to keep this
one change controlled: UWB-realistic neighbor sensing, sensing noise/occlusion, a directional
FOV cone, non-constant-velocity target motion.
**Configuration 1 (diameter floor still active from the prior entry)**: `NUM_AGENTS=3`,
single seed, 600k steps.
**Result 1**: `collision_rate` stayed at 0% (brake unaffected, as expected — it's a separate,
independent mechanism). But `target_lost_rate` was **31% (final) / 40% (best)** — a new
failure mode at a severity comparable to the original collision problem. Training curve showed
`target_lost_rate` oscillating 15-40% with no improving trend across the full run, alongside
`swarm_diameter` pinned at 12-15 (vs. 10-13 before vision-tracking existed) and `r_track`
notably worse than earlier runs (-4.7 to -6.8 vs. -2.3 to -3).
**Hypothesis**: the diameter floor (from the earlier, pre-brake entry above) was built to
stop dangerous tight convergence before the brake existed. The brake now handles that
deterministically, independent of diameter — so the floor might now just be pushing the swarm
wider than `SENSOR_RANGE` allows, for a safety problem it's no longer needed to solve that way
(a wider ring geometrically implies a larger individual radius from the target for
evenly-spaced agents, same relation `TARGET_DIST`/`EDGE_TARGET` already uses).
**Configuration 2**: `DIAMETER_FLOOR_WEIGHT` set to 0.0 (`b59c139`, not deleted — a one-line
revert if wrong), otherwise identical, 600k steps.
**Result 2**: `target_lost_rate` roughly halved — **18% (final) / 14% (best)**, and for the
first time showed a real (if noisy) improving trend over training (0.14-0.22 early → 0.04-0.18
later), instead of the flat non-improving oscillation with the floor active. `tracking_rmse`
unchanged (no regression from removing the floor). **Hypothesis confirmed.**
**Configuration 3**: same (no floor), `TOTAL_STEPS=1_200_000`.
**Result 3**: further improvement — `target_lost_rate` **11% (final) / 4% (best)**,
`tracking_rmse` **2.57/2.58**. Not a clean monotonic trend (noisy middle third, `lost_rate`
briefly back up to 0.28-0.32 around steps 120-180k and 477-600k) before converging in the
final ~20% of training. This run is also where the one rare collision event mentioned above
occurred (1 episode, isolated).
**Configuration 4 (decisive)**: same (no floor), `TOTAL_STEPS=3_000_000`, **3 seeds**.
**Result 4**:

| seed | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) | diameter |
|---|---|---|---|---|
| 1 | 0% / 0% | 0% / 3% | 2.18 / 2.35 | 9.9-10.2 |
| 2 | 0% / 0% | 1% / 3% | 2.99 / 2.73 | 10.8-11.0 |
| 3 | 0% / 1% | 0% / 0% | 1.52 / 1.52 | 9.4-9.6 |

`target_lost_rate` progression across the whole investigation: 31-40% (600k, floor) → 18-14%
(600k, no floor) → 11-4% (1200k, no floor) → **0-3% (3M, no floor, 3 seeds)**. Training curves
show `target_lost` concentrated entirely in the first ~700k-1M steps, then genuinely zero for
the remaining ~2M+ steps in every seed — sustained convergence, not a lucky endpoint.
`tracking_rmse` (1.52-2.99) is the best seen all session, beating even the pre-vision-tracking
numbers. The one collision reading (seed 3 best, 1%) is consistent with the same rare
multi-agent brake edge case noted above, not a new problem.
**Conclusion**: the diameter floor was the blocker, and duration was the remaining lever —
exactly the same two-part story as the earlier (pre-vision-tracking) tracking-improvement
result, for a related reason (both times, removing a stale safety-era constraint plus more
training time let a decoupled objective actually converge).
**Status**: merged to `main` (`b59c139`) after this 3-seed validation. This is now the current
state of the project — see `CURRENT_STATE.md`.
