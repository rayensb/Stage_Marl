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

## `NUM_AGENTS=4`, 3-seed, 3M-step validation — tracking generalizes cleanly, collision does not

**Objective**: validate the full current fix chain (brake + vision-tracking) at the actual
target agent count, `NUM_AGENTS=4`, now that both problems were resolved and validated at
`N=3`.
**Configuration**: same launcher pattern as the `N=3` 3-seed/3M run (parallel per-seed Kaggle
kernels), `NUM_AGENTS=4` (config.py's default), commit `45b42a2`.
**Eval-time results (100 ep, deterministic)**:

| seed | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) |
|---|---|---|---|
| 1 | 0% / 1% | 1% / 2% | 2.25 / 2.37 |
| 2 | 1% / 1% | 3% / 5% | 2.96 / 2.88 |
| 3 | 2% / 0% | 0% / 0% | 1.98 / 1.89 |

**Training-time rolling-window results (the real picture — 50-episode window, ~1465 logged
rollouts per seed, a much larger sample than the 100-episode eval)**: 88/1465 (seed1), 59/1465
(seed2), 269/1465 (seed3) rollouts showed nonzero `collision_rate`, scattered across the
*entire* 3M-step run in every seed rather than converging — seed3 specifically still nonzero
(0.02-0.04) in its last 10 logged rollouts, through step 3,000,320. `target_lost_rate` by
contrast converges cleanly in the first ~0.7M-1M steps and stays at genuine zero for the
remaining ~2M+ steps, in every seed — the same clean pattern `N=3` showed. `mean_brake_reduction`
confirms the mechanism: continuous, substantial engagement throughout the whole `N=4` run
(0.002-0.016) vs. `N=3`'s sparse, occasional engagement (0.0001-0.0069 on the same metric).
**Conclusion**: the eval-time numbers (0-2% collision) understate the real severity — the
brake's no-crossing guarantee (proven only for two agents at a time, see `KNOWN_ISSUES.md` item
8) is being exercised much harder at `N=4`, where each agent has 3 simultaneous "others"
(`K_NEIGHBORS=NUM_AGENTS-1`, full connectivity) instead of 2. Tracking, by contrast, transfers
zero-shot with no changes needed — the vision-tracking design and reward generalize across `N`
cleanly; only the deterministic safety layer's coverage does not.
**Status**: not fixed yet. Two fixes proposed (multi-pass brake convergence, direct
brake-engagement reward penalty) — see `TODO.md` Phase 1. **This is now the project's headline
open problem** — see `CURRENT_STATE.md`.

## Geometric feasibility check: is `N=4`'s target formation (`TARGET_DIST=4.78`, `EDGE_TARGET=7.80`) actually simultaneously satisfiable?

**Objective**: raised directly — maybe the collision persistence at `N=4` is because the target
numbers themselves are geometrically inconsistent (can't really have every drone 4.78 from the
target *and* 7.80 from every other drone at once).
**Check**: `_PACKING_RATIO[4] = (8/3)**0.5 ≈ 1.633` is exactly the circumradius-to-edge ratio of
a **regular tetrahedron**, and `TARGET_DIST` is solved backward from `EDGE_TARGET` using that
exact ratio (`config.py`). A regular tetrahedron with circumradius 4.78 and edge 7.80 is a real,
exact geometric solid — both constraints hold simultaneously, no contradiction. Same check for
`N=3` (equilateral triangle on a great circle): also exact.
**Conclusion**: the target geometry is not infeasible. What's actually different at `N=4` is
that its exact solution is inherently non-planar (no 4 points can be simultaneously
center-equidistant and pairwise-equidistant while coplanar), while `r_spread` — the only reward
term with any angular-arrangement shaping — is horizontal-only and can't see or reward the
vertical structure that solution requires. `N=3`'s exact solution is planar, which is exactly
what horizontal-only `r_spread` can already shape — so this asymmetry didn't matter until `N=4`.
See `KNOWN_ISSUES.md` item 5 (updated) and `TODO.md` Phase 2 (XYZ `r_spread`, proposed, not yet
implemented) for the follow-up.
**Status**: verified finding, not yet acted on — a plausible contributing factor to the `N=4`
collision problem, not confirmed as sufficient on its own (item 8's sequential-correction
mechanism is independently sufficient to explain it).

## Phase 1 — multi-pass brake convergence + linear brake-engagement penalty (v1) — N=4, 3 seeds, 3M steps

**Objective**: directly test the two fixes proposed for the confirmed `N=4` collision
persistence (`KNOWN_ISSUES.md` item 8): sweep the brake's per-neighbor correction to
convergence (POCS-style) instead of one sequential pass, and add `r_brake = BRAKE_PENALTY_COEF
* brake_reduction` (linear from zero) so the reward directly penalizes needing the brake, not
just proximity.
**Configuration**: `NUM_AGENTS=4`, 3 seeds, 3M steps each (kernels
`stage-marl-n4-phase1-seed{1,2,3}`).
**Result** (eval-time, 100 episodes each):

| seed | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) | avg_diameter (final/best) |
|---|---|---|---|---|
| 1 | 0% / 0% | 1% / 0% | 3.15 / 3.21 | 13.36 / 13.38 |
| 2 | 0% / 0% | 1% / 2% | 2.55 / 2.88 | 12.08 / 12.58 |
| 3 | 0% / 0% | 1% / 1% | 2.50 / 2.61 | 11.05 / 11.32 |

**Training-time rolling-window picture** (1465 rollouts/seed, 3,000,320 final step in every
seed — confirmed complete, not truncated): nonzero-`collision_rate` rollouts dropped from the
pre-fix baseline's 88/59/269 (out of 1465, per `KNOWN_ISSUES.md` item 8's original data) to
**14/10/15** — a real, large reduction. All three seeds' last 10 rollouts were clean
(`collision_rate=0` throughout), unlike the pre-fix run where seed3 was still nonzero at the
very end of training.
**Conclusion**: the combined fix worked on the metric it targeted — collision, both at
eval-time and (more importantly, given eval-time numbers previously understated the problem)
in the training-time rolling-window picture. But `avg_diameter` (11.05-13.36) ran noticeably
wider than later fixes achieve (see Phase 1b below), and `tracking_rmse` (2.50-3.15) is worse
than Phase2-combined's eventual `N=4` numbers. Diagnosed as the linear-from-zero brake penalty
taxing trivial, routine engagement the same as genuine emergency corrections, pushing the
policy toward avoiding the brake's trigger zone altogether (i.e. spreading out) rather than
just avoiding needing a large correction inside it — see `DECISIONS.md`.
**Status**: superseded by Phase 1b (below), which fixes the diameter/tracking side effect
without giving up the collision fix.

## Phase 1b — thresholded brake-engagement penalty (v2) — N=4, 3 seeds, 3M steps

**Objective**: test whether only penalizing brake engagement *above* a derived threshold
(`BRAKE_PENALTY_THRESHOLD = 0.5 * MAX_ACTION_SPEED`), rather than linearly from zero, fixes
Phase 1's diameter/tracking regression while keeping its collision fix.
**Configuration**: identical to Phase 1 except `r_brake`'s v2 formula (see `DECISIONS.md`).
`NUM_AGENTS=4`, 3 seeds, 3M steps each (kernels `stage-marl-n4-phase1b-seed{1,2,3}`).
**Result** (eval-time, 100 episodes each):

| seed | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) | avg_diameter (final/best) |
|---|---|---|---|---|
| 1 | 0% / 0% | 0% / 0% | 2.25 / 2.27 | 10.88 / 10.84 |
| 2 | 0% / 0% | 1% / 2% | 2.70 / 2.69 | 12.32 / 12.27 |
| 3 | 0% / 0% | 0% / 0% | 2.12 / 2.15 | 11.57 / 11.60 |

Averaged across seeds: `tracking_rmse` 2.36 final / 2.37 best (down from Phase 1's 2.73/2.90),
`avg_diameter` 11.59 final / 11.57 best (down from Phase 1's 12.16/12.43) — **the threshold
change achieved exactly what it was meant to**, a real, measurable improvement in both
tracking and formation tightness.
**Training-time rolling-window picture** (1465 rollouts/seed, 3,000,320 final step in every
seed): nonzero-`collision_rate` rollouts were **30/31/19** — a counterintuitive increase from
Phase 1's 14/10/15, despite Phase 1b's clearly better diameter/tracking. Reported honestly,
not smoothed over: both are drastically better than the pre-fix 88/59/269 baseline, and all
three Phase 1b seeds' last-10 rollouts were still clean (`collision_rate=0`), so this is
*more transient collision events somewhere earlier in training*, not a failure to converge by
the end. The trade being made is legible — a small amount of additional transient training-time
collision risk for a real, measured gain in tracking/diameter quality — not a regression
disguised as an improvement.
**Conclusion**: adopted as the reference brake configuration — the diameter/tracking
improvement is worth the small increase in mid-training transient collision events, given both
configurations converge cleanly by the end and eval-time collision stayed at 0% throughout.
**Status**: landed, carried forward into Phase2-combined (below).

## N-aware safety margin — standalone hypothesis test, N=4, 1 seed

**Objective**: test `EDGE_TARGET`'s N-aware scaling (`+1.20` margin at `N=4`) in isolation,
on top of the already-validated Phase 1b brake fixes, before combining it with the XYZ
`r_spread` hypothesis.
**Configuration**: `NUM_AGENTS=4`, 1 seed, 3M steps (kernel `stage-marl-n4-margin-seed1`,
branch `n-aware-margin`).
**Result** (eval-time, 100 episodes): `collision_rate` 0% final / 1% best, `target_lost_rate`
0%/0%, `tracking_rmse` **1.69 final / 1.60 best** — notably better than either Phase 1 or
Phase 1b's tracking, `avg_diameter` 11.21/11.01. Training-time: 10/1465 nonzero-collision
rollouts, last 10 rollouts clean.
**Conclusion**: no evidence of the feared failure mode (a wider formation pushing drones past
`SENSOR_RANGE` more than necessary, the same mechanism that made the old diameter floor
counterproductive) — if anything, tracking improved. Single seed, not a full validation on its
own, but a clean enough result to justify combining with the spread hypothesis.
**Status**: folded into Phase2-combined (below) for full multi-seed validation.

## XYZ (true 3D) `r_spread` — standalone hypothesis test, N=4, 1 seed (uses the *pre-fix*, incorrect angle)

**Objective**: test the 3D pairwise-angle `r_spread` replacement in isolation, on top of the
Phase 1b brake fixes.
**Configuration**: `NUM_AGENTS=4`, 1 seed, 3M steps (kernel `stage-marl-n4-spread-seed1`,
branch `xyz-spread`). **Important caveat, confirmed by timeline**: this run used the
*original, incorrect* `_IDEAL_NEIGHBOR_ANGLE` (109.47°/120°, the global target-viewpoint
angle) — the local/global conflation bug (see `DECISIONS.md`) wasn't caught until later.
**Result** (eval-time, 100 episodes): `collision_rate` 0%/0%, `target_lost_rate` 0%/0%,
`tracking_rmse` 2.02 final / 2.19 best, `avg_diameter` 10.76/11.11. Training-time: 10/1465
nonzero-collision rollouts, last 10 clean.
**Conclusion**: no obvious harm from the wrong angle at this scale — the run is clean on every
metric. This does **not** mean the angle bug didn't matter; it means a single-seed test wasn't
sufficient to reveal a subtler cost (the corrected-angle isolated retest, `spread_fixed_seed1`
below, is the real check of whether the bug cost anything measurable).
**Status**: superseded by the corrected-angle version, folded into Phase2-combined for
validation before the bug was caught, then independently re-tested after the fix (see below).

## Phase2-combined — 5-seed N-sweep (all four Phase 1/2 fixes together, N=2/3/4, 5M steps)

**Objective**: full multi-seed validation of the combined reference point: multi-pass brake +
thresholded brake penalty (Phase 1b) + N-aware margin + XYZ `r_spread` — merged to branch
`phase2-combined`. **Note**: this run used the *pre-fix* spread angle (109.47°/120°) — the
angle bug was found and fixed (`ac98d67`) after this run had already launched; see the
isolated re-test (`spread_fixed_seed1`) below for whether that mattered.
**Configuration**: 5 seeds total, 5M steps each — `N=2` (1 seed), `N=3` (1 seed at 5M, plus a
second `N=3` seed at 1.5M specifically to check whether seed 1's `target_lost` noise was real
or seed-variance — see `phase2_n3_seed2` below), `N=4` (3 seeds).
**Result** (eval-time, 100 episodes each):

| run | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) | avg_diameter (final/best) |
|---|---|---|---|---|
| N=2 seed1 | 0% / 0% | 0% / 1% | 2.25 / 1.72 | 8.14 / 7.33 |
| N=3 seed1 | 0% / 0% | 5% / 1% | 2.88 / 2.08 | 11.31 / 10.35 |
| N=4 seed1 | 0% / 0% | 0% / 0% | 1.43 / 1.54 | 10.43 / 10.69 |
| N=4 seed2 | 0% / 0% | 0% / 0% | 1.97 / 1.97 | 11.67 / 11.67 |
| N=4 seed3 | 0% / 0% | 0% / 0% | 2.32 / 2.31 | 13.10 / 12.94 |
| **N=4 avg** | 0% / 0% | 0% / 0% | 1.91 / 1.94 | 11.73 / 11.77 |

(N=4 seed2's final and best rows are identical because `eval_2.csv`/`eval_best_2.csv` are
byte-identical for that run — the two checkpoints evaluated the same on this particular
100-episode set, confirmed via checksum, not a script bug.)
**Training-time rolling-window picture** (2442 rollouts/seed, 5,001,216 final step in every
seed — all confirmed complete): `N=2` clean throughout (0/2442 nonzero collision). `N=3`
seed1: 24/2442. `N=4`: seed1 15/2442, seed2 53/2442 (highest of the three — its last-10 shows
`collision_rate=0.02` for the first 5, then 0 for the final 5, i.e. converges right at the very
end), seed3 40/2442. All three `N=4` seeds' last-10 `target_lost_rate` are exactly zero.
**Conclusion**: this is the best `N=4` result of the project to date — 3/3 seeds clean on
collision at both eval-time and (with the seed2 caveat above) by the end of training, 0%
`target_lost_rate`, and the best average `tracking_rmse` (1.91 final) measured so far. `N=3`'s
one seed showed `target_lost_rate` noise (5% final) worth checking against a second seed
before trusting it (see next entry). `N=2` is unambiguously clean.
**Status**: adopted as the validated reference point (`phase2-combined`, commit `3f9069c`).
Not yet fast-forwarded to `main` — see `TODO.md`.

## `phase2_n3_seed2` — second N=3 seed, checking whether seed1's `target_lost` noise was real

**Objective**: seed1's `N=3` result above showed 5% final `target_lost_rate` — check whether
that's a real, repeatable issue at `N=3` or ordinary seed variance, before trusting the
Phase2-combined validation's `N=3` result.
**Configuration**: `NUM_AGENTS=3`, `SEED=2`, 1.5M steps (shorter than the 5M full-validation
runs — a targeted diagnostic, not a full seed).
**Result** (eval-time, 100 episodes): `collision_rate` 0%/0%, `target_lost_rate` 1%/1%,
`tracking_rmse` 2.32 final / 2.27 best. Training-time: 733 rollouts logged, 0/733 nonzero
collision, last-10 `target_lost_rate` oscillating 0-4% (`[0.02, 0.02, 0.02, 0.04, 0.02, 0.02,
0.02, 0.04, 0.02, 0.02]`) — low and stable, not trending up.
**Conclusion**: seed1's 5% was consistent with ordinary seed variance around a low baseline
(1-5% range), not a systematic `N=3` regression. `N=3` is fine.
**Status**: resolved — no follow-up action needed.

## Corrected spread angle, isolated re-test — `spread_fixed_seed1`, N=4, 1 seed, 1.5M steps

**Objective**: the Phase2-combined 5-seed validation above used the *wrong* spread angle
(109.47°/120°, the global/local conflation bug — see `DECISIONS.md`). After fixing it to the
analytically-correct 60° (`ac98d67`, verified by `test_geometry.py`), check whether the bug
had actually cost anything measurable, in isolation, before trusting the combined result as-is.
**Configuration**: `NUM_AGENTS=4`, 1 seed, 1.5M steps, branch `xyz-spread-fixed` (isolated —
the corrected angle on top of the same base as Phase2-combined, nothing else changed).
**Result** (eval-time, 100 episodes): `collision_rate` 0%/0%, `target_lost_rate` 0%/0%,
`tracking_rmse` 2.58 final / 2.42 best. Training-time: 733 rollouts logged, 0/733 nonzero
collision throughout, last-10 `target_lost_rate` mostly 0-2%.
**Conclusion**: clean on every metric — consistent with (not distinguishably better or worse
than) the Phase2-combined numbers at a comparable step count. The angle bug does not appear to
have cost anything measurable in this metric set, though it was still correct to fix (the
*reasoning* was wrong even where the *outcome* happened not to be visibly harmed — see
`DECISIONS.md` for why the fix was worth making regardless, and `test_geometry.py` for the
regression check that now guards against it recurring).
**Status**: resolved. The corrected angle is what's in the code going forward; no further
action needed on this specific question.

## `N=4` sustained-flight diagnostic (`training/diagnose_horizon.py`) — the trigger for Phase 3

**Objective**: not a designed experiment — a direct check of how the best Phase2-combined
`N=4` checkpoint behaves when run for longer than its training horizon, motivated by the
deployment thread's real-world PX4/Gazebo crash reports.
**Configuration**: frozen checkpoint (no retraining), run for 60 simulated seconds instead of
the 10-second (`MAX_STEPS=200`) training horizon, without terminating on the first failure so
degradation past the horizon is actually observable rather than just a pass/fail.
**Result**: zero collisions or `target_lost` events the whole 60s (the abstract sim has no
ground/physics failure mode at the time of this test, so it can't show a literal crash), but
tracking error climbed from 0.95 (0-10s, matches training-time performance) to a peak of 3.66
(20-30s) before recovering to 1.5-1.8 (40-60s).
**Conclusion**: real, load-bearing evidence — not just a plausible story — that the policy's
behavior degrades once flying longer than anything it ever trained on. The 20-30s worst window
lines up *directly inside* the 15-40s window where the real PX4/Gazebo deployment actually
crashed (see `deployment/docs/PHASE2_HANDOFF.md`), supporting a training-horizon-vs-real-flight
mismatch as at least part of that crash's cause.
**Status**: directly motivated Phase 3's longer-episode change (below) — the first fix in this
chain grounded in a direct measurement of sustained-flight behavior, not an assumption that
"longer training horizon must help."

## Phase 3 bundle — sustained-flight resilience, N=4, 3 seeds, 3M steps — safety succeeded, tracking failed catastrophically

**Objective**: test five bundled changes (longer episodes `MAX_STEPS` 200→1800, larger
network, ground awareness, per-axis Z/XY speed limits, dynamic target motion) as one "leap of
faith" (explicit user framing, with an agreed fallback to isolate individually if this failed)
— direct response to the horizon-diagnostic finding above.
**Configuration**: `NUM_AGENTS=4`, 3 seeds, `TOTAL_STEPS=3_000_000`, branch
`phase3-resilience` off `phase2-combined` (kernels `stage-marl-phase3-resilience-seed{1,2,3}`).
**Result** (eval-time, 100 episodes each — note `MAX_STEPS=1800` now, so raw episode/rollout
counts aren't directly comparable to any run above):

| seed | collision (final/best) | target_lost (final/best) | tracking_rmse (final/best) | worst min_dist |
|---|---|---|---|---|
| 1 | 0% / 0% | 97% / 98% | 7.22 / 7.40 | 4.201 |
| 2 | 0% / 0% | 100% / 100% | 6.95 / 7.30 | 4.203 |
| 3 | 0% / 0% | 89% / 89% | 6.51 / 6.51 | 4.201 |

**Training-time rolling-window picture** (209 rollouts/seed — far fewer than the
`MAX_STEPS=200`-era runs, expected given `ROLLOUT_LEN` is 9x longer per rollout now; final
step 3,009,600 in every seed, confirmed complete): `ground_strike_rate` was **0/209 rollouts
in every seed, max ever 0.000** — ground awareness worked cleanly from the very first test, no
iteration needed (unlike the brake, which needed the multi-pass fix after its first `N=4`
test). Collision: seed1 3/209 nonzero, seed2 and seed3 0/209 — safety essentially untouched by
the bundle. `target_lost_rate` was catastrophic and **flat from the very first logged rollout**
in every seed (seed1 last-10: 92-98%, seed2: 98-100%, seed3: 82-96%) — not a mid-training
regression, a failure that never once improved across the entire 3M-step run.
**Conclusion**: the bundling's risk (from `DECISIONS.md`) materialized partially but
diagnosably. Ground awareness and per-axis dynamics: clean success. Collision: essentially
unaffected (still ~0%), consistent with the brake being an independent mechanism from anything
in this bundle. Tracking: the `target_lost` catastrophe. Because the failure was flat-from-
step-1 (not a mid-run regression) and collision — subject to the same "9x fewer episodes"
risk from the longer-episode change — converged fine, data-starvation from fewer episodes was
argued against as the primary cause, pointing instead at `LOST_TIMEOUT_SEC` (a fixed 2.0s
grace period, never rescaled when `MAX_STEPS` grew 9x) — see the `DECISIONS.md` entry and the
sweep below.
**Status**: root cause diagnosed, fix (env-overridable `LOST_TIMEOUT_SEC`, swept 6/10/18s) in
progress — see below.

## `LOST_TIMEOUT_SEC` sweep — 6s/10s/18s, in progress

**Objective**: find a `LOST_TIMEOUT_SEC` value that works at the new `MAX_STEPS=1800` (90s)
episode length, empirically rather than by computing one analytically (no clean formula maps
episode length to grace-period length — see `DECISIONS.md`).
**Configuration**: `NUM_AGENTS=4`, `TOTAL_STEPS=3_000_000`, branch `phase3-resilience`
(commit `62a685d`, includes the altitude/Z-smoothing fix — see `DECISIONS.md`), 5 kernels: 6s
x2 seeds, 10s x2 seeds, 18s x1 seed.
**Status update**: the stuck 18s kernel (Kaggle's session-cap issue) cleared on its own after
~4-4.5 hours with no manual intervention — see `KNOWN_ISSUES.md` item 15 (resolved). All 5
completed.
**Result** (eval-time, final model, 100 episodes each):

| kernel | `LOST_TIMEOUT_SEC` | collision | target_lost |
|---|---|---|---|
| timeout6-seed1 | 6 | 1% | 85% |
| timeout6-seed2 | 6 | 0% | 96% |
| timeout10-seed1 | 10 | 0% | 93% |
| timeout10-seed2 | 10 | 4% | 4% |
| timeout18-seed1 | 18 | 4% | 3% |

**Conclusion**: **not a clean dose-response.** 3 of 5 runs stayed catastrophically broken
(85-96% `target_lost_rate`) regardless of timeout length; the 2 that worked (10s-seed2,
18s-seed1) happened to be the two longest timeouts tested, but 10s-seed1 — identical
configuration to 10s-seed2 — failed just as badly as the 6s runs, a ~90-point spread between
two seeds at the *exact same* timeout value. Grace-period length alone is not the primary
lever; whichever seeds happen to "break through" during training matters at least as much as
the timeout constant itself — the same bimodal, seed-sensitive pattern already seen elsewhere
in this project (e.g. the `N=3` entropy-recovery relapse in the first `NUM_AGENTS=3` curriculum
run above).
**Status**: resolved. `LOST_TIMEOUT_SEC`'s default was raised to 6.0 (up from the original flat
2.0, not all the way to 10/18, given the lack of a clean dose-response) — see `DECISIONS.md`.
This result is what motivated building active search (below) as the mechanism actually expected
to do the real work, rather than continuing to sweep the timeout constant further.

## Active search — implemented, then corrected after a build/launch discipline lapse, then validated as a real, major improvement

**Objective**: give the swarm something better to do than coast toward an increasingly stale
dead-reckoned point once contact is lost entirely — each agent fans out in its own fixed
heading (assigned once per loss-of-contact event, evenly spread from a randomized base angle)
instead of every agent converging on the same point.
**Configuration**: `_assign_search_directions()` / per-agent `_effective_pos_est`/
`_effective_vel_est` in `envs/formation_env.py` (commit `13b9e38`) — reuses the existing
`r_track`/`r_velocity` reward math unchanged, pointed at a per-agent search waypoint instead of
the shared track estimate during a loss period. No new reward term added.
**Process failure, caught and corrected, not swept under the rug**: this commit (and the
`DISABLE_TARGET_LOST_TERMINATION` ablation flag, `b15f391`, below) were made locally but not
pushed to `origin/phase3-resilience` before several Kaggle kernels claiming to test them had
already launched and completed. Those runs (`stage-marl-active-search-seed{1,2,3}` v1,
`stage-marl-search-t2-seed1`, `stage-marl-search-t18-seed1`, `stage-marl-ablation-noterm-
seed{1,2,3}` v1) actually cloned whatever was on `origin` at the time (`62a685d` — plain
`LOST_TIMEOUT_SEC=6`, no active search, no ablation flag), confirmed directly from their eval
CSVs (old 10-column schema, no `contact_fraction`, numbers matching the pre-active-search 6s
baseline almost exactly) — found via a missing-columns anomaly, not assumed. Fixed by pushing
and relaunching as "v2"; **confirming `git status` shows up to date with origin immediately
before every Kaggle launch is now a standing pre-launch check** (see `ENVIRONMENT.md`), not a
one-off fix. See `PHASE2_CHECKPOINT.md` for the full incident record.
**Result (v2, correct code, 2 seeds, 6s timeout)**: `contact_fraction` 85.0-90.7% (final) — up
from ~16-21% pre-active-search — `collision_rate` 0-1%, `target_lost_rate` still 60-81%. But
**median loss-event length was exactly 121 steps in both seeds — 1 step past the 120-step (6s)
cutoff.** Read: search is working — the median failure is a near-miss on timing, not a
never-finds-it failure (contrast the ablation's median loss streak of ~900-1150 steps, below).
This reverses the earlier (invalid-code) conclusion that active search doesn't help.
**Status**: the single best result of the whole `target_lost` investigation. Directly motivated
the `t8`/`t10` timeout-bracket follow-up (below) — if search reliably gets a swarm to within
~1 step of reacquiring, a slightly longer timeout should convert many of those near-misses into
successes.

## `DISABLE_TARGET_LOST_TERMINATION` ablation — training-time diagnostic, 3 seeds — partial improvement, new cost, not adopted

**Objective**: test whether ending episodes on `target_lost` was itself starving the policy of
the exact experience (sustained contact, or a full loss-then-recovery cycle) it needs to learn
either — isolated from any change to the tracking objective itself (`_target_lost`, `r_contact`,
`TARGET_LOST_PENALTY` all computed identically regardless of this flag; only termination is
gated).
**Configuration**: `NUM_AGENTS=4`, `TOTAL_STEPS=3_000_000`, `DISABLE_TARGET_LOST_TERMINATION=1`,
`phase3-resilience` @ `b15f391` (v2, correctly pushed — see the incident note above), 3 seeds.
**Result**: `contact_fraction` improved consistently across all 3 seeds (31-48%, up from
~16-21% pre-ablation) — a real, reproducible effect. But `target_lost_rate` under the real 6s
timeout stayed catastrophic (79-100%), and **collision safety, previously ~0%, became
seed-dependent and unstable (0-21%)** — likely the brake's two-agents-at-a-time gap (item 13)
being stressed harder by longer average episodes and more divergent per-agent motion while
searching, with no hard stop to bound it.
**Conclusion**: a genuine partial improvement that doesn't reach the finish line and adds a new
cost. Not adopted as a training default.
**Status**: resolved — not pursued further; active search (above) is the mechanism actually
carried forward.

## Active search + longer timeout bracket (`t8`/`t10`) — bimodal, seed-sensitive; when it converges, tracking is excellent but exposes the brake's gap

**Objective**: test whether active search plus a longer grace period (bracketing the "1 step
past cutoff" near-miss finding above) converts more near-misses into successful reacquisitions.
**Configuration**: `NUM_AGENTS=4`, `TOTAL_STEPS=3_000_000`, `phase3-resilience` @ `ddac78e`,
active search + `LOST_TIMEOUT_SEC` overridden: `stage-marl-search-t8-s{1,2}` (8s, 2 seeds),
`stage-marl-search-t10-s{1,2,3}` (10s, 3 seeds).
**Result** (eval-time, final model, 100 episodes each):

| run | timeout | collision | target_lost | contact_fraction |
|---|---|---|---|---|
| t8-s1 | 8s | 8% | 2% | 99.6% |
| t8-s2 | 8s | 14% | 2% | 99.7% |
| t10-s1 | 10s | 1% | 70% | 84.1% |
| t10-s2 | 10s | 0% | 96% | 68.4% |
| t10-s3 | 10s | 6% | 4% | 99.4% |

**Both `t8` seeds converged to excellent tracking** (contact_fraction ~99.6-99.7%,
`target_lost_rate` ~2%) — the best tracking numbers in the whole investigation — **but at a
real collision-safety cost (8-14%)**, not seen at this magnitude anywhere else in the project
since the original `N=4` brake gap. **`t10` split 2-of-3 bad**: seed3 matches `t8`'s good
outcome almost exactly (99.4% contact, 4% `target_lost`, 6% collision); seed1/seed2 got stuck
in a much worse regime (68-84% contact, 70-96% `target_lost`) despite identical configuration.
A training-curve breakdown (first/mid/last rollout-window averages) on the two bad `t10` seeds
showed they were still in the bad regime at the midpoint of training, consistent with "needs
more training steps," not "10s doesn't work" on its own — not yet confirmed by an actual longer
run (see the 5M-step resume entry below).
**Conclusion, two separate findings**: (1) active search + a long-enough timeout genuinely can
produce excellent tracking (99%+ contact) — the seed-sensitivity looks like a
training-duration/convergence question, not evidence the mechanism itself is capped low.
(2) **When a seed does converge to that tight, confident tracking behavior, deterministic
execution exposes a real collision-safety cost that wasn't visible in earlier, less-converged/
looser-tracking runs** — directly motivated the collision-discrepancy investigation and the
relative-velocity brake reformulation below.
**Status**: `t10` seed1/seed2 are being resumed from their ~3.01M-step checkpoints to 5M steps
to test the "just needs more training" hypothesis — in progress, not yet complete (see
`SESSION_HANDOFF.md`). `t8`/`t10`-seed3's collision cost is what the relative-velocity brake
fix (below) targets.

## Train-vs-eval collision-rate discrepancy — investigated and explained

**Objective**: deterministic eval collision rate had run consistently higher than the trailing
training-time (stochastic) rate across many runs project-wide, most starkly `search-t8-s1`/`s2`
(train ~1-1.2%, eval 8-14%) — investigate why, rather than assume it's noise or an eval bug.
**Method**: took the exact same frozen `actor_1.pt`/`actor_2.pt` weights from `search-t8-s1`/
`s2` and ran 100 fresh episodes twice each — once deterministic (`tanh(mu)`, matches
`evaluate.py`), once stochastic (sampled, matches `train.py`'s own rollout collection) —
identical eval seeds both times, only the action-selection mode differs
(`/tmp/.../collision_discrepancy_test.py`, kept local/scratch, not merged into `evaluate.py`).
**Result**: seed1: deterministic 11% vs stochastic 2% — alone reproduces the originally-observed
gap almost exactly (matches training's own ~1%). seed2: deterministic 13% vs stochastic 10% —
determinism only explains a small slice here; the stochastic-replayed-after-training number
(10%) is still far above what training itself logged near the end (~1.2%).
**Conclusion, two separate findings, not one**: (1) **determinism vs. stochasticity is a real,
demonstrated cause**, not speculation — a policy shaped by entropy-driven exploration can
converge its *mean* action to a knife-edge equilibrium riskier than the noise-perturbed
behavior actually seen during training; this alone explains seed1's entire gap. Weak secondary
signal (small-N, not confirmed): deterministic collisions involved 2+ simultaneously-close
agents more often than stochastic ones (seed2: 54% vs 30%) — consistent with the brake's
documented multi-agent gap (item 13/`KNOWN_ISSUES.md`) being easier to trigger without noise
breaking up a persistent near-symmetric closing pattern. (2) **For seed2's remaining gap, an
earlier verbal explanation given mid-session was wrong and is corrected here**: seed2's tracking
was *not* less converged — `search-t8-s2`'s `contact_fraction` was 99.7%, among the best in the
project, so "the policy hadn't converged yet" does not hold up as an explanation. The more
defensible account: the residual gap is between two *stochastic* numbers (10% freshly
re-measured over 100 episodes vs. 1.2% logged near the end of training over a rolling window of
only ~50 episodes) — for a true rate near 10%, a 50-episode sample has enough binomial variance
(expected count ~5, plausible range roughly 1-9) that observing something as low as 1.2% by
chance isn't surprising. Not confirmed with a formal significance test — flagged as the leading,
most parsimonious explanation available, not a proven one.
**Practical implication**: deterministic eval is the number to trust for real deployment (a
flight controller runs the mean policy, not a sampled one) — the higher eval collision rates
are the honest ones, not the more optimistic training-time rolling numbers.
**Status**: resolved/explained. Directly motivated the relative-velocity brake reformulation
below.

## Relative-velocity brake reformulation — implemented, adversarially verified, Kaggle training validation in progress

**Objective**: `_apply_brake`'s `v_closing` was `dot(v_a, dir_to_b)` — agent a's own velocity
toward b, silently treating b as stationary — not the true mutual closing rate. The adversarial-
test guarantee (`DECISIONS.md`) only held because every agent ran the identical symmetric
one-sided formula; a real, independently-moving vehicle (or, per the finding directly above, a
deterministic policy converging every agent to the exact same confident behavior with nothing
to perturb a persistent near-symmetric standoff) breaks that assumption.
**Configuration**: `v_closing` changed to `dot(v_a - v_b, dir_to_b)` in both the multi-pass
correction loop and the post-hoc violation diagnostic (commit `94216fd`) — no other change to
the correction mechanism (agent a still unilaterally corrects only its own velocity, same
multi-pass loop, same `max_closing` shape). No changes to the tracking/search mechanism.
**Verification before any Kaggle spend**: the two adversarial scenarios already described in
`DECISIONS.md` (2-agent head-on, 4-agent converge-on-centroid, both starting outside
`SAFE_DIST_ENTER`, always commanding `MAX_ACTION_SPEED`) were rebuilt as a committed regression
test, `test_brake_relative_velocity.py`, rather than left as a one-off manual claim. Both pass
against the new formulation: minimum distance never breached `COLLISION_DIST`, residual
violation ~0 (2-agent: exactly 0; 4-agent: 4e-8, float-precision noise). Standard local smoke
tests (`test_geometry.py`, `test_env.py` at `N=2/3/4`) also pass clean.
**Result (all 3 seeds complete, 3.01M steps each)**: `target_lost_rate` 88-97%, `collision_rate`
0-2% (eval-time and training-time-rolling-window both), every seed.
**Conclusion: inconclusive for the question this was meant to answer, not a negative result.**
All 3 seeds landed in the same bad-tracking regime the plain 6s timeout sweep already showed
(2 of 2 seeds there were also 85-96% `target_lost`) — none reached the tight, confident tracking
convergence that `search-t8`/`search-t10`-seed3 showed, which is specifically the regime where
the 8-14% collision problem this fix targets was ever observed. With `target_lost_rate` this
high, the swarm spends most of its time coasting/searching rather than tightly converged near
the target, so there's little opportunity for the brake to be stressed the way it was in the
runs that motivated this fix. **Choosing to hold `LOST_TIMEOUT_SEC` at its 6s default "for a
clean test of the brake alone" was reasonable in principle but turned out to be an unlucky
choice in practice** — 6s is exactly the timeout value most likely to produce this
non-convergent regime, based on the sweep and ablation data already in this log.
**Status**: the brake fix itself is unrefuted (nothing here contradicts it), but this batch
doesn't confirm or deny whether it reduces `collision_rate` in the regime that matters. Retesting
at a timeout known to let seeds converge well (8-10s, matching where the original problem was
observed) is the natural next step — not yet launched, pending user direction. See
`KNOWN_ISSUES.md` item 13.

## `search-t10` seed1/seed2 — 5M-step resume, testing "just needs more training"

**Objective**: test whether the two `t10` seeds stuck in a bad tracking regime (see the `t8`/
`t10` bracket entry above) would improve given more training steps, holding everything else
(seed, `LOST_TIMEOUT_SEC=10`, active search, all Phase 3 mechanisms) fixed.
**Configuration**: resumed (not retrained from scratch) from each seed's actual ~3.01M-step
checkpoint (`checkpoints/latest_{1,2}.pt`, uploaded as a private Kaggle dataset and copied in
before `train.py` runs) to `TOTAL_STEPS=5,000,000`. Same commit, same `LOST_TIMEOUT_SEC=10`
override as the original runs — the only change is the step budget.
**Caveat, disclosed before launching, not discovered after**: `train.py`'s LR and entropy-
coefficient schedules are both driven by `total_steps / TOTAL_STEPS`. By the end of the original
3M-step runs that ratio had reached 1.0, so both had annealed fully to their floor. Resuming
with `TOTAL_STEPS=5,000,000` immediately recomputes that ratio at ≈0.60, so LR and entropy
coefficient both jump back up rather than continuing one smooth anneal from step 0 to 5M. This
means the result isn't quite equivalent to what an uninterrupted 5M-step run would have looked
like — it's "3M steps fully annealed, then a fresh top-up of LR/exploration for 2M more steps."
Accepted as a disclosed trade-off (uses the already-completed 3M of compute; a from-scratch 5M
run would take ~5x longer per seed) after the user chose it explicitly over a from-scratch
alternative.
**Infrastructure note**: hit and fixed a real Kaggle platform quirk along the way — datasets
now mount at `/kaggle/input/datasets/<owner>/<dataset-slug>/`, not the classic `/kaggle/input/
<dataset-slug>/` the current `kaggle-cli` docs still describe. See `ENVIRONMENT.md`.
**Result** (eval-time, final model, 100 episodes each; 3M-step baseline numbers alongside for
comparison):

| seed | steps | collision | target_lost | contact_fraction | tracking_rmse |
|---|---|---|---|---|---|
| 1 (3M) | 3.01M | 1% | 70-72% | 0.84 | ~6.9 |
| 1 (5M resume) | 5.01M | 1% | **41%** | **0.917** | **4.011** |
| 2 (3M) | 3.01M | 0-1% | 92-96% | 0.68-0.71 | 7.2-7.7 |
| 2 (5M resume) | 5.01M | 1% | **83%** | **0.771** | **6.157** |

New diagnostics (added to `evaluate.py` specifically for this comparison, see `ARCHITECTURE.md`):
seed1's `p95_true_track_err`/`max_true_track_err` are 8.4/30.9 against a mean of 3.0 — a real
tail, not just a slightly-worse average. Reacquisition-time is bimodal: `median_reacquisition_
steps` is 1-2 (most reacquisitions happen almost instantly, the target re-enters sensor range
essentially immediately) but `p95`/`max` (73-94 / 188-195 steps) pull `mean_reacquisition_steps`
up to ~13 — a long tail of much harder recoveries behind a typically-fast median.
**Conclusion**: **partially confirms "just needs more training," does not fully resolve it.**
Both seeds improved substantially with 2M more steps (seed1: 70-72%→41% `target_lost`, seed2:
92-96%→83%) — real, not noise-sized movement, consistent with the earlier training-curve
finding that both seeds were still in the bad regime mid-training. But neither reached anywhere
near `t10`-seed3's 4-10% `target_lost_rate` — more training helps, but 5M steps (with the
LR/entropy-schedule caveat above) isn't sufficient on its own to make these two seeds converge
like the "lucky" seed did. `collision_rate` stayed low for both (1%) — notably lower than the
already-converged `t8`/`t10`-seed3 checkpoints' 6-14% — consistent with the collision-discrepancy
finding above: a policy that hasn't yet converged to a tight, confident tracking equilibrium
hasn't found the knife-edge behavior that trades into collision risk either.
**Status**: resolved for this specific question (more training helps but isn't sufficient alone
within a 5M budget); whether even more steps, a different seed, or a design change would close
the remaining gap is open, not further pursued in this pass.

## Critic-LR ablation — rejected as a general recovery fix; seed-dependent behavioral redistribution observed

**Objective**: a supervisor-guided investigation (not recorded step-by-step in this file — see
`PHASE2_CHECKPOINT.md`'s 2026-08-25 entry for the full diagnostic chain) traced active search's
`target_lost` failures to the production `CentralCritic`'s second Tanh layer saturating 100%
within the first rollout of training, collapsing its value predictions to an near-constant
regardless of state. A per-update trajectory and LR sweep on a short (2250-update) reproduction
found `CRITIC_LR=1e-5` (30x smaller than the shared actor/critic default) fully prevented that
saturation over the sweep's own budget. This entry tests whether that holds, and whether it
actually improves recovery behavior, in a real, full-length training run.
**Configuration**: matched 3-seed comparison, `NUM_AGENTS=4`, default 6s `LOST_TIMEOUT_SEC`,
`TOTAL_STEPS=3,000,000`, active search, reverted (one-sided) brake -- identical to the original
`search_v2` baseline in every respect except `CRITIC_LR` (now a separate, env-var-overridable
constant in `train.py`, added for this experiment; defaults to the actor's `LR` when unset, so
the control arm is unchanged production behavior). Control: `CRITIC_LR` unset (=3e-4). Treatment:
`CRITIC_LR=1e-5`. Seeds 1/2/3 for both arms (seed 3's treatment run was executed locally rather
than on Kaggle -- faster local throughput measured at ~700 steps/sec vs Kaggle's ~330 -- control
seeds 1-3 and treatment seeds 1-2 ran on Kaggle). Also added per-rollout critic-health logging
to `training_log.csv` (`critic_v_mean/std/min/max`, `critic_target_mean/std`, `critic_pearson`/
`critic_spearman` against real returns, final-layer saturation fraction/preactivation std).
**Bug caught before any Kaggle spend**: the pre-existing LR-anneal block unconditionally set
both `opt_actor` and `opt_critic` to the same actor-derived `lr_now` every rollout, silently
overwriting `CRITIC_LR` back to a `LR`-derived value on the very first update regardless of what
it was constructed with. Caught by a local smoke test producing bit-identical critic-health
output under `CRITIC_LR=3e-4` vs `CRITIC_LR=1e-5` before pushing anything to Kaggle. Fixed --
each optimizer now anneals from its own base LR -- and reverified (divergent `final_sat=0%` vs
`100%` output) before launching.
**Result 1, critic health**: saturation is delayed, not prevented, over a full run. All 3
treatment seeds show `final_sat=0%` at the first rollout but **100% by the final rollout**
(confirmed independently in all 3 -- e.g. the local seed-3 run reached 100% by step 288,000,
~rollout 20 of 209). `critic_pearson`/`critic_spearman` against real in-sample returns stayed
small and sign-mixed (roughly -0.1 to +0.3) for the entire 3M-step run in every seed, both arms
-- no seed, in either condition, ever developed a durable positive value/return relationship.
**Result 2, behavior** (100 deterministic eval episodes, final model, each):

| seed | condition | success | collision | target_lost | contact_fraction | reacquire_rate | mean_reacq_steps |
|---|---|---|---|---|---|---|---|
| 1 | control | 0.050 | 0.000 | 0.950 | 0.831 | 0.842 | 3.30 |
| 1 | 1e-5 | 0.080 | 0.000 | 0.920 | 0.846 | 0.696 | 8.43 |
| 2 | control | 0.130 | 0.010 | 0.860 | 0.832 | 0.813 | 6.70 |
| 2 | 1e-5 | 0.050 | 0.000 | 0.950 | 0.821 | 0.520 | 11.43 |
| 3 | control | 0.020 | 0.000 | 0.980 | 0.753 | 0.698 | 9.54 |
| 3 | 1e-5 | 0.200 | 0.050 | 0.750 | 0.866 | 0.688 | 16.84 |

Seed 3 improved substantially on `target_lost`/`contact_fraction` under the treatment, but
`collision_rate` rose from 0% to 5% and `mean_reacquisition_steps` nearly doubled. Seeds 1-2
showed no improvement (seed 1, within noise) or outright regression (seed 2) on `target_lost`,
`reacquire_rate`, and `mean_reacquisition_steps`. **`mean_reacquisition_steps` was worse under
`CRITIC_LR=1e-5` in all three matched pairs** -- the single most consistent effect in the whole
dataset, and it points against the treatment.
**Conclusion**: critic saturation is a confirmed, real, LR-sensitive training pathology --
directly measured, reproduced across independent seeds and checkpoints, mechanistically traced
to the last nonlinear layer before the value head. But reducing `CRITIC_LR` to delay it does
**not** reliably fix recovery behavior: critic health didn't even durably improve (saturation
returns in every treatment seed by the end of a full run), and behavioral results were mixed --
one seed traded a real tracking improvement for a new collision cost, two seeds showed no
improvement or regression, and reacquisition speed was uniformly worse. Record this as **critic
saturation confirmed as a secondary/contributing pathology, not the dominant behavioral
bottleneck** -- not simply "the fix failed," since seed 3's redistribution (better tracking,
worse safety, slower reacquisition) suggests `CRITIC_LR` does interact with what the actor
learns, just not in a consistently beneficial direction, and there isn't yet evidence for why.
**Status**: resolved -- rejected as a general recovery solution. Further critic-LR sweeps
(`3e-6`, `1e-6`, ...), critic-loss variants, or critic-architecture experiments are not
justified without new evidence specifically motivating them. Investigation moved to the actor's
own action dynamics during search (see the entry below) rather than continuing to chase the
critic.

## Actor search-action dynamics — natural loss events, PPO vs scripted at the action level

**Objective**: with critic saturation demoted to a secondary pathology, return to the actor's
own behavior during search with more precision than the original "bad `SEARCH_SPEED`/heading"
hypothesis (already weakened by the forced-loss diagnostic showing the mechanism is usable) or
the reward-economics framing (already shown to be a weak, same-step signal). Quantify what PPO's
actions actually do, step by step, during a real loss event, and how that differs from the
scripted controller beyond a single aggregate cosine-similarity number.
**Configuration**: same checkpoint as the earlier loss-instrumentation entry (`search_v2_seed1`,
`actor_1.pt`), 30 naturally-occurring loss events (real, deterministic, unforced -- driven
entirely by the trained policy's own behavior) across 40 episodes. Per (event, step, agent):
PPO action, previous PPO action and their cosine similarity/change magnitude (action
persistence), the scripted controller's counterfactual action from the identical state and its
own persistence, PPO-vs-scripted cosine, true distance to target, distance to the synthetic
search waypoint, distance to the sensor boundary (`true_dist - SENSOR_RANGE`) and per-step
progress toward it, real reward components, and `track_confidence`. No environment, reward,
observation, or training changes -- read-only instrumentation of an already-trained checkpoint.
**Result**: 13/30 events reacquired, 17/30 target_lost. Scripted's own action persistence
(`mean cos=0.999`) is actually *higher* than PPO's (`0.906-0.930`) -- the "scripted continuously
re-aims while PPO freezes" hypothesis does not hold; a smooth continuous controller naturally
shows high step-to-step persistence too, since position barely moves in one 0.05s tick.
Per-step progress toward the sensor boundary is negative on average in *both* outcome groups but
far worse in failures (-0.0057/step reacquired vs -0.0246/step target_lost, a ~4x gap; 91% of
target_lost steps show negative progress vs 59% for reacquired). One fully-detailed target_lost
event (seed 3) resolved this into a concrete mechanism: drone1 closed steadily and correctly
(`boundary_dist` 0.64->0.60m over the final 3 logged steps, positive progress throughout) and
was within roughly 30 steps (~1.5s) of saving the whole swarm when the 120-step timeout hit --
while drones 2-4 drifted consistently *away* from the boundary the entire final stretch, never
close at all. Not oscillation, not a frozen action -- confidently wrong headings in 3 of 4
agents, and one correct, promising trajectory that simply ran out of time.
**Caveat**: the pooled cumulative-progress statistics mix all 4 agents per event, but swarm-wide
reacquisition only needs one to succeed -- so a "reacquired" event's average can look weak even
when the one agent that mattered was doing everything right, diluted by teammates who weren't.
The per-event, per-agent breakdown (as above) is more informative than the pooled mean for
exactly this reason; flagged, not silently smoothed over.
**Conclusion**: the failure is not well described as "PPO can't compute the right direction" --
one agent demonstrably could, consistently, in a real failure case. It looks more like (a) most
agents' search headings, once assigned, never get corrected even when unproductive, so a
4-random-heading swarm often has only 0-1 agents on a viable trajectory at any given time, and
(b) even a successful trajectory closes slowly enough (~0.02 boundary-distance/step) that the
120-step budget is a tight fit rather than a comfortable margin, not just a training-quality gap.
**Status**: open. Both candidate mechanisms above are testable but distinct from what's already
been ruled out (search geometry itself, critic saturation as the primary cause) -- next step not
yet decided.

## Network-capacity sweep — N=4 has a sweet spot at the Phase 3 default; N=3 fails to learn at any tested size

**Objective**: Phase 3's hidden-width jump (Actor 64→128, Critic 128→256) was bundled into the
original 5-change "leap of faith" and never isolated, even after the bundle was accepted. With
`ACTOR_HIDDEN`/`CRITIC_HIDDEN` now env-overridable (commit `55e7232`), sweep network capacity
directly, and use the same sweep to attempt the project's first `NUM_AGENTS=3` Phase-3-era
checkpoint on a *working* (reverted, one-sided) brake — the two prior `N=3` attempts
(`deploy-n3-s{1,2}`) both used the since-reverted relative-velocity brake and both failed to
converge, leaving open whether the brake or something else was responsible.
**Configuration**: 5 Kaggle kernels, reverted brake, `LOST_TIMEOUT_SEC=8` (best-evidenced timeout
at the time), `TOTAL_STEPS=3,000,000` each. `stage-marl-netsweep-n4-{small,medium,large}`
(`NUM_AGENTS=4`, hidden 64/128, 128/256, 256/512) and `stage-marl-netsweep-n3-{medium,large}`
(`NUM_AGENTS=3`, hidden 128/256, 256/512). `netsweep-n4-medium` deliberately reuses `SEED=1` with
otherwise the exact same config as the original `search-t8-s1` (which converged cleanly under the
old brake before the relative-velocity detour) — a built-in sanity check that the revert actually
reproduces that result rather than something else having silently changed too.
**Result** (eval-time, final model, 100 episodes each):

| run | hidden (actor/critic) | success | collision | target_lost | contact_fraction | tracking_rmse |
|---|---|---|---|---|---|---|
| n4-small | 64/128 | 53% | 2% | 45% | 0.908 | 5.03 |
| n4-medium | 128/256 | 90% | 8% | 2% | 0.996 | 1.92 |
| n4-large | 256/512 | 2% | 1% | 97% | 0.734 | 7.28 |
| n3-medium | 128/256 | 0% | 0% | 100% | 0.647 | 7.15 |
| n3-large | 256/512 | 0% | 0% | 100% | 0.659 | 7.76 |

`n4-medium`'s sanity check passed: 8% collision / 2% `target_lost` reproduces `search-t8-s1`'s
original 8% / 2-3% almost exactly. Training-curve check (`target_lost_rate` at rollout
0/52/104/156/206 of 209): both `n3` runs are flat at 100% (one single 98% midpoint for
`n3-large`) from the very first rollout (`total_steps=14,400`) straight through to the end —
never showed any learning signal at all, not a failure to fully converge. `n4-large` is similarly
flat around 92-98% the whole run.
**Conclusion, two separate findings**: (1) for `NUM_AGENTS=4`, capacity is not monotonic — the
Phase 3 default (128/256) is a genuine sweet spot, not an arbitrary starting guess that happened
to work. Going smaller (64/128) partially degrades tracking (45% `target_lost`); going larger
(256/512) doesn't just fail to help, it collapses learning almost entirely (97% `target_lost`,
flat from rollout 1 — the same "never learns" shape as a badly-misconfigured run, not a
slower-converging one). (2) for `NUM_AGENTS=3`, **neither tested network size ever learns
anything** — both `n3-medium` and `n3-large` are flat at ~100% `target_lost_rate` from the first
logged rollout, the same shape as `n4-large`'s failure, not a scaled-down version of
`n4-medium`'s success. This is the first time `N=3` has been tried under Phase 3's full mechanism
set (active search, ground awareness, dynamic target, per-axis dynamics) with a *working* brake,
and it still doesn't learn at all — network capacity is ruled out as the explanation (two
different sizes, same total failure), but no alternative explanation has been tested yet.
**Status**: resolved for `N=4` — `ACTOR_HIDDEN=128`/`CRITIC_HIDDEN=256` stays the default, no
reason to change it. **Open and newly concerning for `N=3`**: still no Phase-3-era
`NUM_AGENTS=3` checkpoint exists that learns tracking at all, under either brake formulation.
Since `deployment/inference_node.py` hard-requires `NUM_AGENTS=3`, this is a real blocker for the
deployment track, not just an unexplored corner — see `KNOWN_ISSUES.md`.

## Best-agent progress, productive-agent count, and confidence-response — re-analyzing the same 30 natural loss events with swarm-correct metrics

**Objective**: the actor search-action dynamics entry above pooled all 4 agents' per-step
progress into one mean per event — but swarm-wide reacquisition only needs ONE agent to succeed,
so a pooled mean can look weak even when the one agent that mattered was doing everything right,
diluted by three teammates who weren't (flagged as a caveat there, not yet corrected for).
Re-analyze the identical 30 events (same checkpoint, same seeds, deterministic) with two
swarm-correct lenses, plus a direct check of whether PPO's action actually changes as
`track_confidence` falls or merely correlates with it by coincidence of when confidence is low.
**Configuration**: `actor_progress_reanalysis.py`, re-harvesting the same 30 loss events via
`actor_search_dynamics.py`'s own `run_episode` (imported, not reimplemented) against
`search_v2_seed1`/`actor_1.pt`. (1) Best-agent cumulative boundary-progress: for each event, the
single agent with the highest own cumulative progress over a window, at windows of 10/30/60/120
steps. (2) Productive-agent count: number of agents with positive trailing 10-step progress at
each event's own final point. (3) PPO action statistics bucketed by `track_confidence` (high
>0.8, mid 0.4-0.6, low <0.2), and pairwise cosine similarity between buckets' mean action
vectors.
**Result**: best-agent cumulative progress @120 steps: reacquired events (n=12) mean=+0.467,
100% of events positive; target_lost events (n=17) mean=-1.023, only 23.5% positive. The gap is
present at every window size (10/30/60/120), widening as the window grows — reacquired events'
best agent keeps making net progress the whole time; target_lost events' best agent is
net-negative even at the 10-step window (mean=-0.076, 35.3% positive) and gets worse, not better,
as the event goes on. Productive-agent count: reacquired events average **2.25** productive
agents (0% of events have zero — every single reacquired event had at least one agent making net
progress, usually two or more: 75% multi-productive). target_lost events average **0.41**
(**70.6% have zero** productive agents at all — most failures aren't "one good agent ran out of
time," they're "nobody was making progress"). Confidence-response: mean action norm is nearly
flat across confidence levels (0.552 high, 0.535 mid, 0.509 low) — action *magnitude* barely
changes. Mean action *direction* does not follow the same pattern: cosine(high, mid)=0.198,
cosine(mid, low)=0.922. The actor's mean response changes sharply the moment confidence drops
from "in contact" to "searching," then barely changes at all as confidence keeps falling from mid
to low.
**Conclusion**: the productive-agent-count split is the cleanest single signature in the whole
investigation — reacquired events essentially always have at least one (usually 2+) agents doing
real work; target_lost events usually have *none*. This reframes the failure mode away from "PPO
sometimes searches badly" toward "PPO frequently has zero agents searching productively at all,"
a stronger and more specific claim. The confidence-response result sharpens an open question from
the entry above: the actor is not ignoring confidence (it reacts sharply on the high→mid
transition, i.e. entering search mode) and is not continuously exploiting it either (mid→low is
nearly a no-op, cos=0.922) — its response saturates immediately upon entering search and doesn't
keep adapting as the situation deepens, consistent with (though not yet proof of) headings being
assigned once and not meaningfully revisited.
**Status**: resolved as a sharper characterization of the mechanism described in the entry above;
motivated the search-strategy comparison and the decisive counterfactual experiment below, both
aimed at testing the "headings aren't corrected" and "budget is tight even when things go right"
candidate mechanisms directly.

## Search-strategy comparison (A/B/C) and timeout-margin sweep — scripted controller hits a 100% ceiling regardless of condition

**Objective**: test the two candidate mechanisms the actor-dynamics work flagged directly: (a)
does the specific search-heading strategy (fixed-random vs. last-known-velocity vs. adaptive
reassignment) matter, and (b) is the 120-step (6s) budget a tight margin even for good execution,
by sweeping it wider. Uses the scripted controller (not PPO) as a best-case execution baseline,
isolating the search-strategy/timeout question from the separate question of whether PPO executes
well.
**Configuration**: `search_strategy_comparison.py`, same forced-loss masking technique as
`forced_loss_diagnostic.py` (fake `self.pos_t` far away for the duration of the real
`_update_target_track()` call, forcing its own unmodified contact computation to return empty;
the real, unmodified `_assign_search_directions`/receding-waypoint code runs underneath).
Scripted controller only, 40 trials per condition, real random warmup (300-500 steps) before each
forced-loss window. **Part 1** (forced 6s loss): condition A (current fixed heading, unmodified),
B (last-known-velocity heading, same technique as the original forced-loss diagnostic), C
(adaptive reassignment — every 20 steps/1s, each agent's own trailing boundary-progress is
checked; redraw a fresh random heading for that agent only if progress ≤0). **Part 2** (condition
A only): forced-loss duration swept 6/8/10/12s, measuring whether *any* agent's true distance
crosses back under `SENSOR_RANGE` within the window.
**Result**: all three conditions in Part 1 hit **100% reacquisition within 6s** (n=40 each), 0%
collision, median time-to-first-reacquisition of 1 step in every condition. Productive-agent
count differs (A: mean 2.45, 0% zero-productive; B: mean 3.60, 0% zero-productive; C: mean 3.23,
2.5% zero-productive) but none of that variation shows up in the outcome — it's already saturated
at the ceiling. Part 2: all four timeout durations (6/8/10/12s) also hit 100% reacquisition, 0%
collision.
**Conclusion**: neither candidate mechanism is the bottleneck **when execution quality is already
good**. A scripted controller that never produces a "confidently wrong heading" reacquires from
every single trial regardless of search-heading strategy or timeout window — this rules out "the
fixed-heading strategy itself is inadequate" and "6s is too tight a margin in general" as
standalone explanations. But this result is structurally uninformative about PPO specifically: a
scripted controller starting from a well-formed random warmup never produces the kind of state
the actor-dynamics entries showed PPO actually reaching (3-4 agents drifting confidently the
wrong way at once) — this experiment can't tell us what happens from *those* states, only that
the mechanism has enough headroom in the easy case. That gap is exactly what the next experiment
closes.
**Status**: resolved as a ceiling-effect result — informative for ruling out two mechanisms in
the abstract, but motivates branching directly from PPO's own real states instead of a fresh
scripted warmup, since that's the only way to test the states that actually matter. See the
decisive experiment below.

## Decisive experiment: branching PPO/scripted/adaptive controllers from PPO's own real loss-onset states

**Objective**: close the one remaining gap the search-strategy comparison couldn't touch — what
happens if a *non-PPO* controller is given the exact state PPO's own policy actually produced at
the moment contact was lost, not a fresh well-formed scripted warmup? This is the experiment that
can actually distinguish "the environment/timeout/search-geometry caps recovery at some rate
below 100%" from "PPO's own execution is specifically worse than it needs to be."
**Configuration**: `counterfactual_from_ppo_states.py`. Ran `search_v2_seed1`/`actor_1.pt` for 38
episodes, using `copy.deepcopy(env)` to snapshot the *complete* environment state (positions,
velocities, target state, RNG stream) at every natural steps-since-contact 0→1 transition — a
real loss-onset moment produced entirely by the trained policy's own behavior, not an artificial
mask. Collected 28 such snapshots. From each snapshot, branched three independent continuations
for up to 120 steps (`LOST_TIMEOUT_STEPS`, the real timeout budget): (1) **ppo** — the actual
trained actor continues, a direct replay-equivalent of what really happened; (2) **scripted** —
the same hand-written radial+repulsion controller used throughout this investigation; (3)
**adaptive** — the condition-C reassignment controller from the search-strategy comparison above.
Cross-tabulated all three branches' outcomes against what the *real, unbranched* PPO trajectory
actually did from that same state.
**Result**: real PPO trajectory outcomes at these 28 snapshots: 12 reacquired, 16 `target_lost`.
Aggregate reacquisition rate across all 28 branches: ppo 42.9% (matches the real split, as
expected since it's the same policy), scripted 100.0%, adaptive 100.0% — both non-learned
controllers reacquire from *every* snapshot, mean time-to-reacquire ~1.2 steps. **The decisive
cross-tab**, conditioned on the real trajectory's own outcome:

| real outcome | branch | reacquisition rate | mean n_productive | mean best-agent progress @120 |
|---|---|---|---|---|
| reacquired (n=12) | ppo | 100.0% | 1.67 | 0.422 |
| reacquired (n=12) | scripted | 100.0% | 3.58 | 0.030 |
| reacquired (n=12) | adaptive | 100.0% | 3.58 | 0.030 |
| target_lost (n=16) | ppo | **0.0%** | 0.62 | 0.364 |
| target_lost (n=16) | scripted | **100.0%** | 3.56 | 0.036 |
| target_lost (n=16) | adaptive | **100.0%** | 3.56 | 0.036 |

From the 16 states where PPO's real trajectory failed, **branching from that identical state,
PPO fails again every time (0%) — but scripted and adaptive both reacquire every single time
(100%)**, typically within 1-2 steps. `min_true_dist` reached during those recoveries averages
7.88m, essentially exactly at the 7.91m sensor boundary — these are not unusually hard geometric
states requiring a lucky trajectory; a simple hand-written controller solves them almost
immediately.
**Conclusion**: this resolves the causal fork the entire active-search investigation has been
building toward. The failure is not the environment, the search geometry, the heading-assignment
strategy, the timeout length, or (per the critic-LR ablation above) primarily the critic — all of
those are held exactly fixed between the `ppo` and `scripted`/`adaptive` branches, which start
from the *identical* state. The only thing that differs is which controller is driving from that
point forward, and that difference alone is the entire gap between 0% and 100%. **The problem is
the learned actor's own search execution.** This is consistent with, and sharpens, every earlier
finding in this chain: the productive-agent-count signature (target_lost events average 0.41
productive agents vs. reacquired's 2.25), the near-miss case study (one agent closing correctly
while three drift confidently the wrong way), and the confidence-response saturation (the actor's
response stops adapting once search mode is entered) all describe symptoms of the same underlying
fact — PPO, from a state a much simpler controller solves immediately, usually does not correct
course.
**Status**: resolved — this is the capstone finding of the critic-collapse/actor-search-dynamics
investigation chain. No fix has been designed or attempted yet; candidate directions (untested)
include a training signal that specifically rewards heading correction/reassignment, or a
credit-assignment change that makes a productive search action more clearly attributable across
the many steps between committing to a heading and actually reacquiring. Not yet decided — open
question for the user.
