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
**Status**: **in progress, not yet complete**. 4 of 5 kernels launched and confirmed
`RUNNING` (`stage-marl-timeout6-seed1`, `-timeout6-seed2`, `-timeout10-seed1`,
`-timeout10-seed2`). The 18s kernel (`stage-marl-timeout18-seed1`) failed to launch —
Kaggle's "Maximum batch CPU session count of 5 reached" error persisted even after the other 4
were confirmed running and every other kernel on the account was confirmed `COMPLETE` (not a
counting error on this session's part — checked individually, not just via the account-wide
kernel list, which turned out to have a stale/unreliable `lastRunTime` sort order not worth
trusting for "what's running right now"). Most likely explanation: an orphaned session tied to
an old, interrupted version of one of the other kernels (the initial `timeout6-seed1` push was
interrupted mid-flight earlier in this session before being re-pushed with the altitude fix
included) that isn't visible to the per-kernel `kernels status` check, which appears to report
only the latest version's session. No CLI/API mechanism was found to list or cancel such a
session directly — resolving it needs either Kaggle's own session limit to lapse the orphaned
session automatically, or the user checking Kaggle's web UI directly. See
`SESSION_HANDOFF.md`/`KNOWN_ISSUES.md` for the live troubleshooting state.
