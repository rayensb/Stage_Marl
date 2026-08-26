# AI Context — Start Here

## What this project is

A cooperative multi-agent reinforcement learning (MARL) project: `NUM_AGENTS` simulated
drones ("agents") learn, via CTDE-PPO (Centralized Training, Decentralized Execution), to
fly in a tight 3D formation while cooperatively tracking a moving target — sensed via a
**vision-based system**, not ground-truth telemetry (see below) — and avoiding collisions
with each other, both inter-agent and with the ground. The simulation/training side
(`config.py`, `envs/`, `training/` — what this doc suite covers) is a lightweight custom
physics/kinematics environment written in Python. **This is no longer the whole repo**: a
separate, concurrent workstream (`deployment/`) has since added a real PX4 SITL + Gazebo +
ROS2 inference pipeline for running the trained policy on/against real flight-control
software — see `ARCHITECTURE.md`'s `deployment/` section for how the two connect, and
`deployment/docs/` for that workstream's own documentation (out of scope for this suite).

The environment (`envs/formation_env.py`, class `FormationEnv3D`) implements the PettingZoo
`ParallelEnv` API. Training (`training/train.py`) uses a **single shared actor network**
(parameter sharing across all homogeneous drones) plus a **centralized multi-agent critic**
(shared trunk, one value head per agent) — the standard CTDE pattern. Collision avoidance is
enforced by a **deterministic action-space safety layer** (the closing-speed brake), not
purely learned from reward — see `DECISIONS.md` for why that changed partway through this
project's history.

## Main objective

Train a policy that, at `NUM_AGENTS=4` (the primary validated agent count) and `NUM_AGENTS=3`
(the real deployment target — see below), maintains tight target tracking and formation
cohesion while avoiding both collisions and losing sensor contact with the target, and sustains
that quality over a realistic flight duration (90s), not just the original 10-second training
horizon. Collision avoidance at `N=4` **is resolved** (the closing-speed brake). `target_lost`
during active search — the swarm's response to fully losing contact — **has its root cause
identified but not yet fixed**: a deep investigation this pass (critic-collapse diagnostics,
then actor search-action dynamics, culminating in a decisive counterfactual experiment) proved
the failure is the **learned actor's own search execution**, not the environment, search
geometry, heading-assignment strategy, timeout length, or the critic — branching a simple
scripted controller from PPO's own real failure states reacquires the target 100% of the time
from states where PPO's own trajectory failed 100% of the time. **This is now the single most
important open problem in the project.** Separately, `NUM_AGENTS=3` — the actual deployment
target — currently cannot learn tracking at all under Phase 3's full mechanism set, at any
tested network size, a real blocker independent of the `target_lost` question above. See
`TODO.md`/`SESSION_HANDOFF.md`/`KNOWN_ISSUES.md` items 12 and 18.

## Current status (code as of commit `71327be` on `phase3-resilience`, 2026-08-26;
`origin/main` fast-forwarded to `8b724bb` on 2026-08-23 — no longer behind)

- **Collision avoidance is solid at `NUM_AGENTS=4`.** The closing-speed brake (a deterministic
  mechanism, not reward shaping) uses multi-pass POCS-style convergence plus a thresholded
  direct reward penalty on brake engagement — 3/3 `N=4` seeds clean at 0% `collision_rate` in
  the Phase2-combined validation. **A relative-velocity reformulation of the brake, built to
  close a known symmetry gap, was tested at full training scale and found net-harmful to
  convergence for a reason that was never identified — reverted.** The collision-safety cost
  that motivated it (8-14% under deterministic execution on well-converged checkpoints) remains
  real and unaddressed.
- **Target tracking uses a vision-based cooperative sensing model**, unchanged in mechanism:
  sensor-range-limited direct detection, instant swarm-wide sharing on contact, time-limited
  dead-reckoning, `target_lost` termination beyond the grace period (`LOST_TIMEOUT_SEC`,
  resolved default `6.0` after a sweep found no clean dose-response). **Active search** (each
  agent fans out on its own fixed heading when the swarm loses contact entirely) is a real,
  validated improvement — `contact_fraction` ~16-21%→85-90.7% at 6s, up to 99%+ for seeds that
  converge well at 8-10s.
- **Why active search still doesn't reach 100% is now understood, not just observed.** A
  supervisor-guided investigation traced the gap through several ruled-out candidates — the
  central critic's value predictions were found to collapse to a near-constant early in training
  (confirmed: its final Tanh layer saturates 100% within the first rollout), a real, reproducible
  pathology, but a matched full-scale ablation of the fix (a 30x-smaller critic learning rate)
  found it delays rather than prevents saturation and produces no reliable behavioral benefit —
  **rejected as a general fix, demoted to a secondary/contributing pathology.** A separate
  scripted-controller test (heading strategy A/B/C, timeout 6-12s) hit a 100% reacquisition
  ceiling regardless of condition, ruling out search geometry and timeout length as standalone
  bottlenecks — but only for well-formed starting states. **The decisive experiment** branched
  PPO / scripted / adaptive controllers from PPO's own *real* loss-onset states: from states
  where PPO's real trajectory failed, PPO fails again 100% of the time, but scripted and adaptive
  both reacquire 100% of the time, typically within 1-2 steps. Holding environment, geometry,
  timeout, and starting state exactly fixed, only the controller differs — **the problem is the
  learned actor's own search execution.** No fix has been designed or attempted yet.
- **Network capacity, measured for the first time.** A 5-way hidden-width sweep
  (`ACTOR_HIDDEN`/`CRITIC_HIDDEN`, now env-overridable) confirmed Phase 3's 128/256 default is a
  genuine sweet spot for `N=4` — a smaller network partially hurts, a larger one collapses
  learning almost entirely. **`NUM_AGENTS=3` — the real deployment target — failed to learn
  anything at all at either tested size**, the first time `N=3` was tried under Phase 3's full
  mechanism set with a *working* brake. Not yet explained; not network capacity (ruled out by the
  same sweep), not the brake formulation (failed under both). A real, currently-unexplained
  deployment blocker.
- **3D formation spread, N-aware safety margin, ground awareness, per-axis (Z vs XY) dynamics,
  cruise-altitude preference, and Z-velocity smoothing** — all unchanged since the last pass,
  still validated (ground/per-axis clean from the first test; the vertical-jiggling fix is
  implemented but still not re-measured against a trained checkpoint).

## Important technologies / dependencies

- Python 3.12, PyTorch, NumPy, Gymnasium, PettingZoo (`ParallelEnv`), pandas, matplotlib.
- Training run primarily on **Kaggle** (CPU, not GPU — see `DECISIONS.md` for why CPU beats
  CUDA here). Local dev machine has no working CUDA and is used for code edits, smoke tests
  (`test_env.py`), local diagnostic scripts, and log/plot review — the Kaggle API also works
  directly from the local machine (`~/.kaggle-venv`), so runs can be launched, polled, and their
  results downloaded without a manual notebook workflow. See `ENVIRONMENT.md`.
- See `ENVIRONMENT.md` for exact verified versions.

## What a new AI session should read, in order

1. This file.
2. `SESSION_HANDOFF.md` — what was happening right before the last session ended.
3. `CURRENT_STATE.md` — what's confirmed working vs. unverified vs. broken.
4. `ARCHITECTURE.md` if you need to touch code; `DECISIONS.md` before proposing any design
   change; `KNOWN_ISSUES.md` before investigating a bug that might already be tracked.

## A note on how this project has been worked on

Development happens through a tight loop of: propose a change → verify it against the actual
current code (not memory of past code) → make one isolated change (or a deliberately-scoped,
explicitly-reasoned bundle — see below) → smoke test locally → run on Kaggle → read back the
logged metrics, **both eval-time and training-time rolling-window data, since eval-time alone
has understated a real problem more than once** → decide next step. There is a recurring, now
well-established pattern in this project's history of an external/supervisor-style review, or
even a plausible-sounding internal hypothesis, turning out to be wrong (or at least
unverified) when actually measured — **six consecutive reward-shape fixes for the original
collision problem failed before the real cause was found by direct instrumentation**, and
later, a geometry bug shipped a full test cycle before a review caught that two different
angular quantities had been conflated. Treat any critique or hypothesis — including ones a
user pastes in, and including this project's own early guesses — as something to verify
against real data, not as ground truth.

**A third instance of this same lesson, this time an AI's own verbal explanation, not a shipped
bug**: investigating the train-vs-eval collision discrepancy (2026-08-22), a mid-session verbal
explanation of one seed's residual gap attributed a *different* run's `target_lost_rate` figure
to it, implying its tracking was less converged. It wasn't (that seed's `contact_fraction` was
99.7%, among the best in the project) — the mix-up came from two different experiments each
having a "seed2." Caught and corrected before it was written into the permanent record, but the
lesson generalizes: an explanation that sounds coherent and is offered confidently is not the
same as one that's been checked against the specific numbers it cites — re-verify a causal story
against the actual source data before treating it as settled, even (especially) your own.

**Deterministic execution can expose safety costs training-time numbers never show**, because a
policy shaped by exploration noise can converge its *mean* action to a riskier equilibrium than
the noise-perturbed behavior ever sat at. Trust deterministic eval for safety claims; see
`DECISIONS.md`.

**On bundling multiple changes into one tested run**: this has happened several times now, with
different outcomes worth distinguishing rather than treating as one lesson. `phase2-combined`
bundled two already-individually-tested pairs of changes for a final validation (lower risk);
Phase 3 bundled five genuinely-untested-together changes as an informed, time-constrained user
decision with an agreed fallback plan, and did have a real partial failure (tracking broke,
safety didn't) — exactly the attribution risk bundling creates, though diagnosable without
un-bundling in that case. The takeaway isn't "never bundle" — it's that bundling is a real,
named trade-off to make consciously and record the reasoning for, not a default to fall into
silently.

**Newest lesson (2026-08-25/26): a real, reproducible mechanism doesn't automatically mean
fixing it is the dominant behavioral lever — test the behavioral claim directly, not just the
mechanistic one.** Every diagnostic confirmed critic saturation was real; a full-scale ablation
of the fix still found no reliable behavioral benefit. Relatedly, a diagnostic that hits a 100%
ceiling can mean "this variable doesn't matter" or can mean "this test isn't reaching the states
that matter" — the search-strategy/timeout sweep's ceiling result only became informative once
paired with an experiment that branched from the learned policy's own real, harder states
instead of an idealized warmup. When a mechanism looks confirmed or a result looks saturated,
the next question is always "does this actually predict the behavior I care about," answered by
a direct test, not by the mechanism's plausibility alone.
