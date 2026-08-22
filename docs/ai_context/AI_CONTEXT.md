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

Train a policy that, at `NUM_AGENTS=4` (the target agent count), maintains tight target
tracking and formation cohesion while avoiding both collisions and losing sensor contact with
the target — and, as of Phase 3, sustains that quality over a realistic flight duration (90s),
not just the original 10-second training horizon. The original `N=4` collision problem (brake
gaps under simultaneous multi-threat braking) **is now resolved** — see `DECISIONS.md`.
**Phase 3's `target_lost` catastrophe (89-100%) is substantially better but not solved**:
active search (each agent fans out on its own when the swarm loses contact, instead of everyone
coasting toward the same stale point) got it down to 60-81% at the original 6s timeout, and to
~2-4% for seeds that converge well at a longer (8-10s) timeout — but training is seed-sensitive,
and 2 of 3 seeds tested at 10s got stuck in a much worse regime. **The current headline open
problem is different again**: when a seed *does* converge to that tight, confident tracking
behavior, deterministic execution (what a real flight controller actually runs) exposes a real
collision-safety cost (8-14%) the training-time numbers never showed. The brake has been
reformulated to use true relative velocity instead of each agent's own — adversarially verified,
Kaggle-scale confirmation in progress. See `TODO.md`/`SESSION_HANDOFF.md`/`KNOWN_ISSUES.md`
items 12-13.

## Current status (code as of commit `94216fd`/`3eae55b` on `phase3-resilience`, 2026-08-22;
`main` still at `b59c139`, well behind — see `TODO.md`)

- **Collision avoidance is solid at both `NUM_AGENTS=3` and `4`.** The closing-speed brake
  (a deterministic mechanism, not reward shaping) was extended from a single sequential
  correction pass to multi-pass POCS-style convergence, plus a thresholded direct reward
  penalty on brake engagement — together these closed the one confirmed gap in the original
  mechanism (simultaneous multi-threat braking at `N=4`). A 5-seed validation across
  `N=2/3/4` showed 3/3 `N=4` seeds clean at 0% `collision_rate`, both at eval-time and in the
  training-time rolling-window data that previously told a much worse story. See
  `DECISIONS.md`/`EXPERIMENT_LOG.md`.
- **Target tracking uses a vision-based cooperative sensing model**, unchanged in mechanism
  from the original redesign: sensor-range-limited direct detection, instant swarm-wide
  sharing on contact, time-limited dead-reckoning, `target_lost` termination beyond the grace
  period. Clean at `N=2/3/4` under the original 10-second episode length. **Breaks badly at
  the new 90-second episode length** (see below) — the mechanism itself didn't change, but a
  constant tied to it stopped being valid once episodes got 9x longer.
- **3D formation spread and an N-aware safety margin** were added on top of the brake fixes,
  validated in the same combined sweep — the best `N=4` tracking accuracy of the project to
  date (`tracking_rmse` averaging 1.91). A real geometry bug (conflating two different angular
  quantities) was caught by review before being trusted, fixed, and given a permanent
  regression test. See `DECISIONS.md`.
- **Ground awareness, per-axis (Z vs XY) dynamics, dynamic target motion, and longer (90s)
  episodes** were added together as Phase 3, an explicit "leap of faith" bundle directly
  motivated by real evidence: a sustained-flight diagnostic showed tracking error nearly
  quadrupling once a checkpoint flew longer than its training horizon, in a time window that
  matches where a real PX4/Gazebo deployment (a separate, concurrent workstream — see
  `ARCHITECTURE.md`) actually crashed. Ground awareness worked perfectly from the first test
  (zero ground strikes across every validation rollout). **Target tracking did not** — see
  below.
- **`target_lost_rate`: a plain `LOST_TIMEOUT_SEC` sweep (6s/10s/18s) found no clean
  dose-response** — 3 of 5 runs stayed catastrophically broken regardless of timeout length.
  Default raised to 6.0 anyway; the real lever turned out to be **active search**, not the
  timeout constant.
- **Active search, implemented and validated as a real, major improvement.** Each agent now
  fans out in its own fixed heading when the whole swarm loses contact, instead of every agent
  coasting toward the same stale point — `contact_fraction` improved from ~16-21% to 85-90.7%
  at the 6s timeout. A follow-up 8s/10s timeout bracket showed the mechanism can reach 99%+
  contact / ~2-4% `target_lost_rate` when a seed converges well, but training is seed-sensitive
  (2 of 3 `t10` seeds got stuck worse). Resuming the stuck seeds to 5M steps confirmed "more
  training helps" but didn't fully close the gap. See `KNOWN_ISSUES.md` item 12,
  `EXPERIMENT_LOG.md`.
- **A separately-reported vertical-jiggling issue was confirmed and fixed in code**, bundled
  into the same branch/sweep under time pressure — still not re-verified against a trained
  checkpoint. See `KNOWN_ISSUES.md` item 14.
- **New: deterministic execution (what real deployment runs) exposes a collision-safety cost
  training-time numbers never showed.** Investigated directly — running the same frozen
  checkpoint deterministically vs. stochastically on identical eval seeds showed determinism
  alone can explain most/all of an observed gap (a policy's mean action can converge to a
  knife-edge equilibrium the noise-perturbed training behavior never sat at). Deterministic
  eval numbers, not training-time rolling numbers, are now the trusted safety signal. See
  `DECISIONS.md`, `KNOWN_ISSUES.md` item 16.
- **The brake's known relative-velocity gap (previously theoretical) has been reformulated and
  adversarially verified — Kaggle-scale confirmation in progress.** `v_closing` now uses the
  true mutual closing rate, not just one agent's own velocity. See `KNOWN_ISSUES.md` item 13.

## Important technologies / dependencies

- Python 3.12, PyTorch, NumPy, Gymnasium, PettingZoo (`ParallelEnv`), pandas, matplotlib.
- Training run primarily on **Kaggle** (CPU, not GPU — see `DECISIONS.md` for why CPU beats
  CUDA here). Local dev machine has no working CUDA and is used for code edits, smoke tests
  (`test_env.py`), and log/plot review — but as of this session, the Kaggle API also works
  directly from the local machine (`~/.kaggle-venv`), so runs can be launched, polled, and
  their results downloaded without a manual notebook workflow. See `ENVIRONMENT.md`.
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

**Also new this pass, a genuinely new class of finding, not a bug**: deterministic execution
(what real deployment runs) can expose safety costs training-time numbers never show, because a
policy shaped by exploration noise can converge its *mean* action to a riskier equilibrium than
the noise-perturbed behavior ever sat at. This isn't "eval-time understates, training overstates"
like the earlier `N=4` brake lesson — it's closer to the opposite: training's own numbers can be
optimistic in a way that only shows up once the noise is removed. Trust deterministic eval for
safety claims; see `DECISIONS.md`.

**On bundling multiple changes into one tested run**: this has happened several times now,
with different outcomes worth distinguishing rather than treating as one lesson. Early on it
happened by accident (three changes in one commit, correctly flagged afterward as a discipline
lapse — attribution for that specific commit is genuinely weaker as a result). Later it
happened deliberately and was reasoned about explicitly in `DECISIONS.md` each time:
`phase2-combined` bundled two already-individually-tested pairs of changes for a final
validation pass (lower risk — each half was already checked alone); Phase 3 bundled five
genuinely-untested-together changes as an informed, time-constrained user decision with an
agreed fallback plan. Phase 3's bundle then had a real partial failure (tracking broke, safety
didn't) — exactly the attribution risk bundling creates, though in that specific case the
failure turned out to be diagnosable without un-bundling anyway (see `EXPERIMENT_LOG.md`'s
Phase 3 entry). The takeaway isn't "never bundle" — it's that bundling is a real, named
trade-off to make consciously and record the reasoning for, not a default to fall into
silently.
