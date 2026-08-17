# AI Context — Start Here

## What this project is

A cooperative multi-agent reinforcement learning (MARL) project: `NUM_AGENTS` simulated
drones ("agents") learn, via CTDE-PPO (Centralized Training, Decentralized Execution), to
fly in a tight 3D formation while cooperatively tracking a moving target — sensed via a
**vision-based system**, not ground-truth telemetry (see below) — and avoiding collisions
with each other. The simulation is a lightweight custom physics/kinematics environment
written in Python (no ROS2/Gazebo/PX4 — there is no robotics-middleware or physics-engine
integration anywhere in this repo).

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
the target, via a curriculum first validated at `NUM_AGENTS=2`, then `3`. **As of this
writing, both the collision problem and the tracking problem have been solved and validated
at `NUM_AGENTS=3` — `NUM_AGENTS=4` has not yet been run against the current code. This is the
clear next step.** See `TODO.md`/`SESSION_HANDOFF.md`.

## Current status (as of commit `b59c139`, 2026-08-17)

- **Collision avoidance is solved via a deterministic mechanism, not reward shaping.** After
  six consecutive reward-shape/schedule fixes failed to stop a long-standing collision-rate
  collapse, direct instrumentation revealed the real cause (the policy committing to
  increasingly large-magnitude actions over training, not losing exploration noise as every
  prior fix had assumed). The fix — a closing-speed brake that caps only the velocity
  component actually closing a dangerous gap, independent of the reward — resolved it
  completely: `collision_rate=0.000` across every subsequent tested configuration but one rare
  edge case (~1%, see `KNOWN_ISSUES.md` item 8).
- **Target tracking now uses a vision-based cooperative sensing model, not ground-truth
  telemetry.** Every drone previously knew the target's exact position/velocity regardless of
  distance — an explicit but previously-unexamined simplification. Replaced with:
  sensor-range-limited direct detection, instant swarm-wide sharing on contact, time-limited
  dead-reckoning, and episode termination (`target_lost`) if the whole swarm loses contact too
  long. After one wrong turn (a leftover diameter-safety constraint from the pre-brake era
  turned out to be fighting the new sensor-range constraint) and confirming more training time
  helps, a 3-seed, 3M-step validation confirmed it: 0-3% target-lost rate, 0-1% collision
  rate, and the best tracking accuracy measured all session.
- **`NUM_AGENTS=2` and `NUM_AGENTS=3` configurations have completed training runs** at various
  points in this project's history — see `EXPERIMENT_LOG.md` for the specific ones still
  relevant (much of the very old N=2/N=3 data predates the collision fix and vision-tracking,
  and shouldn't be compared directly against current results without checking which commit
  produced it).

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
current code (not memory of past code) → make one isolated change → smoke test locally → run
on Kaggle → read back the logged metrics → decide next step. There is a recurring, now
well-established pattern in this project's history of an external/supervisor-style review, or
even a plausible-sounding internal hypothesis, turning out to be wrong when actually measured
— **six consecutive reward-shape fixes for the collision problem failed before the real cause
was found by direct instrumentation, not reasoning from first principles.** Treat any
critique or hypothesis — including ones a user pastes in, and including this session's own
early guesses — as something to verify against real data, not as ground truth. When multiple
changes get bundled into one tested commit (it happened twice this session, both times
correctly flagged as a discipline lapse afterward), attribution gets weaker — prefer isolating
one variable per test even when it's tempting to bundle a "free" fix alongside a real one.
