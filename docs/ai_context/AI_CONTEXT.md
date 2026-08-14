# AI Context — Start Here

## What this project is

A cooperative multi-agent reinforcement learning (MARL) project: `NUM_AGENTS` simulated
drones ("agents") learn, via CTDE-PPO (Centralized Training, Decentralized Execution), to
fly in a tight 3D formation while tracking a moving virtual target and avoiding collisions
with each other. The simulation is a lightweight custom physics/kinematics environment
written in Python (no ROS2/Gazebo/PX4 — there is no robotics-middleware or physics-engine
integration anywhere in this repo; if that's expected, it hasn't been built yet).

The environment (`envs/formation_env.py`, class `FormationEnv3D`) implements the PettingZoo
`ParallelEnv` API. Training (`training/train.py`) uses a **single shared actor network**
(parameter sharing across all homogeneous drones) plus a **centralized multi-agent critic**
(shared trunk, one value head per agent) — the standard CTDE pattern. Execution is
decentralized in the sense that each agent only observes itself plus its `K_NEIGHBORS`
nearest neighbors, but neighbor selection itself is computed centrally by the simulator (see
`ARCHITECTURE.md` and `DECISIONS.md` for why, and what caveat that implies).

## Main objective

Train a policy that, at `NUM_AGENTS=4` (the current target agent count — see
`config.py:9`), achieves a low collision rate while maintaining tight target tracking and
formation cohesion, and to do so via a curriculum that was first validated at
`NUM_AGENTS=2`, then `NUM_AGENTS=3`, before scaling to 4. See `DECISIONS.md` for why this
curriculum is "staged comparative validation" and not literal weight transfer.

## Current status (as of commit `b14fe48`, 2026-08-14)

- `NUM_AGENTS=2` and `NUM_AGENTS=3` configurations have completed training runs with
  results reviewed by the user (see `EXPERIMENT_LOG.md`).
- The most recent code changes (commits `a40db9b` through `63274b1`) were a chain of
  reward-shaping and stability fixes: log_std clamping, per-component reward logging,
  reward reweighting (track x4, safety magnitude down), replacing a per-neighbor diverge
  penalty with a global cohesion penalty (to fix a "relock loophole"), entropy annealing
  fixes, and fixing a cohesion/safety conflict that was causing collision-rate collapse.
  **These fixes are UNVERIFIED against a full training run at the time this doc was
  written** — see `SESSION_HANDOFF.md` for the exact pending item.
- An `NUM_AGENTS=4`, 3-seed training run was reportedly launched on Kaggle (per prior
  session) to validate this latest chain of fixes; its results were pending as of the last
  known exchange. **Check `SESSION_HANDOFF.md` for whether results have since arrived.**

## Important technologies / dependencies

- Python 3.12, PyTorch, NumPy, Gymnasium, PettingZoo (`ParallelEnv`), pandas, matplotlib.
- Training run primarily on **Kaggle** (CPU, not GPU — see `DECISIONS.md` for why CPU beats
  CUDA here). Local dev machine (this sandbox) has no working CUDA and is used for code
  edits, smoke tests (`test_env.py`), and log/plot review, not full training runs.
- See `ENVIRONMENT.md` for exact verified versions.

## What a new AI session should read, in order

1. This file.
2. `SESSION_HANDOFF.md` — what was happening right before the last session ended.
3. `CURRENT_STATE.md` — what's confirmed working vs. unverified vs. broken.
4. `ARCHITECTURE.md` if you need to touch code; `DECISIONS.md` before proposing any design
   change; `KNOWN_ISSUES.md` before investigating a bug that might already be tracked.

## A note on how this project has been worked on

Development happens through an unusually tight loop of: propose a change → verify it
against the actual current code (not memory of past code) → make one isolated change →
smoke test locally → run on Kaggle → read back the logged metrics → decide next step. There
is a recurring pattern in this project's history of an external/supervisor-style review
making a claim about the code that turned out to be stale or wrong when checked against the
actual file (e.g. a claimed-missing advantage normalization that was already present in
`training/buffer.py:54`). Treat any critique — including ones a user pastes in — as a
hypothesis to verify against the real file, not as ground truth.
