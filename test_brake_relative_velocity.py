"""Adversarial regression test for _apply_brake's closing-speed guarantee --
originally built to verify the relative-velocity reformulation (see
envs/formation_env.py's step() CAUTION comment), which was later tried,
found to break training convergence at Kaggle scale for reasons still not
understood, and reverted. Kept as a general regression test: both scenarios
are exactly symmetric (every agent runs the identical formula), which is
precisely the condition under which the reverted one-sided formula was
already proven to hold too -- see DECISIONS.md. Still passes against
whichever formula is currently active; re-verify after any future brake
change, not just this one.

Both scenarios start every agent outside SAFE_DIST_ENTER, always commanding
MAX_ACTION_SPEED directly at another agent/the shared centroid, every step,
forever -- an adversarial input _apply_brake must never be allowed to fail
under, not a realistic policy trajectory. Pass criteria, checked every step:
  1. minimum pairwise distance never drops below COLLISION_DIST.
  2. the post-hoc violation diagnostic (_last_brake_violation) stays ~0 --
     confirms the multi-pass loop is actually converging each step, not
     silently hitting its NUM_AGENTS-pass cap with residual violation.
"""
import numpy as np

from envs.formation_env import FormationEnv3D
from config import COLLISION_DIST, SAFE_DIST_ENTER, MAX_ACTION_SPEED, DT

STEPS = 300
VIOLATION_TOL = 1e-6
# The closing-speed cap forces distance-to-COLLISION_DIST to shrink
# geometrically each step (ratio ~0.9-0.95 here) -- within a couple hundred
# steps the true mathematical margin is already far below float32's ~1e-6
# relative precision at this magnitude, so a tolerance tighter than that
# would be checking float32 rounding noise, not real brake behavior. 1e-3
# is a thousand times looser than that floor while still catching any
# actual, physically meaningful breach.
DIST_TOL = 1e-3


def run_adversarial(label, num_agents, positions, command_fn):
    """command_fn(pos_dict, agent_name) -> unit direction this agent always
    commands MAX_ACTION_SPEED toward (recomputed each step from the agent's
    current position, so it keeps aiming at a moving target/centroid)."""
    env = FormationEnv3D(num_agents=num_agents, k_neighbors=num_agents - 1)
    env.pos = {a: np.array(p, dtype=np.float32) for a, p in zip(env.agents, positions)}

    min_dist_ever = float("inf")
    max_violation_ever = 0.0
    for step in range(STEPS):
        raw_vel = {}
        for a in env.agents:
            direction = command_fn(env.pos, a)
            raw_vel[a] = (direction * MAX_ACTION_SPEED).astype(np.float32)

        corrected, _ = env._apply_brake(raw_vel)
        max_violation_ever = max(max_violation_ever, env._last_brake_violation)

        for a in env.agents:
            env.pos[a] = (env.pos[a] + corrected[a] * DT).astype(np.float32)

        dists = [float(np.linalg.norm(env.pos[a] - env.pos[b]))
                 for i, a in enumerate(env.agents) for b in env.agents[i + 1:]]
        min_dist_ever = min(min_dist_ever, min(dists))

        assert min(dists) >= COLLISION_DIST - DIST_TOL, (
            f"{label}: COLLISION_DIST breached at step {step}: min_dist={min(dists):.4f} "
            f"< COLLISION_DIST={COLLISION_DIST}"
        )
        assert env._last_brake_violation <= VIOLATION_TOL, (
            f"{label}: brake left residual violation {env._last_brake_violation:.6f} "
            f"at step {step} (tolerance {VIOLATION_TOL})"
        )

    print(f"{label}: OK over {STEPS} steps -- min_dist_ever={min_dist_ever:.4f} "
          f"(COLLISION_DIST={COLLISION_DIST}), max_violation_ever={max_violation_ever:.2e}")


# Test 1: 2 agents head-on, closing speed MAX_ACTION_SPEED each, forever.
# Start separation chosen so they begin outside SAFE_DIST_ENTER.
start_sep = SAFE_DIST_ENTER + 6.0
run_adversarial(
    "2-agent head-on",
    num_agents=2,
    positions=[(-start_sep / 2, 0, 0), (start_sep / 2, 0, 0)],
    command_fn=lambda pos, a: (pos["drone2" if a == "drone1" else "drone1"] - pos[a])
                                / (np.linalg.norm(pos["drone2" if a == "drone1" else "drone1"] - pos[a]) + 1e-9),
)

# Test 2: 4 agents evenly spaced on a circle, all converging on the shared
# centroid (the origin). Radius chosen so the minimum PAIRWISE distance
# (adjacent points, R*sqrt(2) apart) starts outside SAFE_DIST_ENTER too --
# the geometrically tightest, most adversarial starting arrangement outside
# the safety zone.
R = SAFE_DIST_ENTER / (2 ** 0.5) + 2.0
angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
start_positions = [(R * np.cos(t), R * np.sin(t), 0.0) for t in angles]
run_adversarial(
    "4-agent converge-on-centroid",
    num_agents=4,
    positions=start_positions,
    command_fn=lambda pos, a: (-pos[a]) / (np.linalg.norm(pos[a]) + 1e-9),
)

print("OK -- relative-velocity _apply_brake holds both adversarial scenarios.")
