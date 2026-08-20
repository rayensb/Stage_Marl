#!/usr/bin/env python3
"""Phase 7 -- observation adapter verification (no ROS2/PX4/Gazebo needed).

deployment/inference_node.py calls envs.formation_env.FormationEnv3D._get_obs()
directly -- there is no second, reimplemented observation formula to diff
against, so that formula is out of scope here (it's shared, proven code,
already exercised cleanly by sim_demo.py). The only place a Phase-7-class bug
can hide is in the ~20 lines of _timer_callback (inference_node.py:368-386)
that turn raw PX4 telemetry into that env's pos/vel/pos_t/locked state before
_get_obs() runs. This script exercises that exact transformation -- mirrored
below as literally as possible (see apply_telemetry()), since it's inline in
a ROS2 callback with no importable standalone form -- against synthetic
telemetry, and checks results against independently-derived expectations
(scenario B checks the GLUE's output state directly, not a re-derivation of
_get_obs's formula) plus internal-consistency checks using the real,
unmodified _get_obs()/_update_target_track() (scenarios A/A2/C/D/E).

Run (ROS2 must be sourced -- inference_node.py imports rclpy/px4_msgs at
module level -- but no daemon/simulator/network connection is needed):
    source /opt/ros/jazzy/setup.bash
    source /home/rayen/ws_sensor_combined/install/setup.bash
    NUM_AGENTS=3 python3 deployment/verify_obs_adapter.py [--out summary.json]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("NUM_AGENTS") != "3":
    sys.exit(
        "verify_obs_adapter.py must be run with NUM_AGENTS=3 in the environment\n"
        "(config.py and inference_node.py both read it at import time), e.g.:\n"
        "  NUM_AGENTS=3 python3 deployment/verify_obs_adapter.py"
    )

import argparse
import json
import math

import numpy as np

from config import (
    NUM_AGENTS, K_NEIGHBORS, OBS_DIM, TARGET_DIST, EDGE_TARGET,
    SENSOR_RANGE, OBS_MAX_DIST, OBS_MAX_VEL,
)
from envs.formation_env import FormationEnv3D
from deployment.inference_node import (
    ned_to_sim, sim_to_ned, SPAWN_OFFSET, SPAWN_XY, TARGET_START,
    PHANTOM_DIR, PHANTOM_SPEED, ALL_AGENTS,
)

# Mirrors FormationEnv3D.__init__ (envs/formation_env.py:81).
NEIGHBOR_K = min(K_NEIGHBORS, NUM_AGENTS - 1)

OBS_NAMES = [
    "vx", "vy", "vz",
    "rel_target_x", "rel_target_y", "rel_target_z", "dist_target",
    "rel_tgt_vel_x", "rel_tgt_vel_y", "rel_tgt_vel_z",
    "has_contact", "track_confidence", "contact_age_norm", "observer_count_norm",
    "rel_centroid_x", "rel_centroid_y", "rel_centroid_z", "centroid_valid",
]
for _i in range(NEIGHBOR_K):
    OBS_NAMES += [
        f"nbr{_i}_rel_pos_x", f"nbr{_i}_rel_pos_y", f"nbr{_i}_rel_pos_z",
        f"nbr{_i}_rel_vel_x", f"nbr{_i}_rel_vel_y", f"nbr{_i}_rel_vel_z",
        f"nbr{_i}_dist",
    ]
assert len(OBS_NAMES) == OBS_DIM, f"OBS_NAMES has {len(OBS_NAMES)} entries but OBS_DIM={OBS_DIM}"


def print_obs(obs, label=""):
    if label:
        print(f"  --- {label} ---")
    for i, (name, val) in enumerate(zip(OBS_NAMES, obs)):
        print(f"  {i:2d} | {name:22s} | {float(val): .5f}")


def make_env():
    """Mirrors MarlInferenceNode.__init__ (inference_node.py:220-224) exactly."""
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    env.agents = env.possible_agents[:]
    env.locked = {a: [o for o in ALL_AGENTS if o != a] for a in ALL_AGENTS}
    env.pos = {a: np.zeros(3, np.float32) for a in ALL_AGENTS}
    env.vel = {a: np.zeros(3, np.float32) for a in ALL_AGENTS}
    return env


def apply_telemetry(env, agent, pos, vel, neighbor_pos, neighbor_vel, target_pos):
    """Mirrors inference_node.py's _timer_callback state-mutation block
    (lines 368-386) exactly, minus the ROS/PX4-message unpacking -- callers
    here pass already-sim-frame pos/vel directly, since the NED conversion
    itself is separately tested in scenario C."""
    other_agents = [a for a in ALL_AGENTS if a != agent]
    env.pos[agent] = pos
    env.vel[agent] = vel
    for a in other_agents:
        env.pos[a] = neighbor_pos[a]
        env.vel[a] = neighbor_vel[a]
    env.locked[agent] = sorted(
        other_agents,
        key=lambda a: float(np.linalg.norm(env.pos[a] - env.pos[agent])),
    )
    env.pos_t = target_pos
    env._target_dir = PHANTOM_DIR
    env._target_speed = PHANTOM_SPEED
    env._update_target_track()


def _sanity_check_constants():
    env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
    assert env.k == NEIGHBOR_K, f"env.k={env.k} != computed NEIGHBOR_K={NEIGHBOR_K}"
    assert env._obs_dim == OBS_DIM == len(OBS_NAMES), (
        f"env._obs_dim={env._obs_dim} config.OBS_DIM={OBS_DIM} len(OBS_NAMES)={len(OBS_NAMES)}"
    )
    print(f"Sanity: NUM_AGENTS={NUM_AGENTS} K_NEIGHBORS={K_NEIGHBORS} env.k={env.k} OBS_DIM={OBS_DIM}")


def scenario_a():
    print("\n=== Scenario A: converged-formation sanity ===")
    target = TARGET_START.copy()
    positions = {}
    for i, a in enumerate(ALL_AGENTS):
        ang = 2 * math.pi * i / len(ALL_AGENTS)
        positions[a] = target + np.array(
            [TARGET_DIST * math.cos(ang), TARGET_DIST * math.sin(ang), 0.0], np.float32
        )
    zero_v = np.zeros(3, np.float32)
    expected_dist = TARGET_DIST / OBS_MAX_DIST
    expected_nbr_dist = EDGE_TARGET / OBS_MAX_DIST

    env = make_env()
    ok = True
    for a in ALL_AGENTS:
        others = [o for o in ALL_AGENTS if o != a]
        apply_telemetry(
            env, a, positions[a], zero_v,
            {o: positions[o] for o in others}, {o: zero_v for o in others},
            target,
        )
        obs = env._get_obs(a)
        dist_ok = math.isclose(obs[6], expected_dist, abs_tol=1e-4)
        contact_ok = obs[10] == 1.0
        nbr_dist_ok = all(
            math.isclose(obs[18 + 7 * j + 6], expected_nbr_dist, abs_tol=1e-4)
            for j in range(env.k)
        )
        agent_ok = dist_ok and contact_ok and nbr_dist_ok
        ok = ok and agent_ok
        print_obs(obs, label=f"{a} ({'PASS' if agent_ok else 'FAIL'})")
        print(f"    dist_target: got {obs[6]:.5f} expected {expected_dist:.5f} [{'ok' if dist_ok else 'FAIL'}]")
        print(f"    has_contact: got {obs[10]:.1f} expected 1.0 [{'ok' if contact_ok else 'FAIL'}]")
        print(f"    neighbor dist both slots == {expected_nbr_dist:.5f}: [{'ok' if nbr_dist_ok else 'FAIL'}]")
    print(f"Scenario A: {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_a2():
    print("\n=== Scenario A2: defensive zero-init reachability (informational) ===")
    spawn_radius = math.hypot(*SPAWN_XY["drone1"])
    margin = SENSOR_RANGE - spawn_radius
    print(f"  SENSOR_RANGE={SENSOR_RANGE:.3f}  current spawn radius={spawn_radius:.3f}  margin={margin:.3f}m")
    print("  (given this margin, the 'no contact yet' branch below is unreachable in today's real deployment)")

    env = make_env()
    fixed_positions = {
        "drone1": np.array([0.0, 0.0, 3.0], np.float32),
        "drone2": np.array([1.0, 1.0, 3.0], np.float32),
        "drone3": np.array([-1.0, 1.0, 3.0], np.float32),
    }
    for a in ALL_AGENTS:
        env.pos[a] = fixed_positions[a]
    env.pos_t = np.array([1000.0, 1000.0, 1000.0], np.float32)  # forces zero contact for everyone
    env._target_dir = PHANTOM_DIR
    env._target_speed = PHANTOM_SPEED
    env._update_target_track()

    print(f"  [forced/artificial -- not a real reachable code path today] first _update_target_track()")
    print(f"  call with zero contact on a never-reset() env -> track_pos_est = {env._track_pos_est}")
    print(f"  (the __init__ defensive-zero default, NOT TARGET_START={TARGET_START} -- silently wrong")
    print(f"   if it ever fired for a TARGET_START with nonzero X/Y, though it doesn't crash)")
    obs = env._get_obs(ALL_AGENTS[0])
    finite_and_bounded = bool(np.all(np.isfinite(obs)) and np.all(obs >= -1.0) and np.all(obs <= 1.0))
    print(f"  resulting obs finite and within [-1,1]: {finite_and_bounded} (no crash either way)")
    return True  # informational only, never fails the run


def scenario_b():
    print("\n=== Scenario B: general asymmetric state (checks the GLUE, not the shared formula) ===")
    positions = {
        "drone1": np.array([2.0, -1.0, 3.5], np.float32),
        "drone2": np.array([5.0, 3.0, 2.0], np.float32),
        "drone3": np.array([-1.0, 2.0, 4.0], np.float32),
    }
    velocities = {
        "drone1": np.array([0.5, -0.2, 0.1], np.float32),
        "drone2": np.array([-0.3, 0.4, 0.0], np.float32),
        "drone3": np.array([0.1, 0.1, -0.2], np.float32),
    }
    target = np.array([1.0, 0.0, 3.0], np.float32)
    other = [a for a in ALL_AGENTS if a != "drone1"]

    env = make_env()
    apply_telemetry(
        env, "drone1", positions["drone1"], velocities["drone1"],
        {a: positions[a] for a in other}, {a: velocities[a] for a in other}, target,
    )

    checks = []
    checks.append(("env.pos[drone1] == input pos", bool(np.allclose(env.pos["drone1"], positions["drone1"]))))
    for a in other:
        checks.append((f"env.pos[{a}] == input pos", bool(np.allclose(env.pos[a], positions[a]))))
        checks.append((f"env.vel[{a}] == input vel", bool(np.allclose(env.vel[a], velocities[a]))))
    checks.append(("env.pos_t == target", bool(np.allclose(env.pos_t, target))))

    expected_order = sorted(other, key=lambda a: float(np.linalg.norm(positions[a] - positions["drone1"])))
    checks.append((f"env.locked[drone1] nearest-first == {expected_order}", env.locked["drone1"] == expected_order))

    dists = {a: float(np.linalg.norm(positions[a] - target)) for a in ALL_AGENTS}
    expected_contacts = {a: dists[a] <= SENSOR_RANGE for a in ALL_AGENTS}
    checks.append(("_direct_contact matches SENSOR_RANGE threshold", env._direct_contact == expected_contacts))
    if any(expected_contacts.values()):
        checks.append(("_track_pos_est == target (in-contact case, elapsed=0)", bool(np.allclose(env._track_pos_est, target))))

    # Two small, independently hand-computed numeric spot checks on the
    # actual obs vector (not a full re-derivation of _get_obs's formula --
    # that formula is shared/proven code, out of scope here).
    obs = env._get_obs("drone1")
    expected_dist_target = float(np.linalg.norm(target - positions["drone1"])) / OBS_MAX_DIST
    expected_rel_target_z = (target[2] - positions["drone1"][2]) / OBS_MAX_DIST
    checks.append((f"obs[6] dist_target == {expected_dist_target:.5f}", math.isclose(obs[6], expected_dist_target, abs_tol=1e-4)))
    checks.append((f"obs[5] rel_target_z == {expected_rel_target_z:.5f}", math.isclose(obs[5], expected_rel_target_z, abs_tol=1e-4)))

    ok = all(passed for _, passed in checks)
    for name, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
    print_obs(obs, label="drone1 (formula trusted -- shared/proven code; printed per Phase 7's ask)")

    ok_fast = _check_velocity_clipping(positions, velocities, target, other)
    result = ok and ok_fast
    print(f"Scenario B: {'PASS' if result else 'FAIL'}")
    return result


def _check_velocity_clipping(positions, velocities, target, other):
    print("\n  -- velocity-clipping sub-case (drone1 vel = 3x OBS_MAX_VEL in +X) --")
    fast_velocities = dict(velocities)
    fast_velocities["drone1"] = np.array([3 * OBS_MAX_VEL, 0.0, 0.0], np.float32)

    env = make_env()
    apply_telemetry(
        env, "drone1", positions["drone1"], fast_velocities["drone1"],
        {a: positions[a] for a in other}, {a: fast_velocities[a] for a in other}, target,
    )
    obs = env._get_obs("drone1")
    vx = float(obs[0])
    in_bounds = bool(np.all(obs >= -1.0) and np.all(obs <= 1.0))
    clipped_correctly = math.isclose(vx, 1.0, abs_tol=1e-6)
    print(f"    raw vel={fast_velocities['drone1'][0]:.2f} m/s ({fast_velocities['drone1'][0] / OBS_MAX_VEL:.1f}x OBS_MAX_VEL) "
          f"-> obs[0]={vx:.5f} (expected exactly 1.0, clipped not wrapped)")
    print(f"    entire obs vector within [-1,1]: {in_bounds}")
    ok = in_bounds and clipped_correctly
    print(f"  velocity-clipping sub-case: {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_c():
    print("\n=== Scenario C: NED->sim + SPAWN_OFFSET round-trip ===")
    ned_pos = [1.0, 2.0, -4.0]   # NED: 4m altitude (D is down-positive, so -4.0 == 4m up)
    ned_vel = [0.5, -0.3, 0.1]
    agent = "drone2"

    sim_pos = ned_to_sim(ned_pos) + SPAWN_OFFSET[agent]
    sim_vel = ned_to_sim(ned_vel)   # velocity is offset-invariant -- no SPAWN_OFFSET added

    expected_pos = np.array([1.0, 2.0, 4.0], np.float32) + SPAWN_OFFSET[agent]
    expected_vel = np.array([0.5, -0.3, -0.1], np.float32)

    pos_ok = bool(np.allclose(sim_pos, expected_pos))
    vel_ok = bool(np.allclose(sim_vel, expected_vel))
    z_offset_ok = bool(SPAWN_OFFSET[agent][2] == 0.0)
    roundtrip_ok = bool(np.allclose(sim_to_ned(ned_to_sim(ned_pos)), ned_pos))

    print(f"  position: ned={ned_pos} -> sim={sim_pos} (expected {expected_pos}) [{'ok' if pos_ok else 'FAIL'}]")
    print(f"  velocity: ned={ned_vel} -> sim={sim_vel} (expected {expected_vel}, no SPAWN_OFFSET applied) [{'ok' if vel_ok else 'FAIL'}]")
    print(f"  SPAWN_OFFSET[{agent}][2] == 0.0 (Z never offset): {SPAWN_OFFSET[agent][2]} [{'ok' if z_offset_ok else 'FAIL'}]")
    print(f"  sim_to_ned(ned_to_sim(x)) round-trips to original NED vector: [{'ok' if roundtrip_ok else 'FAIL'}]")

    ok = pos_ok and vel_ok and z_offset_ok and roundtrip_ok
    print(f"Scenario C: {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_d():
    print("\n=== Scenario D: clock-skew quantification (informational -- not pass/fail) ===")
    # Mirrors _current_target_pos()'s formula (inference_node.py:259-261);
    # that method itself needs a live rclpy Node clock and can't be imported
    # standalone, so the one-line formula is duplicated here against the
    # real imported TARGET_START/PHANTOM_DIR/PHANTOM_SPEED constants.
    skews = [0.3, 0.6, 1.0, 2.0]
    for T in (50.0, 200.0):
        print(f"  -- shared wall-clock instant T={T:.0f}s since drone1 (target owner) completed takeoff --")
        for skew in skews:
            pos_true = TARGET_START + PHANTOM_DIR * PHANTOM_SPEED * T
            pos_skewed = TARGET_START + PHANTOM_DIR * PHANTOM_SPEED * (T - skew)
            disagreement = float(np.linalg.norm(pos_true - pos_skewed))
            normalized = disagreement / OBS_MAX_DIST
            print(f"    skew={skew:.1f}s  disagreement={disagreement:.4f}m  normalized={normalized:.5f}")

    print("  -- does this flip contact/observer/centroid flags near the sensor-range boundary? --")
    T = 50.0
    true_target = TARGET_START + PHANTOM_DIR * PHANTOM_SPEED * T
    for label, radius in (("converged formation (r=TARGET_DIST)", TARGET_DIST), ("near-SENSOR_RANGE boundary (r=6.7)", 6.7)):
        p = true_target + np.array([radius, 0.0, 0.0], np.float32)
        for skew in (0.6, 2.0):
            skewed_target = TARGET_START + PHANTOM_DIR * PHANTOM_SPEED * (T - skew)
            d_true = float(np.linalg.norm(p - true_target))
            d_skewed = float(np.linalg.norm(p - skewed_target))
            flips = (d_true <= SENSOR_RANGE) != (d_skewed <= SENSOR_RANGE)
            print(f"    {label}, skew={skew:.1f}s: dist_true={d_true:.3f} dist_skewed={d_skewed:.3f} "
                  f"contact_flips={'YES' if flips else 'no'}")

    print("  -- asymmetry: drone1 (target owner) sets the real Gazebo marker's pose/velocity at ITS")
    print("     OWN takeoff completion, so its belief is exact by construction (skew=0 always) --")
    print("     only drone2/drone3 are exposed to any skew at all.")
    print("  -- conclusion: the disagreement above is CONSTANT vs. elapsed time (T=50 vs T=200 give")
    print("     identical numbers), not growing, and per PHASE2_HANDOFF.md, drone1 -- the drone with")
    print("     PROVABLY ZERO skew -- crashed hardest, while drone2/drone3 (the only ones exposed to")
    print("     skew) merely sank gently. This argues against clock skew as the dominant driver of the")
    print("     observed divergence. It is still a real, minor Rule-4 training-contract violation")
    print("     (deployment has 3 independent pos_t beliefs; training has exactly one, shared) worth")
    print("     fixing eventually, just not the leading suspect for this failure.")
    return True  # informational only


def scenario_e():
    print("\n=== Scenario E: neighbor ordering (nearest-first) ===")
    env = make_env()
    positions = {
        "drone1": np.array([0.0, 0.0, 3.0], np.float32),
        "drone2": np.array([3.0, 0.0, 3.0], np.float32),   # distance 3 from drone1
        "drone3": np.array([6.0, 0.0, 3.0], np.float32),   # distance 6 from drone1
    }
    zero_v = np.zeros(3, np.float32)
    target = np.array([0.5, 0.0, 3.0], np.float32)
    other = ["drone2", "drone3"]
    apply_telemetry(
        env, "drone1", positions["drone1"], zero_v,
        {a: positions[a] for a in other}, {a: zero_v for a in other}, target,
    )

    order_ok = env.locked["drone1"] == ["drone2", "drone3"]
    obs = env._get_obs("drone1")
    slot0_dist = float(obs[18 + 6]) * OBS_MAX_DIST
    slot1_dist = float(obs[25 + 6]) * OBS_MAX_DIST
    dist_ok = math.isclose(slot0_dist, 3.0, abs_tol=1e-3) and math.isclose(slot1_dist, 6.0, abs_tol=1e-3)

    print(f"  env.locked['drone1'] = {env.locked['drone1']} (expected ['drone2','drone3']) [{'ok' if order_ok else 'FAIL'}]")
    print(f"  slot0 dist={slot0_dist:.3f}m (expected 3.0) slot1 dist={slot1_dist:.3f}m (expected 6.0) [{'ok' if dist_ok else 'FAIL'}]")

    ok = order_ok and dist_ok
    print(f"Scenario E: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="optional path to write a JSON pass/fail summary")
    args = parser.parse_args()

    _sanity_check_constants()

    results = {
        "A_converged_formation": scenario_a(),
        "A2_zero_init_reachability": scenario_a2(),
        "B_asymmetric_state": scenario_b(),
        "C_ned_spawn_offset": scenario_c(),
        "D_clock_skew": scenario_d(),
        "E_neighbor_ordering": scenario_e(),
    }

    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: bool(v) for k, v in results.items()}, f, indent=2)
        print(f"\nWrote summary to {args.out}")

    # A2 and D are informational and always True; only these four gate the result.
    gating = [
        results["A_converged_formation"],
        results["B_asymmetric_state"],
        results["C_ned_spawn_offset"],
        results["E_neighbor_ordering"],
    ]
    if all(gating):
        print("\nPhase 7 (observation adapter) VERIFIED -- next per PHASE2_HANDOFF.md is Phase 17 (timing).")
        sys.exit(0)
    else:
        print("\nPhase 7 FOUND A REAL DISCREPANCY -- see FAIL lines above. Do not proceed to a live flight.")
        sys.exit(1)


if __name__ == "__main__":
    main()
