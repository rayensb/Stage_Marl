#!/usr/bin/env python3
"""MARL inference node -- bridges a trained Stage_Marl actor to a real
(PX4 SITL) vehicle over ROS2. Run one instance per real drone (see
deployment/launch_px4_instance.sh) -- each instance controls exactly one
agent and reads the other agents' REAL telemetry for its neighbor
observation. No phantom/scripted agents as of this version.

Reuses envs.formation_env.FormationEnv3D directly for observation
construction (_get_obs) and the closing-speed brake (_apply_brake), instead
of reimplementing that math here -- this is what makes the observation
provably exact to what the policy trained against, rather than an
approximation of it. The two things a real deployment must never get subtly
wrong: the vision-tracking observation, and the brake, which is not part of
the network and does nothing on its own if this node doesn't apply it.

SCOPE (2026-08-19, multi-instance version): all NUM_AGENTS drones are real,
each running its own copy of this process. Only the TARGET remains a
scripted, visible Gazebo marker -- deliberately: the vision-tracking system
already models "sometimes visible, sensor-range-limited" detection
(_update_target_track), so a scripted target exercises that logic honestly
without needing a fourth real vehicle to play the target's role.

Multi-instance coordination without shared Python state: the 3 node
processes are independent OS processes (no shared memory), so they can't
just share a `target_pos` variable. Two things make this work without a
side-channel between them: (1) a fixed, pre-agreed starting scene
(TARGET_START, SPAWN_XY below) that every instance can compute identically
without coordinating; (2) all 3 have the same TARGET_START/PHANTOM_DIR/
PHANTOM_SPEED constants, so each independently computes the SAME
deterministic target trajectory as a function of elapsed time since its own
takeoff completion. Only the agent that owns the marker (drone1) actually
drives it in Gazebo; the others compute the same trajectory internally for
their own observation. Small clock-skew error between instances' takeoff
completion times (each is a fixed-duration climb, so this should stay under
a second or two in practice) is an accepted first-version simplification,
not corrected for.

Coordinate convention: PX4 telemetry/setpoints are NED (Z down positive).
The sim/training convention is Z-up positive, X/Y otherwise arbitrary (the
sim has no compass heading reference, so there's no "true" ENU/NED axis
alignment to preserve -- this picks the simplest self-consistent mapping).
Conversion is a single Z-axis sign flip, X/Y pass through unchanged; see
ned_to_sim/sim_to_ned below. Applied identically to position and velocity.

ROS2 topic namespacing (confirmed against PX4 docs, not guessed): PX4
instance 0 is unnamespaced (/fmu/...), instance N>0 uses /px4_N/fmu/... --
see AGENT_NAMESPACE below for the fixed agent-name-to-namespace mapping
this deployment uses.

Prerequisites (see deployment/launch_px4_instance.sh):
    3 PX4 SITL instances (one Gazebo world, instances 1/2 attached
    standalone), each with NAV_DLL_ACT=0 set in its own working directory's
    parameter store, and one shared Micro-XRCE-DDS-Agent (multiple SITL
    clients can share a single agent -- confirmed via PX4 docs).

Run (one invocation per drone, in separate terminals/processes):
    source /opt/ros/jazzy/setup.bash && source ~/ws_sensor_combined/install/setup.bash
    NUM_AGENTS=3 python3 deployment/inference_node.py --agent-name drone1 \
        --model models/actor_best_n3_3_20260817-1255_b59c139-3m-seed3.pt
    NUM_AGENTS=3 python3 deployment/inference_node.py --agent-name drone2 --model ...
    NUM_AGENTS=3 python3 deployment/inference_node.py --agent-name drone3 --model ...
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import subprocess
import time

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus,
)

from envs.formation_env import FormationEnv3D
from training.networks import Actor
from config import NUM_AGENTS, K_NEIGHBORS, OBS_DIM, ACT_DIM, MAX_ACTION_SPEED, TARGET_DIST, DT

assert NUM_AGENTS in (2, 3, 4), (
    "The namespace map and spawn geometry below are generalized from "
    "NUM_AGENTS but only verified/checkpointed for 2, 3, or 4 agents -- "
    "rerun with one of those in the environment."
)

# PX4 instance-per-agent map -- instance 0 unnamespaced, instance N>0 uses
# /px4_N/fmu/... (confirmed via PX4 ROS2 multi-vehicle docs). Must match
# the -i flag each deployment/launch_px4_instance.sh invocation used.
# Generalized from NUM_AGENTS (2026-08-20) -- verified to reproduce the
# original hardcoded N=3 map exactly (drone1="", drone2="/px4_1",
# drone3="/px4_2"), so the proven N=3 path is unchanged.
ALL_AGENTS = [f"drone{i + 1}" for i in range(NUM_AGENTS)]
AGENT_NAMESPACE = {a: ("" if i == 0 else f"/px4_{i}") for i, a in enumerate(ALL_AGENTS)}

# Fixed, pre-agreed starting scene -- see module docstring for why this is
# computed once here rather than at runtime. Spawn points sit on a circle of
# radius SPAWN_RADIUS around the target's start, evenly spaced, so the
# starting scene is already close to the trained-for equilateral formation.
# deployment/launch_px4_instance.sh's spawn args must match SPAWN_XY.
# Generalized from NUM_AGENTS (2026-08-20) -- for NUM_AGENTS=3 this
# reproduces the original hardcoded (5.0,0.0)/(-2.5,4.33)/(-2.5,-4.33) up to
# float rounding (drone2/3 were hand-rounded to 2dp originally).
TARGET_START = np.array([0.0, 0.0, 3.0], np.float32)   # sim/Gazebo world frame
SPAWN_RADIUS = 5.0
SPAWN_XY = {
    a: (SPAWN_RADIUS * np.cos(2 * np.pi * i / NUM_AGENTS),
        SPAWN_RADIUS * np.sin(2 * np.pi * i / NUM_AGENTS))
    for i, a in enumerate(ALL_AGENTS)
}
# Each PX4 instance's vehicle_local_position is relative to ITS OWN local
# EKF origin (near its own spawn point), not a frame shared across
# instances -- confirmed empirically (two drones spawned ~8.7m apart in
# Gazebo both reported positions near (0,0) right after their own takeoff).
# Adding each agent's known spawn offset back reconstructs a shared frame,
# since PX4_GZ_MODEL_POSE below is exactly that offset. Z is not offset --
# all three spawn at ground level (z=0).
SPAWN_OFFSET = {a: np.array([x, y, 0.0], np.float32) for a, (x, y) in SPAWN_XY.items()}
PHANTOM_SPEED = 0.3                         # m/s, low end of training's 0.3-1.0 target-speed range
PHANTOM_DIR = np.array([0.0, 1.0, 0.0], np.float32)   # target drifts in +Y (sim frame)

# Explicit takeoff before handing off to the trained policy -- PX4 auto-
# disarms an armed vehicle that never leaves the ground within
# COM_DISARM_PRFLT seconds (Commander.cpp handleAutoDisarm), and the trained
# policy has no reason to command a climb from a ground-level start --
# tracking a target is its whole job, not taking off.
TAKEOFF_ALTITUDE = 3.0        # m, sim frame (z-up) -- within training's target-altitude range (2-3)
TAKEOFF_CLIMB_SPEED = 0.5     # m/s

QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def ned_to_sim(v):
    """[N, E, D] -> [x, y, z], z-up. Single sign flip -- see module docstring."""
    return np.array([v[0], v[1], -v[2]], dtype=np.float32)


def sim_to_ned(v):
    """[x, y, z] (z-up) -> [N, E, D]. Inverse of ned_to_sim."""
    return [float(v[0]), float(v[1]), float(-v[2])]


# Visual marker for the scripted target (2026-08-19) -- spawned as a simple,
# physics-light sphere via Gazebo's own /world/<world>/create service (the
# identical pattern PX4's own px4-rc.gzsim init script uses to spawn vehicle
# models), driven by the gz-sim-velocity-control-system plugin so a single
# one-time velocity command keeps it drifting on its own -- no per-cycle
# republishing needed. Gazebo's world frame is natively z-up, matching this
# project's sim-frame convention directly, so no coordinate conversion is
# needed here (unlike the NED conversion PX4 setpoints need). Uses the
# native /usr/bin/gz, not the broken ROS2-vendored one that shadows it on
# PATH by default -- see this session's PX4-launch notes.
GZ_WORLD = "default"
GZ_ENV = {**os.environ, "PATH": "/usr/bin:" + os.environ.get("PATH", "")}


def _marker_sdf(name, rgb):
    r, g, b = rgb
    return (
        f'<sdf version="1.6"><model name="{name}"><pose>0 0 0 0 0 0</pose>'
        f'<static>false</static><link name="link"><gravity>false</gravity>'
        f'<inertial><mass>0.1</mass><inertia><ixx>0.001</ixx><iyy>0.001</iyy>'
        f'<izz>0.001</izz></inertia></inertial>'
        f'<visual name="visual"><geometry><sphere><radius>0.3</radius></sphere></geometry>'
        f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>'
        f'</visual></link>'
        f'<plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">'
        f'<topic>/model/{name}/cmd_vel</topic></plugin></model></sdf>'
    )


def spawn_marker(name, rgb):
    req = f'name: "{name}", allow_renaming: false, sdf: \'{_marker_sdf(name, rgb)}\''
    subprocess.run(["gz", "service", "-s", f"/world/{GZ_WORLD}/create",
                     "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "5000", "--req", req],
                    env=GZ_ENV, capture_output=True)


def set_marker_pose(name, pos):
    req = f'name: "{name}", position: {{x: {pos[0]}, y: {pos[1]}, z: {pos[2]}}}'
    subprocess.run(["gz", "service", "-s", f"/world/{GZ_WORLD}/set_pose",
                     "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                     "--timeout", "2000", "--req", req],
                    env=GZ_ENV, capture_output=True)


def set_marker_velocity(name, vel):
    req = f'linear: {{x: {vel[0]}, y: {vel[1]}, z: {vel[2]}}}'
    subprocess.run(["gz", "topic", "-t", f"/model/{name}/cmd_vel",
                     "-m", "gz.msgs.Twist", "-p", req],
                    env=GZ_ENV, capture_output=True)


CSV_COLUMNS = [
    "wall_time", "cycle", "agent", "armed", "took_off",
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
    "target_x", "target_y", "target_z", "dist_to_target", "min_dist_to_neighbor",
    "action_x", "action_y", "action_z",
    "safe_vel_x", "safe_vel_y", "safe_vel_z", "brake_reduction",
]


class MarlInferenceNode(Node):
    def __init__(self, model_path, agent_name, device="cpu", log_csv=None):
        super().__init__(f"marl_inference_node_{agent_name}")
        self.device = device
        self.agent_name = agent_name
        self.csv_writer = None
        self.csv_file = None
        if log_csv:
            self.csv_file = open(log_csv, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(CSV_COLUMNS)
            self.get_logger().info(f"[{agent_name}] Logging every cycle to {log_csv}")
        self.other_agents = [a for a in ALL_AGENTS if a != agent_name]
        self.is_target_owner = (agent_name == "drone1")
        self.ns = AGENT_NAMESPACE[agent_name]

        self.actor = Actor(OBS_DIM, ACT_DIM).to(device)
        self.actor.load_state_dict(torch.load(model_path, map_location=device))
        self.actor.eval()
        self.get_logger().info(
            f"[{agent_name}] Loaded actor from {model_path} (OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}, ns='{self.ns}')"
        )

        if self.is_target_owner:
            spawn_marker("target_marker", (1.0, 0.8, 0.0))
            self.get_logger().info(f"[{agent_name}] Spawned target marker in Gazebo (this instance owns it)")

        # Reused directly for _get_obs/_apply_brake -- see module docstring.
        # Not reset()/seeded: this node drives its internal pos/vel/pos_t
        # state by hand every cycle from real telemetry (self + both real
        # neighbors) plus the scripted target, which reset()'s random spawn
        # doesn't support.
        self.env = FormationEnv3D(num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS)
        self.env.agents = self.env.possible_agents[:]
        self.env.locked = {a: [o for o in ALL_AGENTS if o != a] for a in ALL_AGENTS}
        self.env.pos = {a: np.zeros(3, np.float32) for a in ALL_AGENTS}
        self.env.vel = {a: np.zeros(3, np.float32) for a in ALL_AGENTS}

        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, f"{self.ns}/fmu/in/offboard_control_mode", QOS)
        self.trajectory_setpoint_pub = self.create_publisher(TrajectorySetpoint, f"{self.ns}/fmu/in/trajectory_setpoint", QOS)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, f"{self.ns}/fmu/in/vehicle_command", QOS)
        # Versioned topic names (_v1/_v4) -- this PX4/px4_msgs version
        # publishes these, not the bare names some older example code uses.
        # Confirmed against live `ros2 topic list` output earlier this session.
        self.create_subscription(VehicleStatus, f"{self.ns}/fmu/out/vehicle_status_v4", self._on_status, QOS)

        self.local_pos = {a: None for a in ALL_AGENTS}   # latest VehicleLocalPosition per agent, or None until first message
        for a in ALL_AGENTS:
            topic = f"{AGENT_NAMESPACE[a]}/fmu/out/vehicle_local_position_v1"
            self.create_subscription(
                VehicleLocalPosition, topic,
                (lambda msg, agent=a: self._on_local_position(agent, msg)), QOS,
            )

        self.status = None         # latest VehicleStatus for self, or None until first message
        self.took_off = False
        self.takeoff_clock_start = None
        self.setpoint_counter = 0
        self.cycle = 0
        self.real_branch_count = 0
        self.fallback_count = 0

        self.create_timer(DT, self._timer_callback)
        self.get_logger().info(f"[{agent_name}] Control loop at {1.0/DT:.0f} Hz (DT={DT}s, matches training)")

    def _on_local_position(self, agent, msg):
        self.local_pos[agent] = msg

    def _on_status(self, msg):
        self.status = msg

    def _current_target_pos(self):
        elapsed = (self.get_clock().now() - self.takeoff_clock_start).nanoseconds / 1e9
        return (TARGET_START + PHANTOM_DIR * PHANTOM_SPEED * elapsed).astype(np.float32)

    def _start_target_motion(self):
        """Called once, the cycle this instance completes takeoff. See
        module docstring for why this is a fixed, independently-computable
        trajectory rather than a runtime handshake between the 3 processes."""
        self.takeoff_clock_start = self.get_clock().now()
        if self.is_target_owner:
            set_marker_pose("target_marker", TARGET_START)
            set_marker_velocity("target_marker", PHANTOM_DIR * PHANTOM_SPEED)
        self.get_logger().info(f"[{self.agent_name}] Takeoff complete -- handing off to the trained policy")

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)

    def _publish_velocity_setpoint(self, ned_vel):
        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        msg.velocity = [float(v) for v in ned_vel]
        msg.yaw = float("nan")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

    def _publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        for i in range(1, 8):
            setattr(msg, f"param{i}", params.get(f"param{i}", 0.0))
        msg.target_system = 0   # broadcast within this instance's own namespaced topic
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def _arm(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(f"[{self.agent_name}] Arm command sent")

    def _engage_offboard(self):
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info(f"[{self.agent_name}] Switching to offboard mode")

    def _timer_callback(self):
        self._publish_offboard_heartbeat()   # required every cycle regardless of what follows
        self.cycle += 1

        in_offboard = self.status is not None and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        armed = self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

        if self.setpoint_counter < 11:
            self.setpoint_counter += 1
        elif not (in_offboard and armed) and self.cycle % int(2.0 / DT) == 0:
            # A single attempt can be rejected by a transient PX4 preflight
            # check (e.g. EKF innovations still settling right after GPS
            # fusion activates) -- retry instead of waiting forever on one shot.
            self._engage_offboard()
            self._arm()

        own_msg = self.local_pos[self.agent_name]
        if own_msg is None:
            self.fallback_count += 1
            self._publish_velocity_setpoint([0.0, 0.0, 0.0])
            return

        own_sim_pos = ned_to_sim([own_msg.x, own_msg.y, own_msg.z]) + SPAWN_OFFSET[self.agent_name]
        own_sim_vel = ned_to_sim([own_msg.vx, own_msg.vy, own_msg.vz])   # velocity is offset-invariant

        if not in_offboard or not armed:
            self.fallback_count += 1
            self._publish_velocity_setpoint([0.0, 0.0, 0.0])
            if self.cycle % int(2.0 / DT) == 0:
                self.get_logger().info(
                    f"[{self.agent_name}] waiting to arm: nav_state={self.status.nav_state if self.status else None} "
                    f"armed={armed}"
                )
            return

        if not self.took_off:
            if own_sim_pos[2] < TAKEOFF_ALTITUDE:
                self._publish_velocity_setpoint(sim_to_ned(np.array([0.0, 0.0, TAKEOFF_CLIMB_SPEED], np.float32)))
                if self.cycle % int(1.0 / DT) == 0:
                    self.get_logger().info(f"[{self.agent_name}] Taking off: alt={own_sim_pos[2]:.2f}/{TAKEOFF_ALTITUDE:.2f}")
                return
            self.took_off = True
            self._start_target_motion()

        # Need real telemetry from both neighbors before running the policy
        # -- hold position rather than guessing if they haven't published yet.
        neighbor_msgs = {a: self.local_pos[a] for a in self.other_agents}
        if any(m is None for m in neighbor_msgs.values()):
            self._publish_velocity_setpoint([0.0, 0.0, 0.0])
            if self.cycle % int(1.0 / DT) == 0:
                missing = [a for a, m in neighbor_msgs.items() if m is None]
                self.get_logger().info(f"[{self.agent_name}] waiting for neighbor telemetry: {missing}")
            return

        # Drive the reused env's state from real telemetry (self + both real
        # neighbors) plus the scripted target for this cycle.
        self.env.pos[self.agent_name] = own_sim_pos
        self.env.vel[self.agent_name] = own_sim_vel
        for a in self.other_agents:
            m = neighbor_msgs[a]
            self.env.pos[a] = ned_to_sim([m.x, m.y, m.z]) + SPAWN_OFFSET[a]
            self.env.vel[a] = ned_to_sim([m.vx, m.vy, m.vz])   # velocity is offset-invariant
        # Nearest-neighbor-first ordering, matching _relock_all's convention
        # (slot 0 is always the closer neighbor) -- with real, moving
        # neighbors this can change over time, unlike the single-drone
        # version's fixed phantom placement.
        self.env.locked[self.agent_name] = sorted(
            self.other_agents,
            key=lambda a: float(np.linalg.norm(self.env.pos[a] - self.env.pos[self.agent_name])),
        )
        target_pos = self._current_target_pos()
        self.env.pos_t = target_pos
        self.env._target_dir = PHANTOM_DIR
        self.env._target_speed = PHANTOM_SPEED
        self.env._update_target_track()

        obs = self.env._get_obs(self.agent_name)
        with torch.no_grad():
            o = torch.as_tensor(obs).unsqueeze(0).to(self.device)
            act, _ = self.actor.get_action(o, deterministic=True)
            action = act.squeeze(0).cpu().numpy()

        raw_vel = (np.clip(action, -1.0, 1.0) * MAX_ACTION_SPEED).astype(np.float32)
        corrected, brake_reduction = self.env._apply_brake({
            self.agent_name: raw_vel,
            **{a: np.zeros(3, np.float32) for a in self.other_agents},
        })
        safe_vel_sim = corrected[self.agent_name]
        ned_vel = sim_to_ned(safe_vel_sim)
        self._publish_velocity_setpoint(ned_vel)
        self.real_branch_count += 1

        dist_t = float(np.linalg.norm(own_sim_pos - target_pos))
        min_pair = min(
            float(np.linalg.norm(self.env.pos[a] - own_sim_pos)) for a in self.other_agents
        )

        if self.csv_writer is not None:
            self.csv_writer.writerow([
                f"{time.time():.3f}", self.cycle, self.agent_name, True, self.took_off,
                *own_sim_pos.tolist(), *own_sim_vel.tolist(),
                *target_pos.tolist(), f"{dist_t:.4f}", f"{min_pair:.4f}",
                *action.tolist(), *safe_vel_sim.tolist(),
                f"{brake_reduction[self.agent_name]:.4f}",
            ])

        if self.cycle % int(1.0 / DT) == 0:   # ~once/sec
            self.get_logger().info(
                f"[{self.agent_name}] pos={own_sim_pos.round(2)} dist_to_target={dist_t:.2f} "
                f"(ideal={TARGET_DIST:.2f}) min_dist_to_neighbor={min_pair:.2f} "
                f"safe_vel_sim={safe_vel_sim.round(2)} brake={brake_reduction[self.agent_name]:.3f} "
                f"real/fallback_cycles={self.real_branch_count}/{self.fallback_count}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to an actor .pt checkpoint (NUM_AGENTS-shaped)")
    parser.add_argument("--agent-name", required=True, choices=ALL_AGENTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-csv", default=None, help="Optional path to write one CSV row per control cycle (see CSV_COLUMNS)")
    args = parser.parse_args()

    rclpy.init()
    node = MarlInferenceNode(args.model, args.agent_name, args.device, log_csv=args.log_csv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
