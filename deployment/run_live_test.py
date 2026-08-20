#!/usr/bin/env python3
"""Mechanical, repeatable live PX4/Gazebo/ROS2 test runner.

Launches --num-agents PX4 SITL instances + Gazebo, the shared DDS agent, and
one inference_node.py per agent against a given checkpoint; runs until
--duration elapses or a safety bound (altitude / distance-to-target) is
violated, lands and tears everything down, and leaves a
deployment/runs/run_YYYYMMDD_HHMMSS/ directory behind with per-agent CSVs
(one row per control cycle -- see inference_node.py's CSV_COLUMNS), each
process's raw log, config.json, and summary.json. Point deployment/plot_run.py
at that directory afterward instead of watching scrollback live.

Requires the ROS2 environment sourced in THIS shell before running (this
script itself doesn't import rclpy, but the inference_node.py children it
launches do):
    source /opt/ros/jazzy/setup.bash
    source /home/rayen/ws_sensor_combined/install/setup.bash
    python3 deployment/run_live_test.py --num-agents 2 \
        --model models/actor_best_n2_1_20260820-1255_3f9069c-n2-5m-phase2-seed1.pt \
        --duration 30
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PX4_BIN = "/home/rayen/PX4-Autopilot/build/px4_sitl_default/bin"
LAUNCH_SCRIPT = os.path.join(HERE, "launch_px4_instance.sh")
ROS_SOURCE = "source /opt/ros/jazzy/setup.bash && source /home/rayen/ws_sensor_combined/install/setup.bash"

SPAWN_RADIUS = 5.0   # must match inference_node.py's SPAWN_RADIUS


def spawn_xy(num_agents):
    return [
        (SPAWN_RADIUS * math.cos(2 * math.pi * i / num_agents),
         SPAWN_RADIUS * math.sin(2 * math.pi * i / num_agents))
        for i in range(num_agents)
    ]


def sh(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, **kwargs)


def bg(cmd):
    # subprocess.Popen(shell=True) defaults to /bin/sh (dash on this system),
    # which has no `source` builtin -- `source: not found` silently short-
    # circuits any `&&`-chained command after it (confirmed the hard way:
    # this killed the DDS agent and both inference_node.py launches on the
    # first real run, while the watchdog below still reported a clean
    # "no violation" because it never saw any CSV data at all). Every launch
    # in this script needs ROS2 sourced, so always use bash explicitly.
    return subprocess.Popen(cmd, shell=True, executable="/bin/bash")


def cleanup():
    for pattern in ["bin/px4 -i", "gz sim", "gzserver", "MicroXRCEAgent", "inference_node.py"]:
        sh(f"pkill -f '{pattern}'", stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(1)


def wait_for(log_path, needle, timeout=40):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(log_path) and needle in open(log_path).read():
            return True
        time.sleep(0.5)
    return False


def rootfs(instance):
    return f"/home/rayen/PX4-Autopilot/build/px4_sitl_default/rootfs/{instance}"


def commander(instance, *args):
    return sh(f"cd {rootfs(instance)} && {PX4_BIN}/px4-commander {' '.join(args)}",
              capture_output=True, text=True)


def preflight_ok(instance):
    r = commander(instance, "check")
    return "Preflight check: OK" in (r.stdout + r.stderr)


def last_row(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def watch(csv_paths, duration, max_alt, max_dist, startup_grace=25.0):
    # Agents only start writing CSV rows once they've armed, climbed through
    # the explicit takeoff phase, and handed off to the policy -- that alone
    # can take 10-15s, so "no rows yet" isn't itself a failure until
    # startup_grace has passed. But it also must NOT be silently read as
    # "no violation" forever -- confirmed the hard way (see bg()'s comment):
    # a completely-failed-to-launch run reported a clean "no violation"
    # because zero data ever arrived to check bounds against.
    start = time.time()
    seen_data = {a: False for a in csv_paths}
    while time.time() - start < duration:
        time.sleep(2)
        elapsed = time.time() - start
        for agent, path in csv_paths.items():
            row = last_row(path)
            if row is None:
                continue
            seen_data[agent] = True
            alt = float(row["pos_z"])
            dist = float(row["dist_to_target"])
            if alt > max_alt:
                return f"SAFETY STOP: {agent} altitude {alt:.1f}m > {max_alt}m at t={elapsed:.0f}s"
            if dist > max_dist:
                return f"SAFETY STOP: {agent} dist_to_target {dist:.1f}m > {max_dist}m at t={elapsed:.0f}s"
        if elapsed > startup_grace and not all(seen_data.values()):
            never_seen = [a for a, ok in seen_data.items() if not ok]
            return f"FAILURE: no CSV data ever appeared for {never_seen} after {elapsed:.0f}s -- check their *_node.log"
    if not all(seen_data.values()):
        never_seen = [a for a, ok in seen_data.items() if not ok]
        return f"FAILURE: no CSV data ever appeared for {never_seen} during the whole {duration:.0f}s run -- check their *_node.log"
    return f"completed full {duration:.0f}s duration with no safety-bound violation"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-agents", type=int, required=True, choices=[2, 3, 4])
    ap.add_argument("--model", required=True, help="checkpoint path, used for every agent")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to watch after launch, before landing")
    ap.add_argument("--max-altitude", type=float, default=12.0, help="safety bound (m, sim z-up) -- early-lands everyone if exceeded")
    ap.add_argument("--max-dist-to-target", type=float, default=20.0, help="safety bound (m) -- early-lands everyone if exceeded")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = args.out_dir or os.path.join(HERE, "runs", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    agents = [f"drone{i + 1}" for i in range(args.num_agents)]
    positions = spawn_xy(args.num_agents)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "num_agents": args.num_agents, "model": args.model, "duration": args.duration,
            "max_altitude": args.max_altitude, "max_dist_to_target": args.max_dist_to_target,
            "spawn_xy": dict(zip(agents, positions)), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print("Cleaning up any stale PX4/Gazebo/DDS/inference processes...")
    cleanup()

    outcome = "did not reach the watch phase"
    try:
        print(f"Launching {args.num_agents} PX4 instances...")
        for i, (x, y) in enumerate(positions):
            log = os.path.join(run_dir, f"px4_instance{i}.log")
            bg(f"bash {LAUNCH_SCRIPT} {i} {x} {y} > {log} 2>&1")
            if not wait_for(log, "Ready for takeoff!"):
                raise RuntimeError(f"PX4 instance {i} never became ready -- see {log}")
            print(f"  instance {i} ready at ({x:.2f}, {y:.2f})")

        print("Starting MicroXRCEAgent...")
        dds_log = os.path.join(run_dir, "dds_agent.log")
        bg(f"{ROS_SOURCE} && MicroXRCEAgent udp4 -p 8888 > {dds_log} 2>&1")
        time.sleep(2)
        if not os.path.exists(dds_log) or os.path.getsize(dds_log) == 0:
            raise RuntimeError(f"MicroXRCEAgent produced no output at all -- it likely failed to start, see {dds_log}")

        for i in range(args.num_agents):
            if not preflight_ok(i):
                raise RuntimeError(f"Instance {i} failed preflight check -- see {rootfs(i)} for its own log/params")
        print("Preflight OK on all instances")

        csv_paths = {}
        print(f"Launching {args.num_agents} inference_node.py processes against {args.model}...")
        for a in agents:
            node_log = os.path.join(run_dir, f"{a}_node.log")
            csv_path = os.path.join(run_dir, f"{a}.csv")
            csv_paths[a] = csv_path
            cmd = (f"{ROS_SOURCE} && cd {REPO_ROOT} && NUM_AGENTS={args.num_agents} python3 "
                   f"deployment/inference_node.py --agent-name {a} --model {args.model} "
                   f"--log-csv {csv_path} > {node_log} 2>&1")
            bg(cmd)

        time.sleep(3)
        for a in agents:
            node_log = os.path.join(run_dir, f"{a}_node.log")
            if not os.path.exists(node_log) or os.path.getsize(node_log) == 0:
                raise RuntimeError(f"{a}'s inference_node.py produced no output at all -- it likely failed to start, see {node_log}")
        print("All inference_node.py processes producing output")

        print(f"Running for up to {args.duration:.0f}s (safety bounds: altitude<={args.max_altitude}m, "
              f"dist_to_target<={args.max_dist_to_target}m)...")
        outcome = watch(csv_paths, args.duration, args.max_altitude, args.max_dist_to_target)
        print(outcome)

        print("Landing all instances...")
        for i in range(args.num_agents):
            commander(i, "land")
        time.sleep(6)

    finally:
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump({"outcome": outcome, "ended": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        print("Tearing down...")
        cleanup()

    print(f"Done. Results in {run_dir}")
    print(f"Plot with: python3 deployment/plot_run.py {run_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
