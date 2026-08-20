#!/bin/bash
# Launch one PX4 SITL instance for the multi-drone deployment.
# Usage: launch_px4_instance.sh <instance_num> <spawn_x> <spawn_y>
#
# Instance 0 starts the Gazebo world (as in the single-drone setup).
# Instances 1+ attach to the already-running world (PX4_GZ_STANDALONE=1) at
# a distinct spawn pose. No -w flag: tried redirecting each instance's
# working directory there for dataman/param isolation, but -w replaces
# where PX4 looks for etc/init.d-posix/rcS too, not just where it writes
# state, and instances 1/2 failed immediately ("Error opening startup file,
# does not exist"). Matches the exact pattern PX4's own multi-vehicle docs
# show (-i alone, no -w) -- PX4 auto-differentiates per-instance state
# within the shared rootfs by instance number.
#
# Uses the native /usr/bin/gz, not the broken ROS2-vendored one that shadows
# it on PATH by default -- see this session's PX4-launch notes.
set -e

INSTANCE="$1"
X="$2"
Y="$3"
if [ -z "$INSTANCE" ] || [ -z "$X" ] || [ -z "$Y" ]; then
  echo "Usage: $0 <instance_num> <spawn_x> <spawn_y>" >&2
  exit 1
fi

PX4_BIN=/home/rayen/PX4-Autopilot/build/px4_sitl_default/bin/px4
ROOTFS=/home/rayen/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_bridge

export PATH="/usr/bin:$PATH"
export PX4_SIM_MODEL=gz_x500
export PX4_SYS_AUTOSTART=4001

cd "$ROOTFS"

if [ "$INSTANCE" = "0" ]; then
  export GZ_IP=127.0.0.1
  # PX4_GZ_MODEL_POSE is honored by px4-rc.gzsim regardless of whether this
  # instance is creating the world or attaching standalone -- confirmed in
  # source. Must be set here too so drone1 spawns at SPAWN_XY["drone1"]
  # (deployment/inference_node.py), not Gazebo's default (0,0): a mismatch
  # here silently corrupts every position calculation for this agent by the
  # missing offset, since SPAWN_OFFSET is added back assuming the spawn
  # actually happened at the configured (x, y).
  export PX4_GZ_MODEL_POSE="${X},${Y}"
  exec "$PX4_BIN" -i 0 -d < /dev/null
else
  export PX4_GZ_STANDALONE=1
  export PX4_GZ_MODEL_POSE="${X},${Y}"
  exec "$PX4_BIN" -i "$INSTANCE" -d < /dev/null
fi
