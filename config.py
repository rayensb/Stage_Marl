"""Shared constants for the MARL formation/collision-avoidance env."""

NUM_AGENTS       = 4       # default swarm size (N-generic, test with 4)
K_NEIGHBORS       = 2       # each drone locks onto its 2 nearest neighbors

TARGET_DIST      = 3.5     # desired drone-to-target distance (m), now 3D
SAFE_DIST_ENTER  = 4.20
SAFE_DIST_EXIT   = 5.20
COLLISION_DIST   = 4.20
DIVERGE_DIST     = 10.0    # also used as re-lock trigger distance

MAX_STEPS        = 200
DT                = 0.05

MAX_ACTION_SPEED = 1.2     # m/s, clamp on vx/vy/vz output by policy

OBS_MAX_DIST      = 15.0
OBS_MAX_VEL        = 2.0
OBS_MAX_ANGLE      = 3.14159
