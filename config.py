"""Shared constants for the MARL formation/collision-avoidance env."""

NUM_AGENTS       = 4
K_NEIGHBORS       = 2

TARGET_DIST      = 3.5
SAFE_DIST_ENTER  = 4.20
SAFE_DIST_EXIT   = 5.20
COLLISION_DIST   = 4.20
DIVERGE_DIST     = 10.0     # used for re-lock trigger only now

MAX_STEPS        = 200
DT                = 0.05

MAX_ACTION_SPEED = 1.2

OBS_MAX_DIST      = 15.0
OBS_MAX_VEL        = 2.0
OBS_MAX_ANGLE      = 3.14159

# Reward weights -- rebalanced so no single term numerically dominates
TRACK_WEIGHT      = -2.0    # was -0.5, too weak vs safety magnitude
SAFETY_MAX_BONUS  = 2.0     # was 5.0, reduce dominance
SAFETY_URGENT_COEF = -30.0  # was -50.0
COHESION_LIMIT     = 13.5   # global swarm-diameter soft cap (~3x TARGET_DIST)
COHESION_WEIGHT     = -0.05
