"""Shared constants for the MARL formation/collision-avoidance env."""

NUM_AGENTS       = 4
K_NEIGHBORS       = 2

SAFE_DIST_ENTER  = 4.60    # was 4.20 == COLLISION_DIST -- left no warning zone before termination
SAFE_DIST_EXIT   = 5.60
COLLISION_DIST   = 4.20
DIVERGE_DIST     = 10.0     # used for re-lock trigger only now

# TARGET_DIST derived, not hand-picked: the best possible arrangement of
# NUM_AGENTS points all at distance TARGET_DIST from a shared target is the
# packing that maximizes their minimum pairwise separation. For N=4 that's a
# regular tetrahedron, edge = TARGET_DIST * sqrt(8/3). At the old flat value
# (3.5) that edge was 5.72 -- just 0.12 above SAFE_DIST_EXIT, i.e. even the
# mathematically best-case formation had ~2% safety margin. Any exploration
# noise or imperfect coordination pushed pairs below the safe zone, which is
# why collision_rate kept relapsing every time the policy got *better* at
# tracking (converging harder onto that razor-thin-margin sphere) no matter
# how SAFETY_URGENT_COEF/COHESION_LIMIT were rebalanced -- reweighting a
# reward can't fix a target geometry that has no safe solution to converge to.
# (No simple closed-form packing ratio for N != 4 -- revisit this formula if
# NUM_AGENTS changes.)
_TETRA_RATIO   = (8 / 3) ** 0.5     # regular-tetrahedron edge/radius ratio, ~1.633
_SAFETY_MARGIN = 1.6                 # target tetrahedron edge 60% above SAFE_DIST_EXIT
TARGET_DIST    = _SAFETY_MARGIN * SAFE_DIST_EXIT / _TETRA_RATIO   # ~5.49

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

# Kept proportional to TARGET_DIST (same ratio as the old 13.5/3.5 pairing)
# rather than left as a fixed absolute -- otherwise raising TARGET_DIST alone
# would make this soft cap relatively *tighter* against the new, naturally
# larger formation, recreating the same cohesion-vs-safety conflict this
# TARGET_DIST change exists to remove.
COHESION_RATIO      = 3.857
COHESION_LIMIT       = COHESION_RATIO * TARGET_DIST   # ~21.2
COHESION_WEIGHT     = -0.05
