"""Shared constants for the MARL formation/collision-avoidance env."""

NUM_AGENTS       = 4
K_NEIGHBORS       = 2

# Single source of truth for network I/O shapes -- train.py and evaluate.py
# both need these to match exactly (evaluate.py loads weights saved by
# train.py), so this is not duplicated in either file.
OBS_DIM = 10 + 7 * K_NEIGHBORS
ACT_DIM = 3

COLLISION_DIST   = 4.20     # hard physical clearance radius (rotor/frame safety)
DIVERGE_DIST     = 10.0     # used for re-lock trigger only now

MAX_STEPS        = 200
DT                = 0.05    # 20 Hz control loop

MAX_ACTION_SPEED = 1.2      # actions set velocity directly -- no acceleration/
                              # inertia limit in this sim, so there's no classical
                              # braking-distance (v^2/2a) term; adding one later
                              # (real dynamics, sim-to-real work) would only make
                              # REACTION_DIST below larger, which is safe.

# Every safety-zone distance below is derived from a single physical quantity:
# how far two drones can close on each other, worst case, before the policy
# can be expected to react. Worst case = head-on, both at MAX_ACTION_SPEED.
# N_REACT is the one real judgment call here (comparable to picking a
# perception-reaction-time constant in vehicle stopping-sight-distance
# standards) -- it covers policy imperfection/action noise (residual std
# floor never reaches 0, entropy never fully collapses), not hardware
# latency. 10 cycles (0.5s) is a moderate budget: enough for a stochastic,
# still-imperfect controller to notice a closing trajectory and commit to a
# correction, without forcing drones absurdly far apart.
N_REACT       = 10
REACTION_DIST = 2 * MAX_ACTION_SPEED * (N_REACT * DT)   # = 1.20

# SAFE_DIST_ENTER: even in the worst-case closing scenario, one full
# reaction window still leaves COLLISION_DIST of clearance -- this is the
# point past which "urgent" backoff pressure must apply.
SAFE_DIST_ENTER = COLLISION_DIST + REACTION_DIST                 # = 5.40
# SAFE_DIST_EXIT: a second reaction-distance of comfortable buffer above
# the warning threshold, in the same physical unit rather than an arbitrary
# band width.
SAFE_DIST_EXIT  = SAFE_DIST_ENTER + REACTION_DIST                 # = 6.60

# TARGET_DIST derived, not hand-picked: the best possible arrangement of
# NUM_AGENTS points all at distance TARGET_DIST from a shared target is the
# packing that maximizes their minimum pairwise separation. For N=4 that's a
# regular tetrahedron, edge = TARGET_DIST * sqrt(8/3). At the original flat
# value (3.5) that edge was 5.72 -- against the *original* SAFE_DIST_EXIT
# (5.60) that was just 0.12 of margin, i.e. even the mathematically best-case
# formation had ~2% safety margin. That's why collision_rate kept relapsing
# every time the policy got *better* at tracking across every prior fix:
# converging harder onto a razor-thin-margin sphere is the correct optimum
# of an over-constrained task, not a policy bug. No amount of reweighting
# SAFETY vs COHESION vs TRACK fixes a target geometry with no safe solution
# to converge to. The optimal formation should clear the whole comfort zone
# with a further full reaction-distance to spare, for imperfect real-world
# coordination (a decentralized policy won't hit a perfect tetrahedron).
# (No simple closed-form packing ratio for N != 4 -- revisit this formula if
# NUM_AGENTS changes.)
_TETRA_RATIO      = (8 / 3) ** 0.5                          # tetrahedron edge/radius ratio, ~1.633
_TETRA_EDGE_TARGET = SAFE_DIST_EXIT + REACTION_DIST          # = 7.80
TARGET_DIST        = _TETRA_EDGE_TARGET / _TETRA_RATIO       # ~4.78

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

# Joint track+safety bonus. Additive reward composition (the terms above)
# lets the policy bank "good enough" total reward from tracking OR safety
# alone -- confirmed in eval.py results (100 deterministic episodes on the
# 600k-step model): episodes split into two modes, tight formation that
# tracks well but collides fast, or spread formation that's safe but tracks
# poorly, never both. This adds an explicit bonus only when both hold at
# once, so the region that's actually the goal has a reward advantage
# instead of just being hoped for by the sum. JOINT_TRACK_TOL is derived
# (within one REACTION_DIST of ideal TARGET_DIST -- an error a single
# reaction window could correct); JOINT_BONUS's magnitude is a starting
# guess sized to be comparable to SAFETY_MAX_BONUS, not a derived value --
# treat as the first thing to tune if this doesn't move the needle.
JOINT_TRACK_TOL = REACTION_DIST   # ~1.20
JOINT_BONUS     = 3.0
