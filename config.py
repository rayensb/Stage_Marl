"""Shared constants for the MARL formation/collision-avoidance env."""

import os

# NUM_AGENTS/N_REACT are env-var overridable (like SEED/DEVICE in train.py)
# so parallel Kaggle sessions can sweep them -- e.g. NUM_AGENTS=2 for a
# curriculum-learning starting point, or different N_REACT values to compare
# safety-margin assumptions -- without hand-editing this file each time.
NUM_AGENTS  = int(os.environ.get("NUM_AGENTS", 4))
# Was a flat 2 -- supervisor review (2026-08-14) pointed out r_safety in
# formation_env.py checks every other agent (collision termination is
# global, so it has to), but the observation only ever included the
# K_NEIGHBORS locked ones -- an agent could take the -300 collision penalty,
# or the ramping urgent-safety penalty, for a drone it never observed. At
# NUM_AGENTS<=3 this was accidentally a non-issue (there are <=2 "others",
# so k=2 already locked everyone -- verified by reading _relock_all: with
# only 2 candidates and k=2, mutual reciprocity is guaranteed). It only
# bites at NUM_AGENTS=4, where the old k=2 left exactly one other agent
# permanently unobserved. Locking everyone (K_NEIGHBORS = NUM_AGENTS-1)
# closes that gap for the agent counts this project actually supports
# ({2,3,4}) without a variable-neighbor encoder -- free at this scale, not
# something that scales past NUM_AGENTS=4 (that's a real architecture
# change, not this one). Also makes the mutual-kNN/relock-staleness concern
# moot here: "locked" now always equals "everyone," so there's nothing
# stale to relock.
K_NEIGHBORS = NUM_AGENTS - 1

# Single source of truth for network I/O shapes -- train.py and evaluate.py
# both need these to match exactly (evaluate.py loads weights saved by
# train.py), so this is not duplicated in either file. Must use the same
# num_agents-capped neighbor count the env actually produces (formation_env's
# self.k = min(K_NEIGHBORS, NUM_AGENTS-1)) -- using raw K_NEIGHBORS here was a
# latent bug: harmless at NUM_AGENTS=4 (min(2,3)=2, matches), but NUM_AGENTS=2
# gives the env k=1 while the network would still expect k=2 -- a shape
# mismatch that only surfaces once anyone actually tries fewer agents.
EFFECTIVE_K = min(K_NEIGHBORS, NUM_AGENTS - 1)
# Own-state block grew 10 -> 18 with vision-based tracking (2026-08-14): own
# velocity (3), relative position/distance/velocity to the current best
# target estimate (3+1+3=7, same slots as before but now sourced from the
# track rather than ground truth), has_direct_contact (1), track confidence
# (1), track age normalized by LOST_TIMEOUT (1), observer_count normalized
# by NUM_AGENTS (1), relative direction to the centroid of currently
# in-contact teammates (3), centroid_valid (1) -- see _get_obs() in
# envs/formation_env.py for the exact layout.
OBS_OWN_DIM = 18
OBS_DIM = OBS_OWN_DIM + 7 * EFFECTIVE_K
ACT_DIM = 3

COLLISION_DIST   = 4.20     # hard physical clearance radius (rotor/frame safety)
DIVERGE_DIST     = 10.0     # used for re-lock trigger only now

# Phase 3 (2026-08-20): raised from a flat 200 (10s) after
# training/diagnose_horizon.py showed the policy's tracking error nearly
# quadrupling (0.95 -> 3.66) once run past its training horizon, in a time
# window that lines up with where the real PX4/Gazebo deployment actually
# crashed (see PHASE2_CHECKPOINT.md) -- the policy had simply never
# experienced sustained flight that long. Env-var overridable like
# NUM_AGENTS/TOTAL_STEPS so this can still be swept/reverted without
# hand-editing. 1800 = 90s at DT=0.05.
MAX_STEPS        = int(os.environ.get("MAX_STEPS", 1800))
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
N_REACT       = int(os.environ.get("N_REACT", 10))
REACTION_DIST = 2 * MAX_ACTION_SPEED * (N_REACT * DT)   # = 1.20 at N_REACT=10

# SAFE_DIST_ENTER: even in the worst-case closing scenario, one full
# reaction window still leaves COLLISION_DIST of clearance -- this is the
# point past which "urgent" backoff pressure must apply.
SAFE_DIST_ENTER = COLLISION_DIST + REACTION_DIST                 # = 5.40
# SAFE_DIST_EXIT: a second reaction-distance of comfortable buffer above
# the warning threshold, in the same physical unit rather than an arbitrary
# band width.
SAFE_DIST_EXIT  = SAFE_DIST_ENTER + REACTION_DIST                 # = 6.60

# Ground awareness (Phase 3, 2026-08-20): the abstract sim had no ground
# plane at all until now -- confirmed by search, not an oversight being
# fixed blind. Same derivation language as the inter-agent zones above,
# just against a fixed floor (z=0, matching the deployment pipeline's
# z-up convention where TAKEOFF_ALTITUDE/TARGET_START are already measured
# from ground level) instead of another agent's position. Mirrors why the
# inter-agent case eventually needed a deterministic clamp on top of reward
# shaping, not reward shaping alone -- see _apply_ground_clamp in
# envs/formation_env.py.
GROUND_Z          = 0.0
GROUND_SAFE_ENTER = GROUND_Z + REACTION_DIST                       # = 1.20
GROUND_SAFE_EXIT  = GROUND_Z + 2 * REACTION_DIST                   # = 2.40

# TARGET_DIST derived, not hand-picked: the best possible arrangement of
# NUM_AGENTS points all at distance TARGET_DIST from a shared target is the
# packing that maximizes their minimum pairwise separation (the Tammes
# problem). Closed-form optimal packings only exist for small N -- N=2 is
# antipodal points (edge = 2x radius), N=3 is an equilateral triangle on a
# great circle (edge = sqrt(3) x radius), N=4 is a regular tetrahedron (edge
# = sqrt(8/3) x radius). At the original flat TARGET_DIST (3.5, N=4) that
# edge was 5.72 -- against the *original* SAFE_DIST_EXIT (5.60) that was just
# 0.12 of margin, i.e. even the mathematically best-case formation had ~2%
# safety margin. That's why collision_rate kept relapsing every time the
# policy got *better* at tracking across every prior fix: converging harder
# onto a razor-thin-margin sphere is the correct optimum of an
# over-constrained task, not a policy bug. No amount of reweighting SAFETY
# vs COHESION vs TRACK fixes a target geometry with no safe solution to
# converge to. The optimal formation should clear the whole comfort zone
# with a further full reaction-distance to spare, for imperfect real-world
# coordination (a decentralized policy won't hit the exact optimal packing).
_PACKING_RATIO = {2: 2.0, 3: 3 ** 0.5, 4: (8 / 3) ** 0.5}
if NUM_AGENTS not in _PACKING_RATIO:
    raise ValueError(
        f"No known optimal spherical-packing ratio for NUM_AGENTS={NUM_AGENTS} "
        f"-- add it to _PACKING_RATIO in config.py (see the Tammes problem) "
        f"before running with this many agents."
    )
# No longer just an internal derivation step -- envs/formation_env.py now
# imports this directly too (the reshaped r_safety bonus ramps up to it),
# so it lost its leading underscore.
#
# N-aware margin (hypothesis, 2026-08-19, n-aware-margin branch): EDGE_TARGET
# was flat regardless of N, but the brake/collision problem confirmed at N=4
# (KNOWN_ISSUES.md item 8) scales with valence (simultaneous-neighbor count),
# not N directly -- K_NEIGHBORS=NUM_AGENTS-1 is full connectivity, so each
# agent has 2 simultaneous "others" at N=3 (the validated case) and 3 at N=4.
# Adds one extra REACTION_DIST of margin per neighbor beyond that validated
# count, so N=3 is exactly unchanged (max(0, 2-2)=0) and N=4 gets +1.20
# (max(0, 3-2)=1) -- targets the actual mechanism instead of an arbitrary
# "make N=4 bigger" fudge. Untested as of this branch; the check is whether
# this measurably helps the confirmed N=4 collision persistence without
# hurting tracking (a wider formation could push drones toward/past
# SENSOR_RANGE, the same failure mode the old diameter floor caused once
# vision-tracking existed -- see DECISIONS.md).
EDGE_TARGET  = SAFE_DIST_EXIT + REACTION_DIST + max(0, K_NEIGHBORS - 2) * REACTION_DIST
TARGET_DIST  = EDGE_TARGET / _PACKING_RATIO[NUM_AGENTS]          # ~4.78 at N=4 pre-this-change

OBS_MAX_DIST      = 15.0
OBS_MAX_VEL        = 2.0

# Reward weights -- rebalanced so no single term numerically dominates
TRACK_WEIGHT      = -2.0    # was -0.5, too weak vs safety magnitude
SAFETY_MAX_BONUS  = 2.0     # was 5.0, reduce dominance
SAFETY_URGENT_COEF = -30.0  # was -50.0

# Kept proportional to the ideal formation edge (EDGE_TARGET) rather than a
# fixed absolute, so it automatically stays meaningful if TARGET_DIST
# changes. Tried tightening this from ~2.7x to 1.6x on the theory that the
# swarm was defaulting to "just stay very loose" as a cheap safety strategy
# (diameter 18-21 in successful episodes) instead of converging toward the
# tight ideal (~7.8) -- measured result: collision_rate got WORSE (0.31 ->
# 0.47), not better. The wide formation was functioning as real safety
# margin, not laziness; removing it made things more dangerous. Reverted to
# the value that was actually measured to work.
COHESION_MARGIN = 2.7
COHESION_LIMIT   = COHESION_MARGIN * EDGE_TARGET   # ~21.1
COHESION_WEIGHT = -0.05

# Diameter floor (2026-08-14) -- COHESION_LIMIT/COHESION_WEIGHT above only ever
# penalize the swarm being too LOOSE; nothing penalized it being too TIGHT, so
# there was no counter-pressure against diameter shrinking as r_track pulled
# harder over training (confirmed across every NUM_AGENTS=3 run so far:
# avg_min_dist/swarm_diameter trend down from a mid-run peak toward the
# collision boundary as entropy collapses). MIN_DIAMETER is a user-specified
# value, not physically re-derived like COHESION_LIMIT -- picked to sit where
# every --best checkpoint across every run so far has actually landed (11-13
# diameter, 2-8% collision_rate), comfortably above the collapsed-regime
# diameters (~8.5-9.5) seen whenever collision_rate climbed, and above
# EDGE_TARGET=7.80 (the theoretical ideal-packing edge) -- so this will pull
# somewhat against r_track's individual radial pull by design; that tension is
# expected, not a bug, and is the whole point of testing this. Explicitly a
# "confirm the hypothesis" value, not a derived one.
MIN_DIAMETER = 10.0
# Deliberately NOT reusing COHESION_WEIGHT (-0.05) for the floor -- that
# weight is why the existing upper-bound cohesion term never mattered (it
# never got the chance to be tested against a real diameter excursion, but
# -0.05 * a few units of violation is negligible next to TRACK_WEIGHT=-2.0's
# pull). Sized to be roughly comparable to TRACK_WEIGHT so it can actually
# compete instead of being dominated by it. Starting guess, not derived --
# first thing to retune if this doesn't move the needle.
#
# Disabled -- set to 0.0 (2026-08-17), not deleted, so this is a one-line
# revert if the test below is wrong. This floor was built to stop dangerous
# tight convergence before the closing-speed brake existed; the brake now
# handles that deterministically, independent of diameter. With vision-based
# tracking added on top, this floor looks like it's now actively
# counterproductive rather than merely redundant: a single-seed N3 run with
# it active showed swarm_diameter pinned at 12-15 (vs the pre-floor/pre-brake
# 10-13) and target_lost_rate oscillating 15-40% with no improving trend
# across the full 600k steps, while r_track sat notably worse (-4 to -6.5)
# than earlier runs. Hypothesis: a wider ring geometrically implies a larger
# individual radius from the target for evenly-spaced agents (same relation
# TARGET_DIST/EDGE_TARGET already uses), so pulling diameter up toward/above
# MIN_DIAMETER=10 was plausibly pushing individual drones past SENSOR_RANGE
# (~6.9-7.2) more often than necessary. COHESION_LIMIT/COHESION_WEIGHT above
# (the upper-bound loose-formation penalty) are untouched -- that's a
# separate concern from the floor and isn't implicated by this data.
# Unverified as of this comment -- next single-seed N3 run on this same
# branch is the check.
DIAMETER_FLOOR_WEIGHT = 0.0

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

# Was hardcoded (-0.05) directly in formation_env.py; a 3-way parallel sweep
# at NUM_AGENTS=2 (-0.05 / -0.5 / -2.0, 100-episode deterministic eval each)
# measured a real, monotonic trade-off: tracking_rmse improved 3.23 -> 2.07
# -> 1.91 as the weight strengthened, but collision_rate got worse in lock
# step, 0.07 -> 0.15 -> 0.24 -- pushing drones to match the target's
# velocity competes with the small evasive corrections collision avoidance
# needs. Returns were also diminishing (-0.5 -> -2.0 bought only 8% more
# RMSE improvement for another 9 points of collision rate). Settled on
# -0.15: a modest, lower-risk pull toward the -0.5 result without chasing
# tracking precision at real cost to collision safety, which is still the
# thing that hasn't been proven reliable at higher NUM_AGENTS. Revisit this
# trade-off after collision avoidance is solid at full agent count, not
# before. Env-var overridable for further sweeps.
VELOCITY_WEIGHT = float(os.environ.get("VELOCITY_WEIGHT", -0.15))

# Vision-based cooperative target tracking (2026-08-14) -- replaces the old
# ground-truth target telemetry (every drone knew the target's exact
# position/velocity every step, no matter how far away) with a scenario
# where each drone must have the target within an omnidirectional,
# range-limited sensor to get a direct reading, shares that reading with the
# swarm, and falls back to short-term dead-reckoning (then eventual mission
# failure) if no one currently has contact. See envs/formation_env.py for
# the mechanism; these are just the tunable/derived constants.
#
# SENSOR_RANGE reuses the exact spawn-radius formula from reset()
# (TARGET_DIST + 2*REACTION_DIST) rather than inventing a new number --
# drones start each episode at the edge of sensor range by construction (a
# "just detected it, closing in" narrative), and if tracking stays near
# TARGET_DIST they remain comfortably inside it. N-dependent, like
# TARGET_DIST itself.
SENSOR_RANGE = TARGET_DIST + 2 * REACTION_DIST

# 2 seconds, per explicit design discussion -- worth being honest about what
# this does and doesn't protect against in the current sim: the target used
# to move at constant velocity for the whole episode (fixed direction/speed
# at reset, never updated), which made dead-reckoning from a last-known
# reading mathematically exact, not an approximation. Phase 3 (2026-08-20)
# closed that gap -- the target now redirects periodically mid-episode (see
# TARGET_REDIRECT_INTERVAL_STEPS) -- so this grace period is now protecting
# against real estimate drift, not just a formality.
#
# Env-var overridable (2026-08-20) -- was a flat 2.0, tuned against
# MAX_STEPS=200 (10s) episodes and never revisited when Phase 3 raised
# MAX_STEPS to 1800 (90s). That mismatch is suspected as the primary cause
# of Phase 3's target_lost_rate sitting near 100% and never improving: a
# fixed 2-second grace window has ~9x more opportunities to be exceeded by
# an ordinary, recoverable contact gap over a 9x longer episode, even with
# no change in the swarm's actual moment-to-moment tracking competence.
# Sweeping 6/10/18s across seeds to find out empirically rather than
# guessing at one value.
LOST_TIMEOUT_SEC   = float(os.environ.get("LOST_TIMEOUT_SEC", 2.0))
LOST_TIMEOUT_STEPS = int(LOST_TIMEOUT_SEC / DT)   # = 40 at 2.0s/DT=0.05

# Ramps 0 (swarm just lost contact) -> CONTACT_URGENT_COEF (right at the
# LOST_TIMEOUT_STEPS boundary, where target_lost termination fires) --
# same shape as SAFETY_URGENT_COEF's ramp toward COLLISION_DIST, reusing an
# already-validated idiom rather than inventing a new penalty shape. Applies
# identically to every agent (it's about swarm-wide contact, not individual
# blame), which is what makes "someone else regaining contact helps
# everyone's reward" a genuinely cooperative signal rather than a per-agent
# one. Starting guess at the same order of magnitude as SAFETY_URGENT_COEF,
# not derived -- first thing to retune if this doesn't move the needle.
CONTACT_URGENT_COEF = -30.0

# Terminal penalty on the step target_lost fires, same idea as
# r_collision_global's -300 -- smaller magnitude than collision on purpose:
# both are mission failures, but a real drone-drone collision risks hardware
# damage in a way that losing visual on a target (recoverable in principle,
# e.g. by returning to base) doesn't, so collision should stay the higher
# priority if the two ever trade off against each other. Starting guess.
TARGET_LOST_PENALTY = -200.0

# Direct reward penalty on brake engagement itself (2026-08-19) -- until now,
# the only pressure against needing the closing-speed brake was indirect:
# r_safety's urgent-zone penalty shares the brake's trigger threshold
# (SAFE_DIST_ENTER), but is proximity-based, not action-based, so a policy
# could keep commanding more-than-safe closing speed without anything
# specifically penalizing that choice beyond ambient proximity. Confirmed
# this mattered at NUM_AGENTS=4: brake engagement was continuous and
# substantial throughout a whole 3M-step run (0.002-0.016 mean speed
# removed/agent/step), not the rare, sparse engagement seen at NUM_AGENTS=3
# (0.0001-0.0069) -- "a good driver rarely triggers ABS" was the design
# goal, and nothing in the reward enforced it. brake_reduction (already
# computed every step in envs/formation_env.py's step(), previously only
# logged) is now also fed back as reward.
#
# v1 (first Kaggle test, 2026-08-19) penalized brake_reduction linearly from
# zero: r_brake = BRAKE_PENALTY_COEF * brake_reduction. Measured result:
# collision_rate genuinely fixed (88/59/269 nonzero rollouts out of 1465
# dropped to 14/10/15, all three seeds clean for their entire last 10
# rollouts) -- but swarm_diameter/avg_min_dist both ran noticeably wider
# than before and tracking_rmse worsened slightly. Diagnosis: a linear-from-
# zero penalty can't distinguish a trivial, routine nudge (a fraction of a
# percent of MAX_ACTION_SPEED) from a real emergency correction -- it taxes
# both, so the policy's cheapest way to minimize the penalty is to avoid
# entering the brake's trigger zone at all, i.e. spread out more than
# necessary, exactly like "a good driver rarely triggers ABS" doesn't mean
# a good driver's ABS light can never so much as flicker.
#
# v2: only penalize the excess above BRAKE_PENALTY_THRESHOLD, zero below it
# -- r_brake = BRAKE_PENALTY_COEF * max(0, brake_reduction -
# BRAKE_PENALTY_THRESHOLD), see envs/formation_env.py's _get_reward. Below
# the threshold this is now a complete no-op (matches "normal braking" being
# free), so it shouldn't create the same pressure to avoid the zone
# altogether -- only to avoid needing a large correction once inside it.
# BRAKE_PENALTY_THRESHOLD is derived from the brake's own geometry, not
# picked independently: max_closing = MAX_ACTION_SPEED * (d-COLLISION_DIST)/
# (SAFE_DIST_ENTER-COLLISION_DIST) means a brake_reduction of
# 0.5*MAX_ACTION_SPEED corresponds to a full-speed closing approach already
# halfway through the safety zone toward COLLISION_DIST -- a genuinely
# aggressive, late correction, not a minor one. BRAKE_PENALTY_COEF itself
# unchanged from v1 (only the zero-point moved) -- still a starting guess,
# first thing to retune if this doesn't move the needle either.
BRAKE_PENALTY_COEF = -10.0
BRAKE_PENALTY_THRESHOLD = 0.5 * MAX_ACTION_SPEED   # = 0.60

# Ground reward/penalty (Phase 3, 2026-08-20) -- same shapes as the
# inter-agent case, reused rather than invented fresh: GROUND_URGENT_COEF
# mirrors SAFETY_URGENT_COEF's ramp toward the hard limit (COLLISION_DIST
# there, GROUND_Z here); GROUND_STRIKE_PENALTY matches r_collision_global's
# -300 magnitude rather than TARGET_LOST_PENALTY's deliberately-softer
# -200 -- a real ground strike is at least as severe as an inter-agent
# collision (see PHASE2_CHECKPOINT.md's diagnose_horizon.py finding for
# why this exists at all), not a softer failure mode like losing contact.
GROUND_URGENT_COEF   = -30.0
GROUND_STRIKE_PENALTY = -300.0

# Cruise-altitude preference + vertical smoothing (2026-08-20, user-reported
# jiggling -- confirmed real, not just a hunch: measured 8-25% of steps
# flipping vertical-velocity sign on a trained Phase 3 checkpoint, one
# agent briefly down to z=0.98). Separate concept from GROUND_SAFE_ENTER
# (1.20, the hard safety floor derived from collision-avoidance reaction
# distance) -- this is a softer "prefer not to loiter this low" preference,
# a comfortable cruise height rather than a crash-avoidance boundary.
CRUISE_ALT_MIN  = 1.5
CRUISE_ALT_COEF = -10.0
# Deterministic smoothing on commanded vertical velocity, same "don't just
# hope the policy learns this" reasoning already validated for collision
# (the brake) and the ground (the clamp) -- blends the new Z command with
# last step's actual Z velocity rather than allowing an instant reversal.
# 0.3 = 30% new command, 70% carried over -- a starting guess at enough
# damping to visibly reduce sign-flipping without making vertical response
# sluggish; first thing to retune if it overshoots either way.
Z_SMOOTHING_ALPHA = 0.3

# Per-axis action scaling (Phase 3, 2026-08-20) -- see envs/formation_env.py
# step()'s comment. 0.6x is a starting guess at a real-ish climb-rate-vs-
# lateral-speed ratio, not measured -- first thing to retune if this
# doesn't move the needle, same spirit as BRAKE_PENALTY_COEF above.
MAX_ACTION_SPEED_Z = 0.6 * MAX_ACTION_SPEED   # = 0.72

# Dynamic target motion (Phase 3, 2026-08-20) -- see envs/formation_env.py
# step()'s comment and KNOWN_ISSUES.md item 9. 500 steps = 25s, giving a
# 1800-step (90s) episode room for ~3 mid-episode redirects.
TARGET_REDIRECT_INTERVAL_STEPS = 500
