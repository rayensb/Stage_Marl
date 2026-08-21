"""
FormationEnv3D — 4-agent PettingZoo ParallelEnv, 3D space.

Reward-balance fix: track weight increased, safety magnitude reduced,
old per-locked-neighbor "diverge" penalty (had a relock loophole) replaced
with a GLOBAL swarm-cohesion penalty based on swarm_diameter -- can't be
gamed by relocking onto a different neighbor.

NEIGHBOR GRAPH DESIGN CHOICE: mutual (reciprocal) k-nearest-neighbors.
Each drone computes its own top-k nearest candidates; a candidate is only
"locked" if the relationship is mutual (A in B_top_k AND B in A_top_k).
Mutual k-NN has no connectivity guarantee on its own (can fragment into
isolated sub-swarms), so _repair_connectivity() force-connects across
components using global position knowledge computed centrally by the
simulator -- this makes EXECUTION decentralized (each actor still only
receives its own local/relative observations) but NOT the neighbor-graph
maintenance mechanism itself. Precise framing for writeups: "decentralized
execution", not "fully decentralized system".

SCOPE: TARGET_DIST in config.py is derived from the regular-tetrahedron
packing ratio for exactly 4 points on a sphere -- it is not a general-N
formula, even though most of this env (NUM_AGENTS, K_NEIGHBORS) is
N-generic. This is a 4-agent study; revisit the derivation before changing
NUM_AGENTS.

SUPERSEDED (was here through the xyz-spread branch, 2026-08-19): r_spread
used to consider only the XY (horizontal) projection of neighbor bearings,
blind to true 3D angular separation -- a drone directly above/below a
neighbor barely registered. Replaced by true pairwise 3D angular
separation between neighbor direction vectors -- see _IDEAL_NEIGHBOR_ANGLE
and _get_reward below.

VISION-BASED COOPERATIVE TARGET TRACKING (2026-08-14): the target is no
longer ground-truth telemetry every drone knows regardless of distance. Each
drone gets a direct reading only when the target is within SENSOR_RANGE
(omnidirectional -- no heading/FOV-cone state exists in this sim, see
config.py), shared instantly across the swarm whenever any drone has
contact, and dead-reckoned (last known position + last known velocity x
elapsed time) for up to LOST_TIMEOUT_STEPS if no one currently does --
_update_target_track() computes this once per step and _get_obs/_get_reward
both read the result, never self.pos_t directly (the one exception:
_dist_to_target() stays ground-truth on purpose, as a diagnostic-only
"true_track_err" in infos, to check the estimate against reality -- it is
NOT used anywhere in the reward or observation). Beyond LOST_TIMEOUT_STEPS
the episode terminates (target_lost) -- a swarm with no trusted estimate has
nothing left to learn from, same reasoning as collision termination.
Deliberately NOT included yet, to keep this one controlled change: UWB-
realistic (noisy, range-only) neighbor sensing, sensing noise/occlusion for
the target reading itself, a directional FOV cone, or non-constant-velocity
target motion -- see config.py's SENSOR_RANGE/LOST_TIMEOUT comments for why
each is deferred rather than skipped.

ACTIVE SEARCH (2026-08-21): added after Phase 3's LOST_TIMEOUT_SEC sweep
showed the grace-period LENGTH was not the primary lever on target_lost_rate
(3 of 4 completed 6s/10s runs stayed near-100% broken; the one exception
looked like a lucky seed, not a working config -- see EXPERIMENT_LOG.md).
Previously, every agent coasted toward the SAME dead-reckoned point
(self._track_pos_est) while contact was lost, which only ever gets more
wrong over time and gives the swarm no way to actively improve its odds of
reacquiring the target. Now, the moment the whole swarm loses contact, each
agent is assigned its own fixed search heading (_assign_search_directions,
called once per loss-of-contact event, not every step) and _get_obs/
_get_reward track a PER-AGENT synthetic waypoint receding from the
last-known position in that heading, reusing the exact same r_track/
r_velocity reward math that already pulls a drone toward a target estimate
-- no new reward term needed. The instant any agent's real sensor comes
within SENSOR_RANGE of the true target, the existing contact-sharing logic
in _update_target_track() takes over unchanged. If nobody finds it before
LOST_TIMEOUT_STEPS, the episode still ends in target_lost exactly as
before -- search is meant to improve the odds within that window, not
remove the window itself.
"""

import math
import functools
import itertools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUM_AGENTS, K_NEIGHBORS, TARGET_DIST, SAFE_DIST_ENTER, SAFE_DIST_EXIT,
    COLLISION_DIST, DIVERGE_DIST, MAX_STEPS, DT, MAX_ACTION_SPEED, REACTION_DIST,
    OBS_MAX_DIST, OBS_MAX_VEL, OBS_OWN_DIM,
    TRACK_WEIGHT, SAFETY_MAX_BONUS, SAFETY_URGENT_COEF, EDGE_TARGET,
    COHESION_LIMIT, COHESION_WEIGHT, MIN_DIAMETER, DIAMETER_FLOOR_WEIGHT,
    JOINT_BONUS, JOINT_TRACK_TOL, VELOCITY_WEIGHT,
    SENSOR_RANGE, LOST_TIMEOUT_STEPS, CONTACT_URGENT_COEF, TARGET_LOST_PENALTY,
    BRAKE_PENALTY_COEF, BRAKE_PENALTY_THRESHOLD,
    GROUND_Z, GROUND_SAFE_ENTER, GROUND_SAFE_EXIT,
    GROUND_URGENT_COEF, GROUND_STRIKE_PENALTY,
    MAX_ACTION_SPEED_Z, TARGET_REDIRECT_INTERVAL_STEPS,
    CRUISE_ALT_MIN, CRUISE_ALT_COEF, Z_SMOOTHING_ALPHA,
    SEARCH_SPEED,
)

# XYZ r_spread (2026-08-19, xyz-spread branch; angle corrected 2026-08-20
# after supervisor review): ideal pairwise angular separation between
# neighbor direction vectors, for however many locked neighbors a drone
# has. NOT the same quantity as config.py's _PACKING_RATIO/TARGET_DIST --
# that's the angle the TARGET sees between two drones (global, e.g. 109.47
# deg for N=4's regular tetrahedron); this is the angle a DRONE sees
# between its neighbors (local). First implementation conflated the two
# and used the global Tammes-optimal-free-points angle (180/120 deg for
# K=2/3) here, which is wrong for this purpose -- it was pushing neighbor
# directions wider than the actual target formation has, fighting
# convergence rather than helping it.
#
# Correct value, verified both analytically and numerically (placing
# vertices as standard basis vectors e_1..e_{k+1}, the angle between any
# two edges (e_i - e_1, e_j - e_1) at a shared vertex has cos = 1/2 for
# ANY k -- see deployment/docs or test_geometry.py for the check): under
# full connectivity (K_NEIGHBORS = NUM_AGENTS - 1, this project's only
# mode), the regular-simplex local vertex angle is exactly 60 degrees,
# independent of K. Only K in {2, 3} ever occurs here (EFFECTIVE_K at
# NUM_AGENTS in {3, 4}; NUM_AGENTS=2 gives K=1, which r_spread already
# skips entirely below) -- both map to the same constant, which is why
# this is no longer a per-K dict.
_IDEAL_NEIGHBOR_ANGLE = math.pi / 3


class FormationEnv3D(ParallelEnv):
    metadata = {"name": "formation_env_3d_v2"}

    def __init__(self, num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS):
        self.possible_agents = [f"drone{i+1}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.k = min(k_neighbors, num_agents - 1)

        self._obs_dim = OBS_OWN_DIM + 7 * self.k
        self._act_dim = 3

        self.pos, self.vel = {}, {}
        self.locked = {}
        self.pos_t = np.zeros(3, np.float32)
        self.step_count = 0
        self._current_diameter = 0.0
        self.np_random = np.random.default_rng()

        # Vision-tracking state -- see _update_target_track() and the module
        # docstring. Defensive zero-init only; _update_target_track() always
        # runs (from reset()) before any of this is read, and the initial
        # spawn is constructed to guarantee direct contact at t=0 (spawn
        # radius's upper bound == SENSOR_RANGE, see config.py), so the
        # "no contact yet" branch below never actually fires on a fresh env.
        self._last_known_pos = np.zeros(3, np.float32)
        self._last_known_vel = np.zeros(3, np.float32)
        self._track_pos_est = np.zeros(3, np.float32)
        self._track_vel_est = np.zeros(3, np.float32)
        self._steps_since_contact = 0
        self._track_confidence = 0.0
        self._target_lost = False
        self._observer_count = 0
        self._direct_contact = {}
        self._contact_centroid = None
        # Active search (2026-08-21) -- see _assign_search_directions/
        # _update_target_track. Defensive zero-init only, same reasoning as
        # the fields above: _update_target_track() always runs before any
        # of this is read, and a fresh env starts in direct contact (see
        # above), so _search_dirs is never actually consulted un-assigned.
        self._search_dirs = {}
        self._effective_pos_est = {}
        self._effective_vel_est = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)

    def _resample_target_motion(self):
        """Sets self._target_dir/_target_speed from the same distribution
        reset() always has -- factored out (Phase 3, 2026-08-20) so step()
        can call it too for mid-episode redirects (see TARGET_REDIRECT_
        INTERVAL_STEPS in step()) without duplicating the sampling logic."""
        self._target_dir = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        self._target_dir[2] *= 0.2
        self._target_dir /= (np.linalg.norm(self._target_dir) + 1e-6)
        self._target_speed = self.np_random.uniform(0.3, 1.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0

        self.pos_t = self.np_random.uniform(-3.0, 3.0, 3).astype(np.float32)
        self.pos_t[2] = self.np_random.uniform(2.0, 3.0)

        for a in self.agents:
            ang = self.np_random.uniform(0, 2 * math.pi)
            elev = self.np_random.uniform(-0.3, 0.3)
            d = self.np_random.uniform(TARGET_DIST, TARGET_DIST + 2 * REACTION_DIST)
            offset = np.array([math.cos(ang) * math.cos(elev),
                                math.sin(ang) * math.cos(elev),
                                math.sin(elev)]) * d
            self.pos[a] = (self.pos_t + offset).astype(np.float32)
            self.vel[a] = np.zeros(3, np.float32)

        # Ground-safety and inter-agent-overlap resolution have to run as a
        # joint fixed point, not one-then-forget (Phase 3, 2026-08-20):
        # _resolve_overlaps() operates in full unconstrained 3D with no
        # ground awareness and can push a spawn well below z=0 (confirmed
        # empirically, NUM_AGENTS=3 seed 0: -1.14, far past what the raw
        # elevation-offset spawn geometry alone would ever produce). But
        # clamping Z afterward, on its own, can just as easily collapse
        # vertical separation the overlap pass had relied on to keep two
        # agents apart -- also confirmed empirically (same seed: clamping
        # drone3's z up to the floor put it 2.38 from drone1, inside
        # COLLISION_DIST, an immediate step-0 collision that didn't exist
        # pre-clamp). Alternate both passes -- same POCS-style "repeat until
        # nothing needs correcting" idea already used for the brake -- until
        # a round makes no ground correction (implying overlaps are also
        # already resolved from that round's own _resolve_overlaps() call).
        for _ in range(20):
            self._resolve_overlaps()
            any_clamped = False
            for a in self.agents:
                if self.pos[a][2] < GROUND_SAFE_ENTER:
                    self.pos[a][2] = GROUND_SAFE_ENTER
                    any_clamped = True
            if not any_clamped:
                break

        self._resample_target_motion()

        self._relock_all()
        self._current_diameter = self.get_swarm_stats()["swarm_diameter"]

        # Spawn radius's upper bound equals SENSOR_RANGE by construction (see
        # config.py's SENSOR_RANGE comment), so every drone starts in direct
        # contact -- steps_since_contact resets to 0 here regardless of
        # whatever it was at the end of the previous episode.
        self._steps_since_contact = 0
        self._update_target_track()

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _resolve_overlaps(self):
        for _ in range(50):
            worst, worst_d = None, COLLISION_DIST + 0.3
            for a, b in itertools.combinations(self.agents, 2):
                d = np.linalg.norm(self.pos[a] - self.pos[b])
                if d < worst_d:
                    worst_d, worst = d, (a, b)
            if worst is None:
                break
            a, b = worst
            diff = self.pos[b] - self.pos[a]
            n = np.linalg.norm(diff) + 1e-6
            self.pos[b] = (self.pos[b] + diff / n * 0.3).astype(np.float32)

    def _compute_all_candidates(self):
        cand = {}
        for a in self.agents:
            dists = [(o, np.linalg.norm(self.pos[a] - self.pos[o]))
                      for o in self.agents if o != a]
            dists.sort(key=lambda x: x[1])
            cand[a] = [o for o, _ in dists[:self.k]]
        return cand

    def _relock_all(self):
        cand = self._compute_all_candidates()
        locked = {}
        for a in self.agents:
            mutual = [b for b in cand[a] if a in cand[b]]
            if not mutual and cand[a]:
                mutual = cand[a][:1]
            locked[a] = mutual[:self.k]
        self._repair_connectivity(locked)
        # _repair_connectivity appends connector agents to whichever end,
        # not by distance -- re-sort so slot 0 is always the nearest locked
        # neighbor and slot 1 the second-nearest, regardless of whether that
        # agent's list went through a repair. Without this, a repair event
        # could silently swap what slot 0 vs slot 1 means in the observation
        # (a discontinuous change in feature semantics the network has to
        # learn around instead of a stable "nearest, second-nearest" contract).
        for a in self.agents:
            locked[a].sort(key=lambda n: np.linalg.norm(self.pos[n] - self.pos[a]))
        self.locked = locked

    def _repair_connectivity(self, locked):
        """Mutual k-NN locking has no guarantee of staying connected -- the
        swarm can fragment into isolated sub-groups (e.g. A<->B, C<->D) that
        never observe each other and drift apart with nothing to stop them.
        Repeatedly force-connect the closest pair of agents across different
        components until the whole swarm is one component, evicting each
        side's farthest current lock if it's already at the k-neighbor cap.
        This uses global position knowledge, which the simulator already has
        centrally -- it is not a decentralized operation and would need a
        distributed protocol before running on real, comms-limited hardware.
        """
        parent = {a: a for a in self.agents}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for a in self.agents:
            for b in locked[a]:
                union(a, b)

        while True:
            components = {}
            for a in self.agents:
                components.setdefault(find(a), []).append(a)
            if len(components) <= 1:
                break

            comp_list = list(components.values())
            best = None
            for i in range(len(comp_list)):
                for j in range(i + 1, len(comp_list)):
                    for a in comp_list[i]:
                        for b in comp_list[j]:
                            d = np.linalg.norm(self.pos[a] - self.pos[b])
                            if best is None or d < best[0]:
                                best = (d, a, b)
            _, a, b = best

            for x, y in ((a, b), (b, a)):
                if y in locked[x]:
                    continue
                if len(locked[x]) < self.k:
                    locked[x].append(y)
                else:
                    farthest = max(locked[x], key=lambda o: np.linalg.norm(self.pos[x] - self.pos[o]))
                    locked[x].remove(farthest)
                    locked[x].append(y)
            union(a, b)

    def _update_target_track(self):
        """Vision-based cooperative target tracking -- see the module
        docstring. Computes, once per step (using current positions), which
        drones currently have the target within SENSOR_RANGE, the swarm's
        shared best estimate of target position/velocity (direct if anyone
        has contact, dead-reckoned otherwise), how long it's been since
        anyone last did, and the resulting confidence/target_lost state.
        Must run after self.pos/self.pos_t are updated for this step (or,
        from reset(), after the initial spawn) and before _get_obs/
        _get_reward, which both read the results as instance attributes
        rather than recomputing them per agent.
        """
        self._direct_contact = {
            a: float(np.linalg.norm(self.pos[a] - self.pos_t)) <= SENSOR_RANGE
            for a in self.agents
        }
        contacts = [a for a in self.agents if self._direct_contact[a]]
        self._observer_count = len(contacts)

        if contacts:
            # Every in-contact drone reads the identical true position/
            # velocity in this noiseless iteration (no per-drone sensing
            # error modeled yet -- deliberately deferred, see the module
            # docstring), so "average the observers' readings" is a no-op
            # today but the structurally correct thing to do once per-drone
            # noise exists.
            self._last_known_pos = self.pos_t.copy()
            self._last_known_vel = (self._target_dir * self._target_speed).astype(np.float32)
            self._steps_since_contact = 0
        else:
            # Active search (2026-08-21): assign fresh search headings
            # exactly once, on the first step of a NEW loss-of-contact
            # event -- self._steps_since_contact is still 0 here iff
            # contact just ended this step (it's only ever reset to 0 in
            # the branch above), so this fires once per event, not every
            # step of a continuing search. See _assign_search_directions
            # and the module docstring.
            if self._steps_since_contact == 0:
                self._assign_search_directions()
            self._steps_since_contact += 1

        self._track_confidence = max(0.0, 1.0 - self._steps_since_contact / LOST_TIMEOUT_STEPS)
        self._target_lost = self._steps_since_contact > LOST_TIMEOUT_STEPS

        elapsed = self._steps_since_contact * DT
        self._track_pos_est = (self._last_known_pos + self._last_known_vel * elapsed).astype(np.float32)
        self._track_vel_est = self._last_known_vel

        # Per-agent effective estimate (2026-08-21, active search): equals
        # the shared self._track_pos_est/_track_vel_est above whenever
        # anyone has contact (self._steps_since_contact == 0 iff contacts
        # this step, per above) -- a strict generalization, not a parallel
        # code path, so this reduces to the pre-search behavior exactly
        # whenever nobody is currently searching. Diverges per-agent only
        # during a lost period: each agent's own waypoint recedes from the
        # last-known position in ITS assigned heading, rather than every
        # agent coasting toward the same increasingly-stale shared point.
        # _get_obs/_get_reward read this dict, never the shared fields
        # directly, so both automatically pick up search behavior with no
        # further reward/observation-shape changes.
        self._effective_pos_est = {}
        self._effective_vel_est = {}
        for a in self.agents:
            if self._steps_since_contact == 0:
                self._effective_pos_est[a] = self._track_pos_est
                self._effective_vel_est[a] = self._track_vel_est
            else:
                search_vel = (self._search_dirs[a] * SEARCH_SPEED).astype(np.float32)
                self._effective_pos_est[a] = (self._last_known_pos + search_vel * elapsed).astype(np.float32)
                self._effective_vel_est[a] = search_vel

        self._contact_centroid = (
            np.mean([self.pos[a] for a in contacts], axis=0).astype(np.float32)
            if contacts else None
        )

    def _assign_search_directions(self):
        """Called exactly once per loss-of-contact event (see
        _update_target_track), not every step while lost -- each agent
        gets a fixed heading to fly in a straight line from the
        last-known position until contact is regained or the episode
        ends, rather than continuously re-randomizing (which would never
        let a drone actually cover ground in a given direction). Evenly
        spread around a randomized base angle so agents search different
        areas without the swarm collapsing onto the same absolute compass
        pattern every single episode. Horizontal-only (z=0) -- altitude
        is independently governed by GROUND_SAFE_ENTER/CRUISE_ALT_MIN,
        not something search should fight."""
        base_angle = self.np_random.uniform(0, 2 * math.pi)
        n = len(self.agents)
        self._search_dirs = {}
        for i, a in enumerate(self.agents):
            angle = base_angle + 2 * math.pi * i / n
            self._search_dirs[a] = np.array(
                [math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)

    def _apply_brake(self, raw_vel):
        """Extracted from step() (2026-08-19) so it's callable standalone --
        e.g. from a real-vehicle deployment driving self.pos from live
        telemetry instead of this env's own integrator. Reads self.pos/
        self.agents; does not mutate them or self.vel, and does not advance
        physics -- callers do that with the returned velocities. See step()
        for the full multi-pass/POCS rationale. Returns (corrected_vel,
        brake_reduction) dicts, one entry per agent in self.agents --
        signature intentionally unchanged (2026-08-20 instrumentation below)
        since deployment/inference_node.py depends on this exact unpack.

        Diagnostics from this call are stashed on self instead of returned,
        for the same reason: self._last_brake_passes (int, how many passes
        this call took), self._last_brake_violation (float, worst remaining
        constraint violation after the loop -- should be ~0 if converged
        before hitting the pass cap, nonzero and positive means it didn't),
        self._last_brake_k_active (dict, how many other agents were within
        SAFE_DIST_ENTER of each agent -- lets callers split brake_reduction
        by whether it happened solo (1 active neighbor) or under multi-
        neighbor competition (2+), the mechanism N-aware margin targets."""
        raw_vel = {a: v.copy() for a, v in raw_vel.items()}
        brake_reduction = {a: 0.0 for a in self.agents}

        k_active = {a: 0 for a in self.agents}
        for a in self.agents:
            for b in self.agents:
                if b == a:
                    continue
                if float(np.linalg.norm(self.pos[b] - self.pos[a])) < SAFE_DIST_ENTER:
                    k_active[a] += 1
        self._last_brake_k_active = k_active

        passes_used = 0
        for pass_num in range(NUM_AGENTS):
            passes_used = pass_num + 1
            any_correction = False
            for a in self.agents:
                v = raw_vel[a]
                for b in self.agents:
                    if b == a:
                        continue
                    diff = self.pos[b] - self.pos[a]
                    d = float(np.linalg.norm(diff))
                    if d < SAFE_DIST_ENTER and d > 1e-6:
                        dir_to_b = diff / d
                        v_closing = float(np.dot(v, dir_to_b))
                        if v_closing > 0:
                            max_closing = MAX_ACTION_SPEED * max(0.0, (d - COLLISION_DIST) / (SAFE_DIST_ENTER - COLLISION_DIST))
                            if v_closing > max_closing:
                                excess = v_closing - max_closing
                                v = v - excess * dir_to_b
                                brake_reduction[a] += excess
                                any_correction = True
                raw_vel[a] = v.astype(np.float32)
            if not any_correction:
                break
        self._last_brake_passes = passes_used

        # Post-hoc check, diagnostic only -- doesn't feed back into raw_vel.
        # Nonzero only if the pass cap was hit before the early-exit fired
        # (see the CAUTION comment in step()); this is what actually turns
        # that caveat into a checkable number instead of an assumption.
        max_violation = 0.0
        for a in self.agents:
            for b in self.agents:
                if b == a:
                    continue
                diff = self.pos[b] - self.pos[a]
                d = float(np.linalg.norm(diff))
                if d < SAFE_DIST_ENTER and d > 1e-6:
                    dir_to_b = diff / d
                    v_closing = float(np.dot(raw_vel[a], dir_to_b))
                    max_closing = MAX_ACTION_SPEED * max(0.0, (d - COLLISION_DIST) / (SAFE_DIST_ENTER - COLLISION_DIST))
                    max_violation = max(max_violation, v_closing - max_closing)
        self._last_brake_violation = max_violation

        return raw_vel, brake_reduction

    def _apply_ground_clamp(self, raw_vel):
        """Ground-plane counterpart to _apply_brake (Phase 3, 2026-08-20).
        Per-agent and independent -- unlike two drones closing on each
        other, the ground doesn't move and isn't itself a constraint that
        can conflict with another agent's, so no multi-pass/POCS loop is
        needed here, one pass is exact. Caps how fast an agent may still
        be descending as its altitude approaches GROUND_Z, ramping to zero
        exactly at the floor -- same shape as _apply_brake's closing-speed
        cap, against a fixed plane instead of another agent's position.
        Uses MAX_ACTION_SPEED_Z (not the horizontal MAX_ACTION_SPEED) since
        this is specifically about vertical motion. Returns (corrected_vel,
        ground_reduction) dicts, one entry per agent in self.agents --
        signature mirrors _apply_brake's, for the same reuse reason."""
        raw_vel = {a: v.copy() for a, v in raw_vel.items()}
        ground_reduction = {a: 0.0 for a in self.agents}
        for a in self.agents:
            z = float(self.pos[a][2])
            v = raw_vel[a]
            if z < GROUND_SAFE_ENTER and v[2] < 0:
                max_descend = MAX_ACTION_SPEED_Z * max(0.0, (z - GROUND_Z) / (GROUND_SAFE_ENTER - GROUND_Z))
                if -v[2] > max_descend:
                    excess = -v[2] - max_descend
                    v[2] += excess
                    ground_reduction[a] = excess
            raw_vel[a] = v.astype(np.float32)
        return raw_vel, ground_reduction

    def step(self, actions):
        self.step_count += 1

        # Closing-speed brake (2026-08-14): deterministic action-space safety
        # layer, not a reward incentive -- five reward-shape attempts (entropy-
        # recovery budget/timing, r_safety zone reshape, diameter floor, longer
        # training) all failed to stop collision_rate climbing over training,
        # and instrumentation (log_std_mean, mean_action_abs) showed why: the
        # policy wasn't losing exploration noise (log_std barely moved), it was
        # committing to increasingly large-magnitude actions over training --
        # a drone closing at high commanded speed has less room to back out of
        # a close encounter than one making smaller corrections, independent
        # of how much stochastic noise sits on top. This caps only the CLOSING
        # component of velocity (the part actually shrinking the gap to a
        # given other agent), once already inside SAFE_DIST_ENTER -- the
        # already-documented "urgent backoff" boundary -- ramping linearly to
        # zero exactly at COLLISION_DIST. Lateral/evasive motion is untouched;
        # outside SAFE_DIST_ENTER this is a no-op. Reuses only already-derived
        # constants, no new ones. Uses pre-step positions for every agent
        # (computed before any position is updated) so this reflects
        # simultaneous action, not sequential. Doesn't touch the reward directly
        # -- r_safety's urgent penalty below still applies in full, and
        # brake_reduction now also feeds r_brake (see _get_reward) -- so the
        # policy has every incentive to avoid needing this, not just a safety
        # net once it's already there.
        #
        # Multi-pass convergence (2026-08-19): each per-neighbor closing-speed
        # cap is a halfspace constraint on velocity (dot(v, dir_to_b) <=
        # max_closing), and the correction below is exactly the Euclidean
        # projection onto that halfspace. A single sequential sweep (the
        # original version) applies each projection once and stops -- fine
        # with only one other agent in range, but with two or more
        # simultaneous threats, correcting for a later neighbor can reintroduce
        # a violation of an earlier one's constraint, since it's never
        # re-checked. Confirmed this mattered at NUM_AGENTS=4 (each agent has
        # 3 simultaneous "others" vs NUM_AGENTS=3's 2): training-time
        # collision_rate stayed nonzero and scattered across an entire 3M-step
        # run instead of converging, unlike every NUM_AGENTS=3 run (see
        # EXPERIMENT_LOG.md). Repeating the sweep until no neighbor needs
        # further correction is POCS (projection onto an intersection of
        # convex sets), which converges in the limit whenever the
        # intersection is nonempty -- always true here physically (v=0
        # satisfies every constraint, since max_closing is never negative).
        #
        # CAUTION (added 2026-08-20 after supervisor review): "converges in
        # the limit" is not the same claim as "NUM_AGENTS passes reaches
        # exact/zero-residual convergence" -- that stronger claim is not
        # proven here, only checked. NUM_AGENTS is used purely as a cheap
        # upper bound on iterations (this project supports at most 4 agents,
        # so at most 3 simultaneous constraints per agent); the early-exit
        # below is what actually terminates it in practice, and if the exit
        # condition never fires the loop just stops at the cap with whatever
        # residual violation remains, silently. Adversarially stress-tested
        # (2026-08-20): 2 agents head-on and 4 agents simultaneously
        # converging on a shared centroid, both starting outside
        # SAFE_DIST_ENTER and always commanding MAX_ACTION_SPEED toward each
        # other/the centroid every step -- minimum distance asymptotically
        # approached but never crossed COLLISION_DIST, zero residual
        # constraint violation after _apply_brake on every step of both
        # tests. Not a proof, and specific to this system: v_closing below is
        # each agent's OWN velocity component toward the other, not the
        # relative closing velocity (v_a - v_b) -- the two adversarial tests
        # hold because every agent goes through this identical symmetric
        # formula, with no way for one side to be unconstrained while the
        # other closes at speed. That symmetry assumption stops holding once
        # a real vehicle (whose actual velocity this code doesn't control)
        # is involved -- see deployment/docs/PHASE2_HANDOFF.md. Reformulating
        # this with explicit relative velocity is queued, not yet done.
        # Per-axis action scaling (Phase 3, 2026-08-20): was a single
        # isotropic MAX_ACTION_SPEED across all 3 components -- real drones
        # have different vertical (climb) vs horizontal (lateral) speed
        # authority, so Z gets its own, separate cap. REACTION_DIST/
        # SAFE_DIST_ENTER/SAFE_DIST_EXIT (config.py) still derive from the
        # larger MAX_ACTION_SPEED, which stays a valid (if now slightly
        # conservative on Z) worst-case bound.
        action_scale = np.array([MAX_ACTION_SPEED, MAX_ACTION_SPEED, MAX_ACTION_SPEED_Z], np.float32)
        raw_vel = {a: (np.clip(actions[a], -1.0, 1.0) * action_scale).astype(np.float32)
                   for a in self.agents}

        # Z-velocity smoothing (2026-08-20, user-reported jiggling --
        # confirmed real, see config.py's Z_SMOOTHING_ALPHA comment).
        # Deterministic, same "don't just hope the policy learns it" idea
        # as the brake/ground clamp -- blends the freshly-commanded Z
        # velocity with last step's ACTUAL Z velocity (self.vel, not yet
        # overwritten at this point in step()) so the vertical component
        # can't instantly reverse direction. Applied before the brake/
        # ground clamp so those safety-critical corrections still always
        # get the final say -- smoothing is a comfort preference, not a
        # safety mechanism, and must never be able to block one.
        for a in self.agents:
            raw_vel[a][2] = (Z_SMOOTHING_ALPHA * raw_vel[a][2]
                              + (1.0 - Z_SMOOTHING_ALPHA) * self.vel[a][2])

        raw_vel, brake_reduction = self._apply_brake(raw_vel)
        raw_vel, ground_reduction = self._apply_ground_clamp(raw_vel)

        for a in self.agents:
            self.vel[a] = raw_vel[a]
            self.pos[a] = (self.pos[a] + self.vel[a] * DT).astype(np.float32)

        # Dynamic target motion (Phase 3, 2026-08-20): closes KNOWN_ISSUES.md
        # item 9 -- _target_dir/_target_speed used to be sampled once in
        # reset() and never updated, so the dead-reckoning grace period
        # (_update_target_track) was mathematically exact whenever used,
        # not a real approximation of an uncertain estimate. Periodic
        # redirects (same distribution reset() uses) make a mid-episode
        # direction change able to genuinely mislead a drone's dead-reckoned
        # estimate while it's out of contact -- the realistic case the grace
        # period was always meant to be tested against.
        if self.step_count % TARGET_REDIRECT_INTERVAL_STEPS == 0:
            self._resample_target_motion()
        self.pos_t = (self.pos_t + self._target_dir * self._target_speed * DT).astype(np.float32)
        self._update_target_track()

        needs_relock = any(
            max((np.linalg.norm(self.pos[a] - self.pos[n]) for n in self.locked[a]), default=0.0)
            > DIVERGE_DIST
            for a in self.agents
        )
        if needs_relock:
            self._relock_all()

        self._current_diameter = self.get_swarm_stats()["swarm_diameter"]

        # Collision termination stays global -- a collision ends the episode
        # for the whole scene, that's a fact about it, not a credit question.
        # But which agents get *penalized* for it is a separate question:
        # colliding_agents tracks only the agents actually within
        # COLLISION_DIST of someone, so an uninvolved agent (e.g. a 3rd
        # drone nowhere near a 2-drone collision) doesn't get the same -300
        # penalty as the agents that actually caused it. The old global-to-
        # everyone version was itself a fix for a worse bug (locked-neighbor-
        # only attribution let some actually-involved agents get zero
        # penalty), but overcorrected: literature on MARL collision avoidance
        # documents this exact failure mode -- shared penalties inject noisy,
        # unrelated gradient into every agent's credit assignment, and it
        # gets worse as agent count grows (more uninvolved bystanders per
        # collision event), which matches collision_rate getting measurably
        # worse from NUM_AGENTS=2 to 3 despite nothing else changing.
        colliding_agents = set()
        for a, b in itertools.combinations(self.agents, 2):
            if np.linalg.norm(self.pos[a] - self.pos[b]) < COLLISION_DIST:
                colliding_agents.add(a)
                colliding_agents.add(b)
        collision = len(colliding_agents) > 0
        truncated = self.step_count >= MAX_STEPS

        # ground_strike (Phase 3, 2026-08-20): same global-termination,
        # targeted-penalty pattern as collision above -- the episode ends
        # for everyone (the formation task is compromised the moment a
        # member is down), but only the agent(s) that actually struck the
        # ground get the terminal penalty, not innocent bystanders.
        ground_struck_agents = {a for a in self.agents if float(self.pos[a][2]) <= GROUND_Z}
        ground_strike = len(ground_struck_agents) > 0

        reward_tuples = {a: self._get_reward(a, a in colliding_agents, brake_reduction[a],
                                               a in ground_struck_agents) for a in self.agents}
        rewards = {a: reward_tuples[a][0] for a in self.agents}
        reward_components = {a: reward_tuples[a][1] for a in self.agents}

        # target_lost/ground_strike (self._update_target_track() above /
        # ground_struck_agents above) are further real terminal conditions
        # alongside collision -- a swarm with no trusted target estimate,
        # or missing a member, has nothing to learn from by continuing,
        # same reasoning as collision termination. Each kept as its own
        # info flag (not folded into "collided") so callers can log/
        # attribute the distinct failure modes separately rather than
        # conflating them.
        terminations = {a: bool(collision or self._target_lost or ground_strike) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"min_dist": self._min_dist(a),
                       "reward_components": reward_components[a],
                       "brake_reduction": brake_reduction[a],
                       # Diagnostics from _apply_brake, 2026-08-20 -- see its
                       # docstring. brake_passes/brake_violation are the same
                       # value for every agent this step (env-wide, not
                       # per-agent); k_active is per-agent.
                       "brake_passes": self._last_brake_passes,
                       "brake_violation": self._last_brake_violation,
                       "k_active": self._last_brake_k_active[a],
                       "collision": collision,
                       "target_lost": self._target_lost,
                       # Ground clamp diagnostic (Phase 3, 2026-08-20),
                       # mirrors brake_reduction/collision above.
                       "ground_reduction": ground_reduction[a],
                       "ground_strike": ground_strike,
                       # Ground-truth track error, NOT used by the reward or
                       # observation (both use the estimate) -- diagnostic
                       # only, to check how far the estimate-based reward
                       # ever actually diverges from reality.
                       "true_track_err": abs(self._dist_to_target(a) - TARGET_DIST)} for a in self.agents}

        if collision or truncated or self._target_lost or ground_strike:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _min_dist(self, agent):
        others = [o for o in self.possible_agents if o != agent]
        return min(np.linalg.norm(self.pos[agent] - self.pos[o]) for o in others)

    def _dist_to_target(self, agent):
        return float(np.linalg.norm(self.pos[agent] - self.pos_t))

    def get_swarm_stats(self):
        dists = np.array([
            np.linalg.norm(self.pos[a] - self.pos[b])
            for a, b in itertools.combinations(self.possible_agents, 2)
        ])
        return {
            "mean_pairwise": float(dists.mean()),
            "std_pairwise": float(dists.std()),
            "min_pairwise": float(dists.min()),
            "swarm_diameter": float(dists.max()),
        }

    def _get_obs(self, agent):
        p, v = self.pos[agent], self.vel[agent]
        # Sourced from this agent's effective estimate
        # (_update_target_track()), never self.pos_t directly -- see the
        # module docstring. Identical slots/normalization to the old
        # ground-truth version, so this is a drop-in swap from the
        # network's point of view. Equals the swarm-shared tracked
        # estimate during direct contact; becomes this agent's own
        # search waypoint during a lost period (active search,
        # 2026-08-21) -- see _update_target_track/_assign_search_directions.
        target_pos_est = self._effective_pos_est[agent]
        target_vel_est = self._effective_vel_est[agent]
        rel_t = (target_pos_est - p) / OBS_MAX_DIST
        dist_t = float(np.linalg.norm(target_pos_est - p)) / OBS_MAX_DIST
        rel_target_vel = (target_vel_est - v) / OBS_MAX_VEL

        has_contact = 1.0 if self._direct_contact.get(agent, False) else 0.0
        age_norm = min(1.0, self._steps_since_contact / LOST_TIMEOUT_STEPS)
        observer_norm = self._observer_count / len(self.possible_agents)
        if self._contact_centroid is not None:
            rel_centroid = (self._contact_centroid - p) / OBS_MAX_DIST
            centroid_valid = 1.0
        else:
            rel_centroid = np.zeros(3, np.float32)
            centroid_valid = 0.0

        feats = [v[0] / OBS_MAX_VEL, v[1] / OBS_MAX_VEL, v[2] / OBS_MAX_VEL,
                  rel_t[0], rel_t[1], rel_t[2], dist_t,
                  rel_target_vel[0], rel_target_vel[1], rel_target_vel[2],
                  has_contact, self._track_confidence, age_norm, observer_norm,
                  rel_centroid[0], rel_centroid[1], rel_centroid[2], centroid_valid]

        neighbors = self.locked[agent]
        for i in range(self.k):
            if i < len(neighbors):
                n = neighbors[i]
                rel_p = (self.pos[n] - p) / OBS_MAX_DIST
                rel_v = (self.vel[n] - v) / OBS_MAX_VEL
                d = np.linalg.norm(self.pos[n] - p) / OBS_MAX_DIST
                feats.extend([rel_p[0], rel_p[1], rel_p[2],
                              rel_v[0], rel_v[1], rel_v[2], d])
            else:
                feats.extend([0, 0, 0, 0, 0, 0, 1.0])

        return np.clip(np.array(feats, dtype=np.float32), -1.0, 1.0)

    def _get_reward(self, agent, agent_collided, brake_reduction, agent_ground_struck=False):
        """Returns (total_reward, components_dict). agent_collided is True
        only if this specific agent is within COLLISION_DIST of another
        agent, not just whether the episode is ending in collision.
        brake_reduction is this agent's total closing speed the brake removed
        this step (see step()), fed back below as r_brake. agent_ground_struck
        (Phase 3, 2026-08-20) is this agent's own GROUND_Z crossing, same
        per-agent-not-global-blame idea as agent_collided -- fed back below
        as r_ground's terminal component."""
        p, v = self.pos[agent], self.vel[agent]
        neighbors = self.locked[agent]

        # Both computed against this agent's effective ESTIMATE, not ground
        # truth (self.pos_t) -- an agent can only be rewarded for what it
        # (or the swarm, via shared contact) can actually know. Using ground
        # truth here while the observation uses the estimate would be
        # exactly the same class of observation/reward mismatch a supervisor
        # review caught for r_safety earlier (see config.py's K_NEIGHBORS
        # comment) -- not repeating that. Equals the shared tracked estimate
        # during direct contact (numerically ~a no-op there, same as
        # before); during a lost period this is now the agent's own search
        # waypoint (active search, 2026-08-21) -- reusing this exact reward
        # shape to pull a searching agent toward ITS assigned heading is
        # deliberate, not an approximation: no new reward term was added for
        # search, this one just gets pointed at a different target.
        target_pos_est = self._effective_pos_est[agent]
        target_vel_est = self._effective_vel_est[agent]
        track_err = abs(float(np.linalg.norm(p - target_pos_est)) - TARGET_DIST)
        r_track = TRACK_WEIGHT * track_err

        r_velocity = VELOCITY_WEIGHT * float(np.linalg.norm(v - target_vel_est))

        # True 3D angular separation (was horizontal/XY-bearing-only) -- see
        # _IDEAL_NEIGHBOR_ANGLE above. For each pair of locked neighbors,
        # the angle between their direction vectors from this drone; penalize
        # the minimum pairwise angle's deviation from the regular-simplex
        # local ideal (60 deg, same for every K under full connectivity --
        # see _IDEAL_NEIGHBOR_ANGLE), same "penalize how far the tightest
        # gap is from even spacing" shape the old 2D version used, just
        # measured on the sphere instead of the horizontal plane.
        r_spread = 0.0
        if len(neighbors) >= 2:
            dirs = []
            for n in neighbors:
                d = self.pos[n] - p
                norm = np.linalg.norm(d)
                if norm > 1e-6:
                    dirs.append(d / norm)
            if len(dirs) >= 2:
                pairwise_angles = [
                    math.acos(float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)))
                    for i in range(len(dirs)) for j in range(i + 1, len(dirs))
                ]
                min_angle = min(pairwise_angles)
                r_spread = -0.3 * abs(min_angle - _IDEAL_NEIGHBOR_ANGLE)

        # Safety is checked against ALL other agents, not just locked
        # neighbors -- collision termination is global (any pair, anywhere),
        # so a drone closing on an unlocked neighbor previously got zero
        # warning gradient until the episode ended under it.
        #
        # Bonus zone reshaped (2026-08-14, supervisor review): the old version
        # peaked at (SAFE_DIST_ENTER+SAFE_DIST_EXIT)/2 = 6.0 and hit exactly 0
        # at SAFE_DIST_EXIT = 6.60 -- below the actual designed-safe formation
        # edge (EDGE_TARGET = 7.80, one full REACTION_DIST beyond
        # SAFE_DIST_EXIT by design, see config.py). That meant the only
        # positive per-pair reward anywhere on this axis sat in a tighter,
        # riskier band than the formation was ever meant to converge to, with
        # nothing rewarding the actual intended equilibrium -- confirmed as
        # the likely driver of the NUM_AGENTS=3 collision-rate relapse
        # (avg_min_dist trended toward ~6-7, i.e. into the old bonus band,
        # not the designed-safe 7.80, as entropy collapsed and the policy got
        # precise enough to reliably hold that tighter spacing). Now a single
        # triangular ramp: 0 at SAFE_DIST_ENTER -> SAFETY_MAX_BONUS at
        # EDGE_TARGET (the actual ideal edge) -> back to 0 one more
        # REACTION_DIST beyond it, so "arbitrarily far" still reads as
        # neutral rather than staying rewarded forever. Also incidentally
        # fixes two discontinuities the old shape had: 0 from the urgent
        # branch vs. 1.0 from the bonus branch right at SAFE_DIST_ENTER, and
        # a hard cliff to 0 at SAFE_DIST_EXIT. Unverified against a completed
        # run as of this comment -- next single-seed N=3 run is the check.
        r_safety = 0.0
        all_clear = True
        for n in self.agents:
            if n == agent:
                continue
            d = float(np.linalg.norm(self.pos[n] - p))
            if d < SAFE_DIST_ENTER:
                all_clear = False
                # Ramps linearly from 0 at SAFE_DIST_ENTER to full SAFETY_URGENT_COEF
                # exactly at COLLISION_DIST, so the policy gets a real warning
                # gradient before termination instead of a flat bonus up to the cliff.
                span = SAFE_DIST_ENTER - COLLISION_DIST
                r_safety += SAFETY_URGENT_COEF * min(1.0, (SAFE_DIST_ENTER - d) / span)
            elif d < EDGE_TARGET:
                r_safety += SAFETY_MAX_BONUS * (d - SAFE_DIST_ENTER) / (EDGE_TARGET - SAFE_DIST_ENTER)
            elif d < EDGE_TARGET + REACTION_DIST:
                r_safety += SAFETY_MAX_BONUS * (1.0 - (d - EDGE_TARGET) / REACTION_DIST)

        # GLOBAL cohesion penalty -- based on swarm_diameter, shared by all agents.
        # Cannot be evaded by relocking onto a nearer neighbor (unlike the old
        # per-locked-neighbor diverge penalty).
        r_cohesion = COHESION_WEIGHT * max(0.0, self._current_diameter - COHESION_LIMIT)
        # Diameter floor (2026-08-14): symmetric counter-pressure against the
        # swarm getting too TIGHT, not just too loose -- see MIN_DIAMETER's
        # comment in config.py for why. Additive with the excess-diameter term
        # above; the two can never both be active at once (MIN_DIAMETER <
        # COHESION_LIMIT).
        r_cohesion += DIAMETER_FLOOR_WEIGHT * max(0.0, MIN_DIAMETER - self._current_diameter)

        r_collision_global = -300.0 if agent_collided else 0.0

        # Contact-loss urgency (2026-08-14, vision-based tracking): ramps 0
        # (swarm currently has direct contact) -> CONTACT_URGENT_COEF (right
        # at LOST_TIMEOUT_STEPS, where target_lost termination fires next),
        # same shape as r_safety's urgent ramp. Identical for every agent --
        # this is about swarm-wide contact, not individual blame, which is
        # what makes "someone else regaining contact helps everyone's
        # reward" a genuinely cooperative signal instead of a per-agent one.
        # Combined with TARGET_LOST_PENALTY on the terminating step itself,
        # mirroring the r_safety-ramp + r_collision-terminal two-tier pattern
        # already validated in this file.
        r_contact = CONTACT_URGENT_COEF * min(1.0, self._steps_since_contact / LOST_TIMEOUT_STEPS)
        if self._target_lost:
            r_contact += TARGET_LOST_PENALTY

        # Ground urgency + terminal penalty (Phase 3, 2026-08-20) -- same
        # two-tier pattern as collision (r_safety ramp + r_collision_global
        # terminal) and target_lost (r_contact ramp + TARGET_LOST_PENALTY
        # terminal): graduated warning as altitude approaches GROUND_Z, plus
        # a large one-time penalty on the step this specific agent actually
        # strikes it. See config.py's GROUND_URGENT_COEF/GROUND_STRIKE_PENALTY
        # comment for why the terminal magnitude matches collision, not the
        # softer target_lost.
        r_ground = 0.0
        z = float(p[2])
        if z < GROUND_SAFE_ENTER:
            span = GROUND_SAFE_ENTER - GROUND_Z
            r_ground = GROUND_URGENT_COEF * min(1.0, (GROUND_SAFE_ENTER - z) / span)
        if agent_ground_struck:
            r_ground += GROUND_STRIKE_PENALTY

        # Cruise-altitude preference (2026-08-20) -- separate from r_ground
        # above: that's crash-avoidance (derived from REACTION_DIST, a hard
        # physical boundary), this is a softer "don't loiter uncomfortably
        # low" preference, graduated the same way but against CRUISE_ALT_MIN
        # instead of GROUND_SAFE_ENTER. No bonus for flying high, only a
        # penalty for dipping below the comfort floor.
        r_altitude = 0.0
        if z < CRUISE_ALT_MIN:
            r_altitude = CRUISE_ALT_COEF * (CRUISE_ALT_MIN - z) / CRUISE_ALT_MIN

        # Joint bonus: an ADDITIVE reward composition lets the policy bank
        # "good enough" reward from tracking OR safety alone, which is
        # exactly the two-mode failure eval.py measured -- tight formation
        # that tracks well but collides fast, or spread formation that's
        # safe but tracks poorly. Neither mode needed to find the region
        # where both are true, because the sum rewards each dimension
        # independently. This adds an explicit bonus only when both are
        # true at once, so that joint region has a reward advantage instead
        # of just being hoped for. Existing weights are untouched so this is
        # an isolated addition, not a re-balance of what's already there.
        r_joint = JOINT_BONUS if (track_err < JOINT_TRACK_TOL and all_clear) else 0.0

        # Direct brake-engagement penalty (2026-08-19) -- see config.py's
        # BRAKE_PENALTY_COEF comment for why: r_safety's urgent-zone penalty
        # above is proximity-based, not action-based, so it doesn't
        # specifically penalize *needing* the brake to intervene, only being
        # close. This closes that gap with a direct, precise signal on the
        # actual over-commitment. v2: only the excess above
        # BRAKE_PENALTY_THRESHOLD is penalized -- a linear-from-zero v1
        # measurably fixed collision_rate but also taxed trivial, routine
        # engagement, not just aggressive corrections, which pushed the
        # policy to avoid the brake's trigger zone altogether (wider
        # formation, worse tracking) rather than just avoid needing a large
        # correction once inside it. See config.py for the full before/after
        # data and how the threshold was derived.
        r_brake = BRAKE_PENALTY_COEF * max(0.0, brake_reduction - BRAKE_PENALTY_THRESHOLD)

        total = float(r_track + r_spread + r_safety + r_cohesion + r_collision_global
                      + r_velocity + r_joint + r_contact + r_brake + r_ground + r_altitude)
        components = {
            "track": float(r_track), "spread": float(r_spread),
            "safety": float(r_safety), "cohesion": float(r_cohesion),
            "collision": float(r_collision_global), "velocity": float(r_velocity),
            "joint": float(r_joint), "contact": float(r_contact),
            "brake": float(r_brake), "ground": float(r_ground),
            "altitude": float(r_altitude),
        }
        return total, components
