"""
FormationEnv3D — N-agent PettingZoo ParallelEnv, 3D space.

Reward-balance fix: track weight increased, safety magnitude reduced,
old per-locked-neighbor "diverge" penalty (had a relock loophole) replaced
with a GLOBAL swarm-cohesion penalty based on swarm_diameter -- can't be
gamed by relocking onto a different neighbor.

NEIGHBOR GRAPH DESIGN CHOICE: mutual (reciprocal) k-nearest-neighbors.
Each drone computes its own top-k nearest candidates; a candidate is only
"locked" if the relationship is mutual (A in B_top_k AND B in A_top_k).
Fully decentralized -- no global consensus step needed.
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
    COLLISION_DIST, DIVERGE_DIST, MAX_STEPS, DT, MAX_ACTION_SPEED,
    OBS_MAX_DIST, OBS_MAX_VEL, OBS_MAX_ANGLE,
    TRACK_WEIGHT, SAFETY_MAX_BONUS, SAFETY_URGENT_COEF,
    COHESION_LIMIT, COHESION_WEIGHT,
)


class FormationEnv3D(ParallelEnv):
    metadata = {"name": "formation_env_3d_v2"}

    def __init__(self, num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS, scenario=None):
        self.possible_agents = [f"drone{i+1}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.k = min(k_neighbors, num_agents - 1)
        self.forced_scenario = scenario

        self._obs_dim = 10 + 7 * self.k
        self._act_dim = 3

        self.pos, self.vel = {}, {}
        self.locked = {}
        self.pos_t = np.zeros(3, np.float32)
        self.step_count = 0
        self._current_diameter = 0.0
        self.np_random = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)

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
            d = self.np_random.uniform(3.5, 6.0)
            offset = np.array([math.cos(ang) * math.cos(elev),
                                math.sin(ang) * math.cos(elev),
                                math.sin(elev)]) * d
            self.pos[a] = (self.pos_t + offset).astype(np.float32)
            self.vel[a] = np.zeros(3, np.float32)

        self._resolve_overlaps()

        self._target_dir = self.np_random.uniform(-1, 1, 3).astype(np.float32)
        self._target_dir[2] *= 0.2
        self._target_dir /= (np.linalg.norm(self._target_dir) + 1e-6)
        self._target_speed = self.np_random.uniform(0.3, 1.0)

        self._relock_all()
        self._current_diameter = self.get_swarm_stats()["swarm_diameter"]

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
        for a in self.agents:
            mutual = [b for b in cand[a] if a in cand[b]]
            if not mutual and cand[a]:
                mutual = cand[a][:1]
            self.locked[a] = mutual[:self.k]

    def step(self, actions):
        self.step_count += 1

        for a in self.agents:
            act = np.clip(actions[a], -1.0, 1.0) * MAX_ACTION_SPEED
            self.vel[a] = act.astype(np.float32)
            self.pos[a] = (self.pos[a] + self.vel[a] * DT).astype(np.float32)

        self.pos_t = (self.pos_t + self._target_dir * self._target_speed * DT).astype(np.float32)

        needs_relock = any(
            max((np.linalg.norm(self.pos[a] - self.pos[n]) for n in self.locked[a]), default=0.0)
            > DIVERGE_DIST
            for a in self.agents
        )
        if needs_relock:
            self._relock_all()

        self._current_diameter = self.get_swarm_stats()["swarm_diameter"]

        collision = any(
            np.linalg.norm(self.pos[a] - self.pos[b]) < COLLISION_DIST
            for a, b in itertools.combinations(self.agents, 2))
        truncated = self.step_count >= MAX_STEPS

        reward_tuples = {a: self._get_reward(a, collision) for a in self.agents}
        rewards = {a: reward_tuples[a][0] for a in self.agents}
        reward_components = {a: reward_tuples[a][1] for a in self.agents}

        terminations = {a: bool(collision) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"min_dist": self._min_dist(a),
                       "reward_components": reward_components[a]} for a in self.agents}

        if collision or truncated:
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
        p, v, t = self.pos[agent], self.vel[agent], self.pos_t
        rel_t = (t - p) / OBS_MAX_DIST
        dist_t = self._dist_to_target(agent) / OBS_MAX_DIST
        target_vel = self._target_dir * self._target_speed
        rel_target_vel = (target_vel - v) / OBS_MAX_VEL

        feats = [v[0] / OBS_MAX_VEL, v[1] / OBS_MAX_VEL, v[2] / OBS_MAX_VEL,
                  rel_t[0], rel_t[1], rel_t[2], dist_t,
                  rel_target_vel[0], rel_target_vel[1], rel_target_vel[2]]

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

    def _get_reward(self, agent, global_collision):
        """Returns (total_reward, components_dict)."""
        p, v = self.pos[agent], self.vel[agent]
        neighbors = self.locked[agent]

        r_track = TRACK_WEIGHT * abs(self._dist_to_target(agent) - TARGET_DIST)

        target_vel = self._target_dir * self._target_speed
        r_velocity = -0.05 * float(np.linalg.norm(v - target_vel))

        r_spread = 0.0
        if len(neighbors) >= 2:
            bearings = []
            for n in neighbors:
                d = self.pos[n] - p
                bearings.append(math.atan2(d[1], d[0]))
            bearings.sort()
            gaps = [(bearings[(i + 1) % len(bearings)] - bearings[i]) % (2 * math.pi)
                    for i in range(len(bearings))]
            min_gap = min(gaps)
            ideal_gap = 2 * math.pi / len(neighbors)
            r_spread = -0.3 * abs(min_gap - ideal_gap)

        r_safety = 0.0
        for n in neighbors:
            d = float(np.linalg.norm(self.pos[n] - p))
            if d < SAFE_DIST_ENTER:
                # Ramps linearly from 0 at SAFE_DIST_ENTER to full SAFETY_URGENT_COEF
                # exactly at COLLISION_DIST, so the policy gets a real warning
                # gradient before termination instead of a flat bonus up to the cliff.
                span = SAFE_DIST_ENTER - COLLISION_DIST
                r_safety += SAFETY_URGENT_COEF * min(1.0, (SAFE_DIST_ENTER - d) / span)
            elif d < SAFE_DIST_EXIT:
                center = (SAFE_DIST_ENTER + SAFE_DIST_EXIT) / 2.0
                err = abs(d - center) / (SAFE_DIST_EXIT - SAFE_DIST_ENTER)
                r_safety += SAFETY_MAX_BONUS * (1.0 - err)

        # GLOBAL cohesion penalty -- based on swarm_diameter, shared by all agents.
        # Cannot be evaded by relocking onto a nearer neighbor (unlike the old
        # per-locked-neighbor diverge penalty).
        r_cohesion = COHESION_WEIGHT * max(0.0, self._current_diameter - COHESION_LIMIT)

        r_collision_global = -300.0 if global_collision else 0.0

        total = float(r_track + r_spread + r_safety + r_cohesion + r_collision_global + r_velocity)
        components = {
            "track": float(r_track), "spread": float(r_spread),
            "safety": float(r_safety), "cohesion": float(r_cohesion),
            "collision": float(r_collision_global), "velocity": float(r_velocity),
        }
        return total, components
