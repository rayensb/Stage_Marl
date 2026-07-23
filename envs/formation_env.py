"""
FormationEnv3D — N-agent PettingZoo ParallelEnv, 3D space.

Each drone locks onto its K nearest neighbors (RECON/LOCKED state machine)
and controls (vx, vy, vz) to: track a moving target, maintain angular spread
among its locked neighbors (emergent triangle/pyramid), avoid collision with
locked neighbors, and avoid drifting too far from them.
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
)


class FormationEnv3D(ParallelEnv):
    metadata = {"name": "formation_env_3d_v0"}

    def __init__(self, num_agents=NUM_AGENTS, k_neighbors=K_NEIGHBORS, scenario=None):
        self.possible_agents = [f"drone{i+1}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.k = min(k_neighbors, num_agents - 1)
        self.forced_scenario = scenario

        self._obs_dim = 7 + 7 * self.k   # own_vel(3)+rel_target(3)+dist_t(1) + k*(relpos3+relvel3+dist1)
        self._act_dim = 3                 # vx, vy, vz

        self.pos, self.vel = {}, {}
        self.locked = {}   # agent -> list of neighbor ids
        self.pos_t = np.zeros(3, np.float32)
        self.step_count = 0
        self.np_random = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32)

    # ---- reset ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0

        self.pos_t = self.np_random.uniform(-3.0, 3.0, 3).astype(np.float32)
        self.pos_t[2] = self.np_random.uniform(2.0, 3.0)  # altitude

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
        self._target_dir[2] *= 0.2  # gentle vertical drift
        self._target_dir /= (np.linalg.norm(self._target_dir) + 1e-6)
        self._target_speed = self.np_random.uniform(0.3, 1.0)

        for a in self.agents:
            self.locked[a] = self._select_neighbors(a)

        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _resolve_overlaps(self):
        max_iter = 50
        for _ in range(max_iter):
            worst = None
            worst_d = COLLISION_DIST + 0.3
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

    def _select_neighbors(self, agent):
        dists = [(o, np.linalg.norm(self.pos[agent] - self.pos[o]))
                  for o in self.agents if o != agent]
        dists.sort(key=lambda x: x[1])
        return [o for o, _ in dists[:self.k]]

    # ---- step ----
    def step(self, actions):
        self.step_count += 1

        for a in self.agents:
            act = np.clip(actions[a], -1.0, 1.0) * MAX_ACTION_SPEED
            self.vel[a] = act.astype(np.float32)
            self.pos[a] = (self.pos[a] + self.vel[a] * DT).astype(np.float32)

        self.pos_t = (self.pos_t + self._target_dir * self._target_speed * DT).astype(np.float32)

        # re-lock (RECON) if a locked neighbor drifted too far
        for a in self.agents:
            max_locked_d = max(
                (np.linalg.norm(self.pos[a] - self.pos[n]) for n in self.locked[a]),
                default=0.0)
            if max_locked_d > DIVERGE_DIST:
                self.locked[a] = self._select_neighbors(a)

        collision = any(
            np.linalg.norm(self.pos[a] - self.pos[b]) < COLLISION_DIST
            for a, b in itertools.combinations(self.agents, 2))
        truncated = self.step_count >= MAX_STEPS

        rewards = {a: self._get_reward(a) for a in self.agents}
        terminations = {a: bool(collision) for a in self.agents}
        truncations = {a: bool(truncated) for a in self.agents}
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"min_dist": self._min_dist(a)} for a in self.agents}

        if collision or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ---- helpers ----
    def _min_dist(self, agent):
        others = [o for o in self.possible_agents if o != agent]
        return min(np.linalg.norm(self.pos[agent] - self.pos[o]) for o in others)

    def _dist_to_target(self, agent):
        return float(np.linalg.norm(self.pos[agent] - self.pos_t))

    def _get_obs(self, agent):
        p, v, t = self.pos[agent], self.vel[agent], self.pos_t
        rel_t = (t - p) / OBS_MAX_DIST
        dist_t = self._dist_to_target(agent) / OBS_MAX_DIST

        feats = [v[0] / OBS_MAX_VEL, v[1] / OBS_MAX_VEL, v[2] / OBS_MAX_VEL,
                  rel_t[0], rel_t[1], rel_t[2], dist_t]

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
                feats.extend([0, 0, 0, 0, 0, 0, 1.0])  # sentinel: no neighbor

        return np.clip(np.array(feats, dtype=np.float32), -1.0, 1.0)

    def _get_reward(self, agent):
        p = self.pos[agent]
        neighbors = self.locked[agent]

        r_track = -0.5 * abs(self._dist_to_target(agent) - TARGET_DIST)

        # angular spread among locked neighbors (ideal = evenly spaced, 2*pi/k apart)
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

        # local collision safety vs locked neighbors only
        r_safety = 0.0
        r_diverge = 0.0
        for n in neighbors:
            d = float(np.linalg.norm(self.pos[n] - p))
            if d < COLLISION_DIST:
                r_safety += -100.0
            elif d < SAFE_DIST_ENTER:
                r_safety += -50.0 * (SAFE_DIST_ENTER - d) / SAFE_DIST_ENTER
            elif d < SAFE_DIST_EXIT:
                center = (SAFE_DIST_ENTER + SAFE_DIST_EXIT) / 2.0
                err = abs(d - center) / (SAFE_DIST_EXIT - SAFE_DIST_ENTER)
                r_safety += 5.0 * (1.0 - err)
            r_diverge += -5.0 * max(0.0, d - DIVERGE_DIST) / DIVERGE_DIST

        return float(r_track + r_spread + r_safety + r_diverge)
