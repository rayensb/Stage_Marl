import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, agents, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.agents = agents
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.joint_obs_dim = obs_dim * len(agents)
        self.reset()

    def reset(self):
        n = self.size
        self.obs = {a: np.zeros((n, self.obs_dim), np.float32) for a in self.agents}
        self.joint_obs = np.zeros((n, self.joint_obs_dim), np.float32)
        self.actions = {a: np.zeros((n, self.act_dim), np.float32) for a in self.agents}
        self.logp = {a: np.zeros(n, np.float32) for a in self.agents}
        self.rewards = {a: np.zeros(n, np.float32) for a in self.agents}
        self.dones = np.zeros(n, np.float32)
        self.values = {a: np.zeros(n, np.float32) for a in self.agents}
        # Search-direction auxiliary objective (2026-08-26): aux_dir is the
        # unit direction the auxiliary loss encourages the actor toward
        # (unit(_last_known_vel), swarm-wide so identical across agents at a
        # given row -- stored per-agent anyway to match every other field's
        # shape/indexing); in_search flags which rows it actually applies to
        # (steps_since_contact > 0). Both are inert (never read) unless
        # AUX_DIR_COEF > 0 in training/train.py.
        self.aux_dir = {a: np.zeros((n, 3), np.float32) for a in self.agents}
        self.in_search = {a: np.zeros(n, np.float32) for a in self.agents}
        # AUX_DIR_RAMP (2026-08-27): how far into the LOST_TIMEOUT_STEPS
        # window this row was taken (0 at loss onset, 1 at the timeout) --
        # see AUX_DIR_RAMP's comment in train.py. Inert (never read) unless
        # AUX_DIR_RAMP > 0.
        self.aux_ramp_frac = {a: np.zeros(n, np.float32) for a in self.agents}
        self.ptr = 0

    def store(self, obs_dict, joint_obs, act_dict, logp_dict, rew_dict, done, value_dict,
              aux_dir_dict, in_search_dict, aux_ramp_frac_dict):
        i = self.ptr
        for a in self.agents:
            self.obs[a][i] = obs_dict[a]
            self.actions[a][i] = act_dict[a]
            self.logp[a][i] = logp_dict[a]
            self.rewards[a][i] = rew_dict[a]
            self.values[a][i] = value_dict[a]
            self.aux_dir[a][i] = aux_dir_dict[a]
            self.in_search[a][i] = in_search_dict[a]
            self.aux_ramp_frac[a][i] = aux_ramp_frac_dict[a]
        self.joint_obs[i] = joint_obs
        self.dones[i] = done
        self.ptr += 1

    def compute_gae(self, last_value, agent):
        values = self.values[agent]
        adv = np.zeros(self.size, np.float32)
        lastgae = 0.0
        for t in reversed(range(self.size)):
            nextval = last_value if t == self.size - 1 else values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[agent][t] + self.gamma * nextval * nextnonterminal - values[t]
            lastgae = delta + self.gamma * self.lam * nextnonterminal * lastgae
            adv[t] = lastgae
        returns = adv + values
        return adv, returns

    def get_tensors(self, agent, last_value):
        """Advantage normalized; returns kept in raw scale (correct for critic target)."""
        adv, ret = self.compute_gae(last_value, agent)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return (
            torch.as_tensor(self.obs[agent]),
            torch.as_tensor(self.joint_obs),
            torch.as_tensor(self.actions[agent]),
            torch.as_tensor(self.logp[agent]),
            torch.as_tensor(adv),
            torch.as_tensor(ret),
            torch.as_tensor(self.aux_dir[agent]),
            torch.as_tensor(self.in_search[agent]),
            torch.as_tensor(self.aux_ramp_frac[agent]),
        )
