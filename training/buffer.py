import numpy as np
import torch

class RolloutBuffer:
    """Stores transitions for both agents + joint obs for the central critic."""
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
        self.values = np.zeros(n, np.float32)
        self.ptr = 0

    def store(self, obs_dict, joint_obs, act_dict, logp_dict, rew_dict, done, value):
        i = self.ptr
        for a in self.agents:
            self.obs[a][i] = obs_dict[a]
            self.actions[a][i] = act_dict[a]
            self.logp[a][i] = logp_dict[a]
            self.rewards[a][i] = rew_dict[a]
        self.joint_obs[i] = joint_obs
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value, agent):
        adv = np.zeros(self.size, np.float32)
        lastgae = 0.0
        for t in reversed(range(self.size)):
            nextval = last_value if t == self.size - 1 else self.values[t + 1]
            nextnonterminal = 1.0 - self.dones[t]
            delta = self.rewards[agent][t] + self.gamma * nextval * nextnonterminal - self.values[t]
            lastgae = delta + self.gamma * self.lam * nextnonterminal * lastgae
            adv[t] = lastgae
        returns = adv + self.values
        return adv, returns

    def get_tensors_normalized(self, agent, last_value):
        adv, ret = self.compute_gae(last_value, agent)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret_norm = (ret - ret.mean()) / (ret.std() + 1e-8)
        return (
            torch.as_tensor(self.obs[agent]),
            torch.as_tensor(self.joint_obs),
            torch.as_tensor(self.actions[agent]),
            torch.as_tensor(self.logp[agent]),
            torch.as_tensor(adv),
            torch.as_tensor(ret_norm),
        )
