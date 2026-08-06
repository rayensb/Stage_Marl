import torch
import torch.nn as nn

LOG_STD_MIN = -2.0   # std ~= 0.135 (near-deterministic floor)
LOG_STD_MAX = 0.5    # std ~= 1.65 (exploration ceiling)

class Actor(nn.Module):
    """Per-agent policy: local obs -> tanh-squashed Gaussian action."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def get_action(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        if deterministic:
            return torch.tanh(mu), None
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(-1)
        logp = logp_u - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def evaluate(self, obs, action):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        a_clamped = torch.clamp(action, -0.999999, 0.999999)
        u = torch.atanh(a_clamped)
        logp_u = dist.log_prob(u).sum(-1)
        logp = logp_u - torch.log(1 - a_clamped.pow(2) + 1e-6).sum(-1)
        # dist.entropy() is the entropy of the untransformed base Gaussian,
        # not the actual tanh-squashed action distribution -- it systematically
        # overstates true action-space randomness. -logp is a Monte Carlo
        # estimate of entropy under the real (squashed) distribution.
        entropy = -logp
        return logp, entropy


class CentralCritic(nn.Module):
    """Sees joint obs, outputs one value head per agent -- agents have
    different individual rewards, so a single shared scalar output would be
    trained against N conflicting regression targets per update."""
    def __init__(self, joint_obs_dim, num_agents, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.heads = nn.Linear(hidden, num_agents)

    def forward(self, joint_obs):
        return self.heads(self.trunk(joint_obs))  # (batch, num_agents)
