import torch
import torch.nn as nn

class Actor(nn.Module):
    """Per-agent policy: local obs -> action mean (Gaussian)."""
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
        mu = torch.tanh(self.mu(h))
        std = torch.exp(self.log_std)
        return mu, std

    def get_action(self, obs, deterministic=False):
        mu, std = self.forward(obs)
        if deterministic:
            return mu, None
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return torch.clamp(action, -1, 1), logp

    def evaluate(self, obs, action):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy


class CentralCritic(nn.Module):
    """Centralized critic: sees concatenated obs of ALL agents -> value."""
    def __init__(self, joint_obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, joint_obs):
        return self.net(joint_obs).squeeze(-1)
