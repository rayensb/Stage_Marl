import torch
import torch.nn as nn

class Actor(nn.Module):
    """Per-agent policy: local obs -> tanh-squashed Gaussian action."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)          # RAW mean, unbounded
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std)
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
        """Recompute log-prob for a STORED (already tanh-squashed) action."""
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        a_clamped = torch.clamp(action, -0.999999, 0.999999)
        u = torch.atanh(a_clamped)
        logp_u = dist.log_prob(u).sum(-1)
        logp = logp_u - torch.log(1 - a_clamped.pow(2) + 1e-6).sum(-1)
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
