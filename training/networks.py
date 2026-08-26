import torch
import torch.nn as nn

LOG_STD_MIN = -2.0   # std ~= 0.135 (near-deterministic floor)
LOG_STD_MAX = 0.5    # std ~= 1.65 (exploration ceiling)

class Actor(nn.Module):
    """Per-agent policy: local obs -> tanh-squashed Gaussian action."""
    # hidden 64->128 (Phase 3, 2026-08-20): bundled into the sustained-flight
    # resilience test alongside the longer horizon/ground/per-axis changes --
    # a starting guess that more capacity helps the policy hold formation
    # over a much longer episode than it saw at NUM_AGENTS=4's already-good
    # 64-wide results at 200 steps, not a measured need. Env-var overridable
    # since 2026-08-23 (config.py's ACTOR_HIDDEN) specifically to test that
    # "not measured" gap -- see EXPERIMENT_LOG.md for the sweep.
    def __init__(self, obs_dim, act_dim, hidden=128):
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

    def get_log_std(self):
        """log_std is a single obs-independent nn.Parameter (not a function of
        state), so this is cheap to call anytime for logging -- added
        (2026-08-14) to distinguish two different possible causes of the
        `entropy` metric declining over training: log_std actually still
        shrinking (exploration noise itself dropping, until it hits
        LOG_STD_MIN) vs. log_std already pinned at its floor while the mean
        action increasingly saturates near tanh's +-1 boundary (which also
        drives -logp more negative via the squashing-Jacobian term, with no
        further change in log_std at all). Those need different fixes."""
        return torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

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
        # mean_dir (2026-08-26, search-direction auxiliary objective): the
        # policy's current mean action direction, gradient-connected to
        # THIS forward pass -- unlike the buffer's stored `action` (sampled
        # during rollout collection under old parameters, detached from the
        # actor being updated now), this is what an auxiliary loss on
        # "which direction does the policy currently intend" must use.
        mean_dir = torch.tanh(mu)
        return logp, entropy, mean_dir


class CentralCritic(nn.Module):
    """Sees joint obs, outputs one value head per agent -- agents have
    different individual rewards, so a single shared scalar output would be
    trained against N conflicting regression targets per update."""
    def __init__(self, joint_obs_dim, num_agents, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.heads = nn.Linear(hidden, num_agents)

    def forward(self, joint_obs):
        return self.heads(self.trunk(joint_obs))  # (batch, num_agents)
