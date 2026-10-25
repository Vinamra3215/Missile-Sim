"""Neural networks shared by the DQN and PPO agents."""
import math
import torch
import torch.nn as nn


def mlp(in_dim, out_dim, hidden=256, layers=2, out_gain=1.0):
    mods, d = [], in_dim
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        mods += [lin, nn.ReLU()]
        d = hidden
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    mods.append(out)
    return nn.Sequential(*mods)


class QNetwork(nn.Module):
    """DQN (dueling): state -> Q-value per action."""

    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.body = mlp(obs_dim, hidden, hidden, layers=1)
        self.value = mlp(hidden, 1, hidden, layers=1)
        self.adv = mlp(hidden, n_actions, hidden, layers=1)

    def forward(self, x):
        h = torch.relu(self.body(x))
        a = self.adv(h)
        return self.value(h) + a - a.mean(-1, keepdim=True)


class ActorCritic(nn.Module):
    """PPO: separate policy (actor) and value (critic) networks."""

    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden, out_gain=0.01)
        self.critic = mlp(obs_dim, 1, hidden, out_gain=1.0)

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Categorical(logits=self.actor(x))

    @torch.no_grad()
    def act(self, x, greedy=False):
        logits = self.actor(x)
        if greedy:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()
