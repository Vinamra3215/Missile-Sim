"""
DQN for batched environments: Double DQN + dueling network + n-step returns,
with a tensor-based replay buffer.

Each env step adds N transitions (one per parallel env) to the buffer, then
`updates_per_step` gradient steps are taken on random minibatches.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from agents.networks import QNetwork


@dataclass
class DQNConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    n_step: int = 3
    batch_size: int = 256
    buffer_size: int = 300_000
    learning_starts: int = 50_000
    updates_per_step: int = 1
    target_tau: float = 0.005           # soft target update per gradient step
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_frac: float = 0.2         # fraction of training to anneal epsilon
    hidden: int = 256
    max_grad_norm: float = 10.0


class ReplayBuffer:
    """Circular buffer of transitions, stored as tensors on the training device."""

    def __init__(self, capacity, obs_dim, device):
        self.cap, self.device = capacity, device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.disc = torch.zeros(capacity, device=device)   # gamma^k * (1 - terminal)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, disc):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.obs[idx], self.act[idx], self.rew[idx] = obs, act, rew
        self.next_obs[idx], self.disc[idx] = next_obs, disc
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.size, (b,), device=self.device)
        return self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.disc[i]

    def __len__(self):
        return self.size


class NStepQueue:
    """Turns per-step transitions of N envs into n-step transitions."""

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.q = []

    def push(self, obs, act, rew, next_obs, term, done):
        """term: true terminal (no bootstrap). done: episode ended (term or truncation)."""
        self.q.append((obs, act, rew, next_obs, term.float(), done.float()))
        if len(self.q) < self.n:
            return None
        obs0, act0 = self.q[0][0], self.q[0][1]
        R = torch.zeros_like(rew)
        disc = torch.ones_like(rew)
        alive = torch.ones_like(rew)            # still inside the same episode
        nxt = self.q[0][3].clone()
        for (_, _, r, no, te, dn) in self.q:
            R += alive * disc * r
            upd = alive.bool()
            nxt[upd] = no[upd]
            disc = torch.where(upd, disc * self.gamma, disc)
            end_term = alive * te
            alive = alive * (1 - dn)
            disc = torch.where(end_term.bool(), torch.zeros_like(disc), disc)
        self.q.pop(0)
        return obs0, act0, R, nxt, disc


class DQNAgent:
    def __init__(self, obs_dim, n_actions, cfg: DQNConfig, device):
        self.cfg, self.device, self.n_actions = cfg, device, n_actions
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.q_target = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim, device)
        self.nstep = NStepQueue(cfg.n_step, cfg.gamma)
        self.eps = cfg.eps_start

    @torch.no_grad()
    def act(self, obs, greedy=False):
        a = self.q(obs).argmax(-1)
        if greedy:
            return a
        rand = torch.rand(obs.shape[0], device=self.device) < self.eps
        return torch.where(rand, torch.randint(0, self.n_actions, a.shape, device=self.device), a)

    def set_epsilon(self, frac_done):
        c = self.cfg
        k = min(frac_done / c.eps_decay_frac, 1.0)
        self.eps = c.eps_start + k * (c.eps_end - c.eps_start)

    def observe(self, obs, act, rew, final_obs, term, done):
        tr = self.nstep.push(obs, act, rew, final_obs, term, done)
        if tr is not None:
            self.buffer.add(*tr)

    def train_step(self):
        c = self.cfg
        obs, act, rew, nxt, disc = self.buffer.sample(c.batch_size)
        q = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.q(nxt).argmax(-1, keepdim=True)          # Double DQN
            q_next = self.q_target(nxt).gather(1, a_star).squeeze(1)
            target = rew + disc * q_next
        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.max_grad_norm)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                tp.mul_(1 - c.target_tau).add_(c.target_tau * p)
        return loss.item()

    def save(self, path, extra=None):
        torch.save({"algo": "dqn", "model": self.q.state_dict(), "cfg": self.cfg.__dict__,
                    **(extra or {})}, path)
