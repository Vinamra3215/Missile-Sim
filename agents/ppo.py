"""
PPO (Proximal Policy Optimization) for batched environments.

One iteration:
  1. Roll out `rollout_steps` steps in all N parallel envs with the current policy.
  2. Compute advantages with GAE(lambda).
  3. Optimise the clipped surrogate objective for `epochs` epochs over minibatches.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from agents.networks import ActorCritic


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatches: int = 4
    rollout_steps: int = 32
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    anneal_lr: bool = True


class PPOAgent:
    def __init__(self, obs_dim, n_actions, cfg: PPOConfig, device):
        self.cfg, self.device = cfg, device
        self.net = ActorCritic(obs_dim, n_actions, cfg.hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def act(self, obs):
        dist = self.net.dist(obs)
        a = dist.sample()
        return a, dist.log_prob(a), self.net.value(obs)

    def update(self, buf, last_value, frac_remaining=1.0):
        """buf: dict of tensors shaped [T, N, ...]. Returns diagnostics."""
        c = self.cfg
        if c.anneal_lr:
            for g in self.opt.param_groups:
                g["lr"] = c.lr * max(frac_remaining, 0.05)

        # ── GAE ──
        T = buf["rew"].shape[0]
        adv = torch.zeros_like(buf["rew"])
        gae = torch.zeros_like(last_value)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else buf["val"][t + 1]
            not_done = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + c.gamma * next_v * not_done - buf["val"][t]
            gae = delta + c.gamma * c.gae_lambda * not_done * gae
            adv[t] = gae
        ret = adv + buf["val"]

        # ── flatten ──
        obs = buf["obs"].flatten(0, 1)
        act = buf["act"].flatten(0, 1)
        old_lp = buf["logp"].flatten(0, 1)
        adv, ret = adv.flatten(), ret.flatten()
        n = obs.shape[0]
        mb = n // c.minibatches

        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        count = 0
        for _ in range(c.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n - mb + 1, mb):
                i = perm[s:s + mb]
                dist = self.net.dist(obs[i])
                lp = dist.log_prob(act[i])
                ratio = torch.exp(lp - old_lp[i])
                a = adv[i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * a).mean()

                v = self.net.value(obs[i])
                v_loss = 0.5 * ((v - ret[i]) ** 2).mean()
                ent = dist.entropy().mean()

                loss = pg + c.value_coef * v_loss - c.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pg"] += pg.item()
                    stats["v"] += v_loss.item()
                    stats["ent"] += ent.item()
                    stats["kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > c.clip_eps).float().mean().item()
                count += 1
        return {k: v / max(count, 1) for k, v in stats.items()}

    def save(self, path, extra=None):
        torch.save({"algo": "ppo", "model": self.net.state_dict(), "cfg": self.cfg.__dict__,
                    **(extra or {})}, path)
