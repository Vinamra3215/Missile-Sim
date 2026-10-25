"""Load trained agents (or simple baselines) as greedy policies: obs -> actions."""
import torch

from agents.networks import ActorCritic, QNetwork


def steer_baseline(obs):
    """Hand-coded rule: turn toward the target, ignore obstacles."""
    k = obs.shape[1] - 11                 # target direction (fwd, right, up) starts after the rays
    vr, vu = obs[:, k + 1], obs[:, k + 2]
    a = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    a[vu > 0.05] = 1
    a[vu < -0.05] = 2
    level = vu.abs() <= 0.05
    a[level & (vr > 0.05)] = 4
    a[level & (vr < -0.05)] = 3
    return a


def load_policy(name_or_path, obs_dim, n_actions, device="cpu"):
    if name_or_path == "random":
        return lambda o: torch.randint(0, n_actions, (o.shape[0],), device=o.device)
    if name_or_path == "steer":
        return steer_baseline

    ckpt = torch.load(name_or_path, map_location=device, weights_only=False)
    hidden = ckpt["cfg"].get("hidden", 256)
    if ckpt["algo"] == "ppo":
        net = ActorCritic(obs_dim, n_actions, hidden).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()
        return lambda o: net.act(o, greedy=True)
    if ckpt["algo"] == "dqn":
        net = QNetwork(obs_dim, n_actions, hidden).to(device)
        net.load_state_dict(ckpt["model"])
        net.eval()

        @torch.no_grad()
        def pol(o):
            return net(o).argmax(-1)
        return pol
    raise ValueError(f"unknown algo in {name_or_path}")
