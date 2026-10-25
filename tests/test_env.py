"""
Sanity tests for the 3D tunnel environment.

    python tests/test_env.py        (or: pytest tests/)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim3d.config import EnvConfig, N_ACTIONS   # noqa: E402
from sim3d.tunnel_env import TunnelEnv          # noqa: E402


def make(n=64, level=1.0, seed=0):
    env = TunnelEnv(EnvConfig(num_envs=n), device="cpu", seed=seed)
    env.fixed_level = level
    return env


def test_shapes_and_ranges():
    for lvl in (0.0, 1.0, 2.0):
        env = make(level=lvl)
        obs = env.reset()
        assert obs.shape == (64, env.obs_dim)
        assert torch.isfinite(obs).all()
        rays = obs[:, :env.n_rays]
        assert (rays >= 0).all() and (rays <= 1).all()


def test_no_collision_at_start():
    env = make(n=512, level=2.0)
    env.reset()
    _, _, term, _, _ = env.step(torch.zeros(512, dtype=torch.long))
    assert term.sum() == 0


def test_same_seed_same_maps():
    a, b = make(seed=5), make(seed=5)
    a.reset(), b.reset()
    assert torch.equal(a.pipe_p, b.pipe_p) and torch.equal(a.target, b.target)
    c = make(seed=6)
    c.reset()
    assert not torch.equal(a.target, c.target)


def test_gate_geometry():
    env = make(n=256, level=1.5)
    env.reset()
    i = int(env.gate_active.any(1).nonzero()[0])
    j = int(env.gate_active[i].nonzero()[0])
    gx, gc, gr = env.gate_x[i, j].item(), env.gate_c[i, j], env.gate_r[i, j].item()
    d = torch.zeros(env.N, 1, 3)
    d[:, 0, 0] = 1.0
    o = torch.zeros(env.N, 3)
    o[i] = torch.tensor([gx - 5, gc[0], gc[1]])                  # aim through the hole
    assert env.cast_rays(o, d)[i] > 5.0
    side = gc[0] + gr + 0.5 if gc[0] < 0 else gc[0] - gr - 0.5   # aim at the wall beside it
    o[i] = torch.tensor([gx - 5, side, gc[1]])
    assert abs(env.cast_rays(o, d)[i].item() - (5 - env.cfg.gate_thickness / 2)) < 1e-3
    env.pos[i] = torch.tensor([gx, side, gc[1]])
    assert env._gate_hit(env.pos)[i]
    env.pos[i] = torch.tensor([gx, gc[0], gc[1]])
    assert not env._gate_hit(env.pos)[i]


def test_episodes_end_and_reset():
    env = make(n=64, level=0.0)
    env.reset()
    ended = 0
    for _ in range(400):
        _, r, term, trunc, info = env.step(torch.randint(0, N_ACTIONS, (64,)))
        assert torch.isfinite(r).all()
        if "episodes" in info:
            ended += len(info["episodes"]["return"])
    assert ended > 0
    assert (env.steps < env.max_steps).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
    print("all tests passed")
