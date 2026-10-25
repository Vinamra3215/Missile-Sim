"""
Evaluate policies on held-out procedurally generated maps.

Every policy is tested on exactly the same maps (same seed) at each difficulty
level, with greedy actions. The evaluation seed is different from all training
seeds, so these maps were never seen during training.

Examples:
    python evaluate.py --policies random steer models/ppo_s1_best.pth models/dqn_s1_best.pth
    python evaluate.py --policies models/ppo_s*_best.pth --levels 0.5 1.0 --episodes 4000
"""
import argparse
import glob
import json
import os

import torch

from agents.common import pick_device
from agents.policy import load_policy
from sim3d.config import EnvConfig
from sim3d.tunnel_env import TunnelEnv

EVAL_SEED = 777_777


@torch.no_grad()
def evaluate(policy, level, episodes, device, seed=EVAL_SEED):
    env = TunnelEnv(EnvConfig(num_envs=episodes), device=device, seed=seed)
    env.fixed_level = level
    obs = env.reset()
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    res = {k: torch.zeros(episodes, device=device) for k in
           ["success", "crash", "timeout", "return", "length"]}
    while not finished.all():
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term | trunc
        if done.any():
            ep = info["episodes"]
            idx = done.nonzero(as_tuple=True)[0]
            new = ~finished[idx]                   # only count each env's first episode
            for k in res:
                res[k][idx[new]] = ep[k][new].float()
            finished[idx] = True
    return {k: v.mean().item() for k, v in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["random", "steer"])
    p.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()
    device = pick_device(args.device)

    names = []
    for s in args.policies:
        names += sorted(glob.glob(s)) if any(ch in s for ch in "*?") else [s]

    probe = TunnelEnv(EnvConfig(num_envs=1), device=device)
    table = {}
    print(f"{'policy':38s} " + " ".join(f"lvl {l:.1f}".rjust(10) for l in args.levels))
    for name in names:
        pol = load_policy(name, probe.obs_dim, probe.n_actions, device)
        row = {}
        for lvl in args.levels:
            row[str(lvl)] = evaluate(pol, lvl, args.episodes, device)
        table[name] = row
        print(f"{os.path.basename(name):38s} " +
              " ".join(f"{row[str(l)]['success']*100:9.1f}%" for l in args.levels))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"levels": args.levels, "episodes": args.episodes, "results": table}, f, indent=2)
    print(f"(success rate on held-out maps)  saved -> {args.out}")


if __name__ == "__main__":
    main()
