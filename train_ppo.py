"""
Train PPO on the 3D tunnel environment.

Examples:
    python train_ppo.py                          # default settings used for our results (~10 min)
    python train_ppo.py --seed 2
    python train_ppo.py --total-steps 200_000    # quick smoke test
"""
import argparse
import os
import time

import torch

from agents.common import Curriculum, Tracker, pick_device, set_seed
from agents.ppo import PPOAgent, PPOConfig
from sim3d.config import EnvConfig
from sim3d.tunnel_env import TunnelEnv


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--total-steps", type=float, default=3e6)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--promote-threshold", type=float, default=0.75)
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=20, help="iterations between log rows")
    return p.parse_args()


def main():
    args = parse()
    device = pick_device(args.device)
    set_seed(args.seed)
    run = args.run_name or f"ppo_s{args.seed}"
    out = os.path.join("results", run)
    os.makedirs("models", exist_ok=True)

    env = TunnelEnv(EnvConfig(num_envs=args.num_envs), device=device, seed=args.seed)
    cfg = PPOConfig(lr=args.lr, rollout_steps=args.rollout_steps, entropy_coef=args.entropy_coef)
    agent = PPOAgent(env.obs_dim, env.n_actions, cfg, device)
    cur = Curriculum(env, threshold=args.promote_threshold,
                     window=max(500, min(4 * args.num_envs, 8000)),
                     enabled=not args.no_curriculum)
    tracker = Tracker(out)

    N, T = args.num_envs, cfg.rollout_steps
    iters = int(args.total_steps // (N * T))
    print(f"[PPO] device={device} envs={N} rollout={T} iterations={iters} -> results/{run}")

    buf = {
        "obs": torch.zeros(T, N, env.obs_dim, device=device),
        "act": torch.zeros(T, N, dtype=torch.long, device=device),
        "logp": torch.zeros(T, N, device=device),
        "val": torch.zeros(T, N, device=device),
        "rew": torch.zeros(T, N, device=device),
        "done": torch.zeros(T, N, device=device),
    }
    obs = env.reset()
    best = -1.0
    step = 0
    t0 = time.time()
    for it in range(1, iters + 1):
        for t in range(T):
            a, logp, v = agent.act(obs)
            nobs, r, term, trunc, info = env.step(a)
            if trunc.any():                       # bootstrap through time-limit truncation
                with torch.no_grad():
                    r = r + cfg.gamma * agent.net.value(info["final_obs"]) * trunc.float()
            buf["obs"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, a, logp, v
            buf["rew"][t], buf["done"][t] = r, (term | trunc).float()
            ep = info.get("episodes")
            tracker.add(ep)
            if cur.update(ep):
                print(f"  >> curriculum level -> {env.level:.1f}")
            obs = nobs
        step += N * T

        with torch.no_grad():
            last_v = agent.net.value(obs)
        stats = agent.update(buf, last_v, 1.0 - (it - 1) / iters)

        if it % args.log_every == 0 or it == iters:
            row = tracker.log(step, env.level, loss=stats["v"],
                              extra=f"ent={stats['ent']:.3f};kl={stats['kl']:.4f}")
            if row:
                sps = step / (time.time() - t0)
                print(f"it {it:5d} | step {step/1e6:7.1f}M | lvl {env.level:.1f} | "
                      f"ret {row['return']:7.2f} | succ {row['success']:.3f} "
                      f"(top {row['success_top']:.3f}) | crash {row['crash']:.3f} | "
                      f"ent {stats['ent']:.2f} | {sps/1e3:.0f}k sps")
                score = env.level + row["success_top"]
                if score == score and score > best:          # (nan check)
                    best = score
                    agent.save(f"models/{run}_best.pth", {"level": env.level})
        if it % 200 == 0:
            agent.save(f"models/{run}_last.pth", {"level": env.level})

    agent.save(f"models/{run}_last.pth", {"level": env.level})
    tracker.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; models/{run}_best.pth")


if __name__ == "__main__":
    main()
