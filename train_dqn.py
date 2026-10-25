"""
Train DQN (baseline) on the 3D tunnel environment.

Examples:
    python train_dqn.py                          # default settings used for our results (~15 min)
    python train_dqn.py --seed 2
    python train_dqn.py --total-steps 200_000    # quick smoke test
"""
import argparse
import os
import time

import torch

from agents.common import Curriculum, Tracker, pick_device, set_seed
from agents.dqn import DQNAgent, DQNConfig
from sim3d.config import EnvConfig
from sim3d.tunnel_env import TunnelEnv


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--total-steps", type=float, default=3e6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--promote-threshold", type=float, default=0.75)
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-every", type=int, default=1000, help="env steps between log rows")
    return p.parse_args()


def main():
    args = parse()
    device = pick_device(args.device)
    set_seed(args.seed)
    run = args.run_name or f"dqn_s{args.seed}"
    out = os.path.join("results", run)
    os.makedirs("models", exist_ok=True)

    env = TunnelEnv(EnvConfig(num_envs=args.num_envs), device=device, seed=args.seed)
    cfg = DQNConfig(lr=args.lr, batch_size=args.batch_size, buffer_size=args.buffer_size,
                    updates_per_step=args.updates_per_step,
                    learning_starts=max(args.batch_size * 4, 10 * args.num_envs))
    agent = DQNAgent(env.obs_dim, env.n_actions, cfg, device)
    cur = Curriculum(env, threshold=args.promote_threshold,
                     window=max(500, min(4 * args.num_envs, 8000)),
                     enabled=not args.no_curriculum)
    tracker = Tracker(out)

    N = args.num_envs
    n_iters = int(args.total_steps // N)
    print(f"[DQN] device={device} envs={N} iterations={n_iters} -> results/{run}")

    obs = env.reset()
    best, losses, step = -1.0, [], 0
    t0 = time.time()
    for it in range(1, n_iters + 1):
        agent.set_epsilon(it / n_iters)
        a = agent.act(obs)
        nobs, r, term, trunc, info = env.step(a)
        agent.observe(obs, a, r, info["final_obs"], term, term | trunc)
        ep = info.get("episodes")
        tracker.add(ep)
        if cur.update(ep):
            print(f"  >> curriculum level -> {env.level:.1f}")
        obs = nobs
        step += N

        if len(agent.buffer) >= cfg.learning_starts:
            for _ in range(cfg.updates_per_step):
                losses.append(agent.train_step())

        if it % args.log_every == 0 or it == n_iters:
            loss = sum(losses) / len(losses) if losses else float("nan")
            losses = []
            row = tracker.log(step, env.level, loss=loss, extra=f"eps={agent.eps:.3f}")
            if row:
                sps = step / (time.time() - t0)
                print(f"it {it:7d} | step {step/1e6:7.1f}M | lvl {env.level:.1f} | "
                      f"ret {row['return']:7.2f} | succ {row['success']:.3f} "
                      f"(top {row['success_top']:.3f}) | eps {agent.eps:.2f} | "
                      f"loss {loss:.3f} | {sps/1e3:.0f}k sps")
                score = env.level + row["success_top"]
                if score == score and score > best:
                    best = score
                    agent.save(f"models/{run}_best.pth", {"level": env.level})
        if it % 20000 == 0:
            agent.save(f"models/{run}_last.pth", {"level": env.level})

    agent.save(f"models/{run}_last.pth", {"level": env.level})
    tracker.close()
    print(f"done in {(time.time()-t0)/60:.1f} min; models/{run}_best.pth")


if __name__ == "__main__":
    main()
