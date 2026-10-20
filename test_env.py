"""
Quick test script for the MissileEnv.
Run this to verify the environment works with random actions.

Usage:
    python test_env.py              # with rendering
    python test_env.py --headless   # without rendering (faster)
"""
import sys
import numpy as np
from env.missile_env import MissileEnv


def main():
    headless = "--headless" in sys.argv
    render_mode = None if headless else "human"

    env = MissileEnv(render_mode=render_mode)

    num_episodes = 20 if headless else 5
    results = {"target": 0, "obstacle": 0, "wall": 0, "timeout": 0}

    for ep in range(num_episodes):
        obs, info = env.reset()

        # sanity checks on observation
        assert obs.shape == (8,), f"Bad obs shape: {obs.shape}"
        assert env.observation_space.contains(obs), (
            f"Obs out of bounds: min={obs.min():.3f}, max={obs.max():.3f}"
        )

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            assert obs.shape == (8,), f"Step obs shape wrong: {obs.shape}"

        # figure out why the episode ended
        if terminated and reward > 50:
            results["target"] += 1
            reason = "HIT TARGET"
        elif terminated and reward < -20:
            results["obstacle"] += 1
            reason = "CRASH"
        elif truncated:
            results["timeout"] += 1
            reason = "TIMEOUT"
        else:
            results["wall"] += 1
            reason = "OTHER"

        print(f"Ep {ep+1:3d} | Steps: {steps:4d} | Reward: {total_reward:8.1f} | {reason}")

    env.close()

    print(f"\n{'='*50}")
    print(f"Results over {num_episodes} episodes (random agent):")
    print(f"  Target reached: {results['target']}")
    print(f"  Crashed:        {results['obstacle']}")
    print(f"  Timed out:      {results['timeout']}")
    print(f"  Wall hits:      {results['wall']}")
    print(f"{'='*50}")
    print("ALL TESTS PASSED — Environment is working!")


if __name__ == "__main__":
    main()
