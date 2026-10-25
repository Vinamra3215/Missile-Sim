"""
Test script for the stage-based MissileEnv.
Usage:
    python test_env.py              # with Pygame window
    python test_env.py --headless   # no rendering
"""
import sys
from env.missile_env import MissileEnv


def main():
    headless = "--headless" in sys.argv
    render_mode = None if headless else "human"

    env = MissileEnv(render_mode=render_mode)

    num_episodes = 20 if headless else 3
    results = {"boss": 0, "crash": 0, "timeout": 0}

    for ep in range(num_episodes):
        obs, info = env.reset()
        assert obs.shape == (10,), f"Bad obs shape: {obs.shape}"

        done = False
        total_reward = 0.0
        steps = 0
        max_progress = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            max_progress = max(max_progress, info.get("progress", 0))

        if terminated and reward > 100:
            results["boss"] += 1
            reason = "HIT BOSS!"
        elif terminated:
            results["crash"] += 1
            reason = f"CRASH ({info.get('zone', '?')})"
        else:
            results["timeout"] += 1
            reason = "TIMEOUT"

        print(f"Ep {ep+1:3d} | Steps: {steps:4d} | Reward: {total_reward:8.1f} "
              f"| Progress: {max_progress*100:4.0f}% | {reason}")

    env.close()

    print(f"\n{'='*60}")
    print(f"Results over {num_episodes} episodes (random agent):")
    print(f"  Boss reached: {results['boss']}")
    print(f"  Crashed:      {results['crash']}")
    print(f"  Timed out:    {results['timeout']}")
    print(f"{'='*60}")
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
