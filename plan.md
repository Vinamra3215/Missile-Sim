# MissileSim: Complete Project Plan

## Overview
**Topic:** MissileSim — RL for Autonomous Missile Control  
**Algorithms:** DQN (baseline) + PPO (main)  
**Deadline:** Oct 26, 2026, 10:00 AM  
**Team:** 2 persons working in **parallel**  

---

## How Both Work in Parallel (Not Cascaded)

```
         Day 1          Day 2          Day 3          Day 4          Day 5
Person 1 ████████████   ████████████   ██████░░░░░░   ████████████   ██████████
         Physics+Obs    Gym Wrapper    Integration    Rendering+     Final
         +Collision     +Renderer v1   Bug fixes      Demo GIF       Polish

Person 2 ████████████   ████████████   ██████░░░░░░   ████████████   ██████████
         Networks+      DQN+PPO on     Integration    Train on       Plots+
         Buffer+DQN     CartPole test  Switch to env  Real Env+Tune  Cleanup

                                       ▲▲▲▲▲▲▲▲▲▲
                                       MERGE POINT
```

**Key trick for parallelism:** Person 2 develops and tests ALL agents on `CartPole-v1`
(same pattern: discrete actions, continuous obs) while Person 1 builds the real env.
On Day 3 they merge — Person 2 just swaps `CartPole-v1` → `MissileEnv` and retrains.

---

## Shared Interface Contract (Agree on Day 0)

> Both persons MUST agree on this BEFORE starting any code.

```python
# Observation: 8 floats in [-1, 1]
# [0] missile_x (normalized)     [1] missile_y (normalized)
# [2] missile_vx (normalized)    [3] missile_vy (normalized)
# [4] heading_sin                [5] heading_cos
# [6] target_distance (norm)     [7] target_angle (norm)
obs_shape = (8,)

# Actions: Discrete(5)
# 0=coast, 1=turn_left, 2=turn_right, 3=accelerate, 4=decelerate
n_actions = 5

# Gym API:
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(action)
```

---

## Project Structure

```
Missile-Sim/
├── README.md
├── requirements.txt
├── configs/
│   └── default.py              # ALL hyperparameters (shared)
│
├── env/                        # ──── Person 1 ────
│   ├── __init__.py
│   ├── missile_env.py          # Gymnasium environment (main file)
│   ├── missile_physics.py      # Missile movement & physics
│   ├── obstacles.py            # Obstacle generation & collision
│   └── renderer.py             # Pygame visualization
│
├── agents/                     # ──── Person 2 ────
│   ├── __init__.py
│   ├── networks.py             # NN architectures (shared by DQN & PPO)
│   ├── replay_buffer.py        # Experience replay for DQN
│   ├── dqn_agent.py            # DQN agent
│   └── ppo_agent.py            # PPO agent
│
├── train_dqn.py                # Person 2
├── train_ppo.py                # Person 2
├── evaluate.py                 # Person 1
├── plot_results.py             # Person 2
│
├── models/                     # Saved checkpoints
├── results/                    # CSVs + plots
└── demo/                       # GIFs for README
```

---

## Person 1: Environment + Evaluation + README

### P1-Task 1: Missile Physics (`env/missile_physics.py`) — Day 1

```python
class Missile:
    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.speed = 0.0
        self.heading = heading  # radians

    def update(self, action):
        """
        action: 0=coast, 1=left, 2=right, 3=accel, 4=decel
        Updates heading, speed, position.
        """
        # 1. Adjust heading (turn rate per step)
        if action == 1: self.heading += TURN_RATE
        elif action == 2: self.heading -= TURN_RATE

        # 2. Adjust speed
        if action == 3: self.speed = min(self.speed + THRUST, MAX_SPEED)
        elif action == 4: self.speed = max(self.speed - THRUST, 0)

        # 3. Update position
        self.vx = self.speed * math.cos(self.heading)
        self.vy = self.speed * math.sin(self.heading)
        self.x += self.vx
        self.y += self.vy

    def get_state(self):
        return (self.x, self.y, self.vx, self.vy, self.heading, self.speed)
```

Physics rules:
- Heading in radians, wraps around [-π, π]
- Speed clamped to [0, MAX_SPEED]
- No negative speed (missile can only go forward or coast)
- Position updates every step

### P1-Task 2: Obstacles (`env/obstacles.py`) — Day 1

```python
class Obstacle:
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

def generate_obstacles(n, width, height, missile_pos, target_pos, min_gap=60):
    """Generate n non-overlapping obstacles avoiding start & target."""
    obstacles = []
    for _ in range(n * 10):  # max attempts
        cx = random.randint(50, width - 50)
        cy = random.randint(50, height - 50)
        r = random.randint(15, 40)
        # check not near missile start or target
        # check not overlapping existing obstacles
        ...
    return obstacles

def check_collision(mx, my, missile_radius, obstacles):
    """Circle-circle collision check."""
    for obs in obstacles:
        dist = math.sqrt((mx - obs.cx)**2 + (my - obs.cy)**2)
        if dist < missile_radius + obs.radius:
            return True
    return False

def check_wall(mx, my, width, height):
    return mx < 0 or mx > width or my < 0 or my > height
```

### P1-Task 3: Gymnasium Wrapper (`env/missile_env.py`) — Day 2

This is the **most critical file** — the bridge between Person 1 and Person 2.

```python
import gymnasium as gym
import numpy as np
from env.missile_physics import Missile
from env.obstacles import generate_obstacles, check_collision, check_wall
from configs.default import *

class MissileEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.renderer = None  # lazy init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # random missile start (bottom-left quadrant)
        mx = self.np_random.uniform(50, ENV_WIDTH * 0.3)
        my = self.np_random.uniform(ENV_HEIGHT * 0.6, ENV_HEIGHT - 50)
        # random target (top-right quadrant, far from start)
        tx = self.np_random.uniform(ENV_WIDTH * 0.6, ENV_WIDTH - 50)
        ty = self.np_random.uniform(50, ENV_HEIGHT * 0.4)

        self.missile = Missile(mx, my)
        self.target = (tx, ty)
        self.obstacles = generate_obstacles(NUM_OBSTACLES, ENV_WIDTH, ENV_HEIGHT, (mx,my), (tx,ty))
        self.steps = 0
        self.prev_dist = self._dist_to_target()
        self.initial_dist = self.prev_dist

        return self._get_obs(), {}

    def step(self, action):
        self.missile.update(action)
        self.steps += 1

        terminated = False
        truncated = False
        reward = REWARD_TIME_PENALTY  # small negative each step

        curr_dist = self._dist_to_target()

        # distance shaping: reward getting closer
        reward += REWARD_DISTANCE_SHAPING * (self.prev_dist - curr_dist) / self.initial_dist

        # heading bonus: small reward for facing target
        angle_to_target = math.atan2(self.target[1]-self.missile.y, self.target[0]-self.missile.x)
        angle_diff = abs(angle_to_target - self.missile.heading)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        if angle_diff < math.pi/4:
            reward += REWARD_HEADING_BONUS

        # check target reached
        if curr_dist < TARGET_RADIUS:
            reward = REWARD_HIT_TARGET
            terminated = True

        # check obstacle collision
        elif check_collision(self.missile.x, self.missile.y, 5, self.obstacles):
            reward = REWARD_HIT_OBSTACLE
            terminated = True

        # check wall collision
        elif check_wall(self.missile.x, self.missile.y, ENV_WIDTH, ENV_HEIGHT):
            reward = REWARD_HIT_WALL
            terminated = True

        # check truncation
        if self.steps >= MAX_STEPS:
            truncated = True

        self.prev_dist = curr_dist
        return self._get_obs(), reward, terminated, truncated, {"dist": curr_dist}

    def _get_obs(self):
        diag = math.sqrt(ENV_WIDTH**2 + ENV_HEIGHT**2)
        angle_to_t = math.atan2(self.target[1]-self.missile.y, self.target[0]-self.missile.x)
        rel_angle = angle_to_t - self.missile.heading
        # normalize to [-pi, pi]
        rel_angle = (rel_angle + math.pi) % (2*math.pi) - math.pi

        return np.array([
            self.missile.x / ENV_WIDTH,
            self.missile.y / ENV_HEIGHT,
            self.missile.vx / MISSILE_MAX_SPEED,
            self.missile.vy / MISSILE_MAX_SPEED,
            math.sin(self.missile.heading),
            math.cos(self.missile.heading),
            self._dist_to_target() / diag,
            rel_angle / math.pi,
        ], dtype=np.float32)

    def _dist_to_target(self):
        return math.sqrt((self.missile.x-self.target[0])**2 + (self.missile.y-self.target[1])**2)

    def render(self):
        # delegate to renderer.py
        ...

    def close(self):
        if self.renderer: self.renderer.quit()
```

### P1-Task 4: Renderer (`env/renderer.py`) — Day 2

Pygame rendering with:
- Dark background with subtle grid
- Missile as colored triangle/arrow (rotated to heading)
- Target as pulsing green circle
- Obstacles as red circles
- Trail behind missile (fading dots)
- HUD overlay: steps, distance, reward

### P1-Task 5: Evaluate & Demo (`evaluate.py`) — Day 4

```python
# Usage: python evaluate.py --agent ppo --model models/ppo_best.pth
# Loads trained model, runs episodes with rendering, prints stats
```

### P1-Task 6: README.md — Day 4-5

Write clear README with:
- Project title & description
- Team members & roll numbers
- Installation (`pip install -r requirements.txt`)
- How to train: `python train_dqn.py` / `python train_ppo.py`
- How to evaluate: `python evaluate.py --agent ppo`
- Project structure
- Results summary with comparison plot image
- Demo GIF embedded

### P1 Testing (Self-validation before merge)

```python
# test_env.py — Person 1 runs this before pushing
from env.missile_env import MissileEnv
env = MissileEnv(render_mode="human")
for ep in range(5):
    obs, info = env.reset()
    assert obs.shape == (8,), f"Bad obs shape: {obs.shape}"
    assert env.observation_space.contains(obs), "Obs out of bounds"
    done = False
    total_r = 0
    while not done:
        action = env.action_space.sample()  # random agent
        obs, r, term, trunc, info = env.step(action)
        total_r += r
        env.render()
        done = term or trunc
    print(f"Ep {ep}: reward={total_r:.1f}")
env.close()
print("ALL TESTS PASSED")
```

---

## Person 2: RL Agents + Training + Plots

### P2-Task 1: Neural Networks (`agents/networks.py`) — Day 1

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """DQN: state → Q-value for each action"""
    def __init__(self, obs_dim=8, n_actions=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class PolicyNetwork(nn.Module):
    """PPO Actor: state → action probabilities"""
    def __init__(self, obs_dim=8, n_actions=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    """PPO Critic: state → scalar value"""
    def __init__(self, obs_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)
```

### P2-Task 2: Replay Buffer (`agents/replay_buffer.py`) — Day 1

```python
from collections import deque
import random, torch, numpy as np

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
```

### P2-Task 3: DQN Agent (`agents/dqn_agent.py`) — Day 1-2

```
DQN Algorithm (from Lecture 24):
1. Two networks: Q-network (trained) + Target network (frozen copy)
2. ε-greedy action selection
3. Store transitions in replay buffer
4. Sample random mini-batch → compute TD target → MSE loss → backprop
5. Periodically copy Q → Target
6. Decay ε over time
```

Key implementation:
```python
class DQNAgent:
    def __init__(self, obs_dim, n_actions, config):
        self.q_net = QNetwork(obs_dim, n_actions, config.DQN_HIDDEN_DIM)
        self.target_net = QNetwork(obs_dim, n_actions, config.DQN_HIDDEN_DIM)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.DQN_LR)
        self.buffer = ReplayBuffer(config.DQN_BUFFER_SIZE)
        self.epsilon = config.DQN_EPSILON_START
        self.gamma = config.DQN_GAMMA
        ...

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q_vals = self.q_net(torch.FloatTensor(state))
            return q_vals.argmax().item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
```

**Day 1-2 testing:** Test on `CartPole-v1` to verify DQN works:
```python
import gymnasium as gym
env = gym.make("CartPole-v1")
# should solve CartPole (reward > 400) within ~300 episodes
```

### P2-Task 4: PPO Agent (`agents/ppo_agent.py`) — Day 2-3

```
PPO Algorithm (from Lecture 24):
1. Actor (policy) + Critic (value) networks
2. Collect rollout of N steps using current policy
3. Compute GAE advantages
4. For K epochs, run mini-batch SGD with clipped objective
5. Repeat
```

Key implementation:
```python
class PPOAgent:
    def __init__(self, obs_dim, n_actions, config):
        self.actor = PolicyNetwork(obs_dim, n_actions, config.PPO_HIDDEN_DIM)
        self.critic = ValueNetwork(obs_dim, config.PPO_HIDDEN_DIM)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.PPO_LR
        )
        ...

    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor(torch.FloatTensor(state))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        for _ in range(self.ppo_epochs):
            # shuffle and create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                batch = indices[start:start+self.batch_size]
                b_states = states[batch]
                b_actions = actions[batch]
                b_old_lp = old_log_probs[batch]
                b_returns = returns[batch]
                b_advantages = advantages[batch]

                # actor loss (clipped)
                probs = self.actor(b_states)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(b_actions)
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                values = self.critic(b_states).squeeze()
                critic_loss = nn.MSELoss()(values, b_returns)

                # entropy bonus
                entropy = dist.entropy().mean()

                loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

**Day 2-3 testing:** Test on `CartPole-v1` before using real env.

### P2-Task 5: Training Scripts — Day 2-3

**`train_dqn.py`:**
```python
# Works with ANY gym env — swap env name to switch
ENV_NAME = "MissileEnv"  # change to "CartPole-v1" for testing

def train():
    env = make_env(ENV_NAME)
    agent = DQNAgent(obs_dim=8, n_actions=5, config=config)
    rewards_log = []

    for episode in range(config.DQN_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, float(term))
            loss = agent.train_step()
            obs = next_obs
            total_reward += reward
            done = term or trunc

        agent.decay_epsilon()
        if episode % config.DQN_TARGET_UPDATE == 0:
            agent.update_target()

        rewards_log.append(total_reward)
        if episode % 10 == 0:
            avg = np.mean(rewards_log[-100:])
            print(f"Ep {episode} | Reward: {total_reward:.1f} | Avg100: {avg:.1f} | ε: {agent.epsilon:.3f}")

    # save model and logs
    agent.save("models/dqn_best.pth")
    pd.DataFrame({"reward": rewards_log}).to_csv("results/dqn_rewards.csv")
```

**`train_ppo.py`:** similar structure with rollout collection loop.

### P2-Task 6: Plotting (`plot_results.py`) — Day 4-5

```python
# Load CSVs, plot smoothed curves, DQN vs PPO comparison
# Save to results/comparison_plot.png
```

Generate:
1. **Training curve:** Episode reward vs Episode (smoothed, rolling avg 100)
2. **DQN vs PPO comparison:** both curves on same axes
3. **Success rate:** % of episodes where target was reached (rolling window)
4. **Steps to target:** average episode length over training

### P2 Testing (Self-validation before merge)

```python
# test_agents.py — run on CartPole before merge
import gymnasium as gym
env = gym.make("CartPole-v1")
agent = DQNAgent(obs_dim=4, n_actions=2, config=cartpole_config)
# train for 300 episodes, verify reward > 400
# if yes → DQN code is correct

agent = PPOAgent(obs_dim=4, n_actions=2, config=cartpole_config)
# train for 50000 steps, verify reward > 400
# if yes → PPO code is correct
```

---

## Hyperparameters (`configs/default.py`) — Both Agree

```python
# ─── Environment ───
ENV_WIDTH = 800
ENV_HEIGHT = 600
MAX_STEPS = 500
NUM_OBSTACLES = 5
TARGET_RADIUS = 20
MISSILE_MAX_SPEED = 5.0
MISSILE_TURN_RATE = 0.1        # radians per step (~5.7 degrees)
MISSILE_THRUST = 0.3

# ─── Rewards ───
REWARD_HIT_TARGET = +100.0
REWARD_HIT_OBSTACLE = -50.0
REWARD_HIT_WALL = -50.0
REWARD_TIME_PENALTY = -0.1
REWARD_DISTANCE_SHAPING = 1.0
REWARD_HEADING_BONUS = 0.1

# ─── DQN ───
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.01
DQN_EPSILON_DECAY = 0.995
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 50000
DQN_TARGET_UPDATE = 10
DQN_HIDDEN_DIM = 128
DQN_EPISODES = 2000

# ─── PPO ───
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
PPO_BATCH_SIZE = 64
PPO_ROLLOUT_STEPS = 2048
PPO_HIDDEN_DIM = 128
PPO_TOTAL_TIMESTEPS = 500000
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_COEFF = 0.5
```

---

## Dependencies (`requirements.txt`)

```
torch>=2.0
gymnasium>=0.29
pygame>=2.5
numpy>=1.24
matplotlib>=3.7
pandas>=2.0
```

---

## Day-by-Day Timeline

### Day 1 (Oct 21) — Both start independently

| Person 1 (Environment) | Person 2 (Agents) |
|---|---|
| Create repo structure, `configs/default.py` | Pull repo, read config |
| `env/missile_physics.py` — Missile class | `agents/networks.py` — all 3 networks |
| `env/obstacles.py` — generation + collision | `agents/replay_buffer.py` |
| Test physics manually (print positions) | `agents/dqn_agent.py` — full DQN |
| **Push to `env/` branch** | Test DQN on CartPole → must solve it |
| | **Push to `agents/` branch** |

### Day 2 (Oct 22) — Continue independently

| Person 1 (Environment) | Person 2 (Agents) |
|---|---|
| `env/missile_env.py` — full Gym wrapper | `agents/ppo_agent.py` — full PPO |
| `env/renderer.py` — basic Pygame render | `train_dqn.py` |
| Run `test_env.py` — random agent works | `train_ppo.py` |
| Fix any obs normalization bugs | Test PPO on CartPole → must solve it |
| **Push env to main branch** | **Push agents to main branch** |

### Day 3 (Oct 23) — MERGE & First Real Training

| Person 1 | Person 2 |
|---|---|
| Pull Person 2's code | Pull Person 1's env |
| Help debug env issues | **Switch from CartPole → MissileEnv** |
| Watch training, tune rewards | First DQN training on real env |
| Adjust physics if too hard/easy | First PPO training on real env |
| Fix rendering bugs | Tune learning rates if needed |

### Day 4 (Oct 24) — Train + Polish

| Person 1 | Person 2 |
|---|---|
| Polish renderer (trail, HUD, colors) | Long DQN training run (~2000 eps) |
| `evaluate.py` — load model + render | Long PPO training run (~500K steps) |
| Record demo GIF | Save best models |
| Start README.md | `plot_results.py` — generate plots |

### Day 5 (Oct 25) — Final

| Person 1 | Person 2 |
|---|---|
| Finish README.md with plots + GIF | Retrain with tuned hyperparams if needed |
| Test: clone fresh, `pip install`, train | Final comparison plots |
| Clean up code, add comments | Double-check all scripts run cleanly |
| **Both: final push to GitHub by 8 PM** | |

---

## Integration Checklist

### Before Merge (Day 3 morning)
- [ ] Person 1: `test_env.py` passes (random agent, 100 episodes, no crash)
- [ ] Person 2: DQN solves CartPole (reward > 400 within 300 eps)
- [ ] Person 2: PPO solves CartPole (reward > 400 within 50K steps)
- [ ] Both: `configs/default.py` has same values on both branches

### After Merge (Day 3 evening)
- [ ] `python train_dqn.py` runs on MissileEnv without crashing for 100 eps
- [ ] Reward trend is generally increasing (even if noisy)
- [ ] `python train_ppo.py` runs without NaN or crashes
- [ ] Rendering shows missile moving sensibly

### Before Submission (Day 5)
- [ ] `python train_dqn.py` → model saved to `models/dqn_best.pth`
- [ ] `python train_ppo.py` → model saved to `models/ppo_best.pth`
- [ ] `python evaluate.py --agent ppo` → trained missile navigates to target
- [ ] `python plot_results.py` → `results/comparison_plot.png` exists
- [ ] `README.md` has install, train, evaluate instructions
- [ ] `requirements.txt` is complete
- [ ] Demo GIF in `demo/` and embedded in README
- [ ] Meaningful commit history (not just 1 big commit)
- [ ] Repo shared with: shubhambagwari, divyaanshmertia (if private)

---

## Who Does What — Final Summary

| File | Person 1 | Person 2 |
|------|:--------:|:--------:|
| `configs/default.py` | ✅ creates | reviews |
| `env/missile_physics.py` | ✅ | |
| `env/obstacles.py` | ✅ | |
| `env/missile_env.py` | ✅ | |
| `env/renderer.py` | ✅ | |
| `agents/networks.py` | | ✅ |
| `agents/replay_buffer.py` | | ✅ |
| `agents/dqn_agent.py` | | ✅ |
| `agents/ppo_agent.py` | | ✅ |
| `train_dqn.py` | | ✅ |
| `train_ppo.py` | | ✅ |
| `evaluate.py` | ✅ | |
| `plot_results.py` | | ✅ |
| `README.md` | ✅ | |
| `requirements.txt` | ✅ | reviews |
| Demo GIF | ✅ | |
| Comparison plots | | ✅ |
