import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.missile_physics import Missile
from env.obstacles import (
    generate_obstacles,
    check_obstacle_collision,
    check_wall_collision,
    check_target_reached,
)
from configs.default import *


class MissileEnv(gym.Env):
    """
    Gymnasium environment for missile navigation.

    The missile must navigate from a random start position to a random
    target position while avoiding circular obstacles.

    Observation (8 floats, normalized to [-1, 1]):
        [0] missile_x / ENV_WIDTH
        [1] missile_y / ENV_HEIGHT
        [2] missile_vx / MAX_SPEED
        [3] missile_vy / MAX_SPEED
        [4] sin(heading)
        [5] cos(heading)
        [6] distance_to_target / diagonal  (normalized)
        [7] relative_angle_to_target / pi  (normalized)

    Actions (Discrete 5):
        0 = coast, 1 = turn left, 2 = turn right,
        3 = accelerate, 4 = decelerate

    Rewards:
        +100  hit target
        -50   hit obstacle or wall
        -0.1  per step (time penalty)
        + shaping for getting closer to target
        + small bonus for facing target
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # 8 continuous observations
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # will be initialized in reset()
        self.missile = None
        self.target = None
        self.obstacles = []
        self.steps = 0
        self.prev_dist = 0.0
        self.initial_dist = 1.0

        # renderer is created lazily on first render() call
        self._renderer = None
        self._diag = math.sqrt(ENV_WIDTH**2 + ENV_HEIGHT**2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # missile starts in bottom-left region
        mx = self.np_random.uniform(50, ENV_WIDTH * 0.3)
        my = self.np_random.uniform(ENV_HEIGHT * 0.6, ENV_HEIGHT - 50)

        # target in top-right region (far from start)
        tx = self.np_random.uniform(ENV_WIDTH * 0.6, ENV_WIDTH - 50)
        ty = self.np_random.uniform(50, ENV_HEIGHT * 0.4)

        # point missile roughly toward the target with some noise
        angle_to_target = math.atan2(ty - my, tx - mx)
        heading_noise = self.np_random.uniform(-0.5, 0.5)

        self.missile = Missile(mx, my, heading=angle_to_target + heading_noise)
        self.target = (tx, ty)
        self.obstacles = generate_obstacles(
            NUM_OBSTACLES, ENV_WIDTH, ENV_HEIGHT,
            (mx, my), (tx, ty), rng=self.np_random
        )

        self.steps = 0
        self.prev_dist = self._dist_to_target()
        self.initial_dist = max(self.prev_dist, 1.0)

        obs = self._get_obs()
        info = {"dist": self.prev_dist}
        return obs, info

    def step(self, action):
        self.missile.update(action)
        self.steps += 1

        terminated = False
        truncated = False
        reward = REWARD_TIME_PENALTY

        curr_dist = self._dist_to_target()

        # reward shaping: bonus for getting closer to target
        dist_delta = self.prev_dist - curr_dist
        reward += REWARD_DISTANCE_SHAPING * dist_delta / self.initial_dist

        # small bonus for facing the target
        angle_to_target = math.atan2(
            self.target[1] - self.missile.y,
            self.target[0] - self.missile.x,
        )
        angle_diff = abs(angle_to_target - self.missile.heading)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        if angle_diff < math.pi / 4:
            reward += REWARD_HEADING_BONUS

        # check if target reached
        if check_target_reached(self.missile.x, self.missile.y,
                                self.target, TARGET_RADIUS):
            reward = REWARD_HIT_TARGET
            terminated = True

        # check obstacle collision
        elif check_obstacle_collision(self.missile.x, self.missile.y,
                                      MISSILE_RADIUS, self.obstacles):
            reward = REWARD_HIT_OBSTACLE
            terminated = True

        # check wall collision
        elif check_wall_collision(self.missile.x, self.missile.y,
                                  ENV_WIDTH, ENV_HEIGHT):
            reward = REWARD_HIT_WALL
            terminated = True

        # check step limit
        if self.steps >= MAX_STEPS:
            truncated = True

        self.prev_dist = curr_dist
        obs = self._get_obs()
        info = {"dist": curr_dist, "steps": self.steps}

        # auto-render if human mode
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Build the 8-dimensional observation vector."""
        mx, my = self.missile.x, self.missile.y
        tx, ty = self.target

        # angle from missile to target
        angle_to_t = math.atan2(ty - my, tx - mx)
        # relative angle (how far off the missile is pointing)
        rel_angle = angle_to_t - self.missile.heading
        # normalize to [-pi, pi]
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        obs = np.array([
            mx / ENV_WIDTH,
            my / ENV_HEIGHT,
            self.missile.vx / max(MISSILE_MAX_SPEED, 0.01),
            self.missile.vy / max(MISSILE_MAX_SPEED, 0.01),
            math.sin(self.missile.heading),
            math.cos(self.missile.heading),
            self._dist_to_target() / self._diag,
            rel_angle / math.pi,
        ], dtype=np.float32)

        return obs

    def _dist_to_target(self):
        dx = self.missile.x - self.target[0]
        dy = self.missile.y - self.target[1]
        return math.sqrt(dx * dx + dy * dy)

    def render(self):
        """Render the current state using Pygame."""
        if self._renderer is None:
            from env.renderer import Renderer
            self._renderer = Renderer(ENV_WIDTH, ENV_HEIGHT, self.render_mode)

        return self._renderer.draw(
            missile=self.missile,
            target=self.target,
            obstacles=self.obstacles,
            step=self.steps,
            dist=self._dist_to_target(),
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
