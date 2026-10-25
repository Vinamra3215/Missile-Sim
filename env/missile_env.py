"""
MissileSim — Stage-based RL environment.
The rocket navigates through 6 sequential zones to reach and hit the Boss.
Side-scrolling camera tracks the rocket.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.missile_physics import Missile
from env.obstacles import build_level, check_obstacle_collision, check_wall_collision
from configs.default import *


class MissileEnv(gym.Env):
    """
    Stage-based missile navigation environment.

    Observation (10 floats):
        [0] rocket_y / ENV_HEIGHT          (vertical position)
        [1] rocket_vx / MAX_SPEED          (horizontal velocity)
        [2] rocket_vy / MAX_SPEED          (vertical velocity)
        [3] sin(heading)
        [4] cos(heading)
        [5] progress (x / WORLD_WIDTH)     (how far along the level)
        [6] dist_to_boss / diagonal        (normalized)
        [7] angle_to_boss / pi             (normalized)
        [8] speed / MAX_SPEED              (current speed)
        [9] nearest_obstacle_dist / 200    (clamped, normalized)

    Actions (Discrete 7):
        0=coast, 1=turn_left, 2=turn_right,
        3=accelerate, 4=decelerate,
        5=thrust+left, 6=thrust+right
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(10,), dtype=np.float32
        )

        # build the level
        self.zones, self.all_obstacles, self.boss_pos, self.checkpoints = \
            build_level(ENV_HEIGHT)

        self.missile = None
        self.steps = 0
        self.prev_x = 0.0
        self.passed_checkpoints = set()
        self._diag = math.sqrt(WORLD_WIDTH**2 + ENV_HEIGHT**2)
        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # rocket starts mid-height in the launch zone, aimed toward chimney
        start_x = 80
        start_y = 300
        heading = -0.2  # slightly upward-right toward chimney opening

        self.missile = Missile(start_x, start_y, heading=heading)
        self.steps = 0
        self.prev_x = start_x
        self.passed_checkpoints = set()

        # rebuild level for consistent layout
        self.zones, self.all_obstacles, self.boss_pos, self.checkpoints = \
            build_level(ENV_HEIGHT)

        return self._get_obs(), {"zone": "Launch Pad"}

    def step(self, action):
        self.missile.update(action)
        self.steps += 1

        terminated = False
        truncated = False
        reward = REWARD_TIME_PENALTY

        mx, my = self.missile.x, self.missile.y

        # forward progress reward
        dx = mx - self.prev_x
        if dx > 0:
            reward += REWARD_PROGRESS * (dx / 10.0)

        # checkpoint bonuses
        for cp in self.checkpoints:
            if cp not in self.passed_checkpoints and mx >= cp:
                reward += REWARD_CHECKPOINT
                self.passed_checkpoints.add(cp)

        # heading bonus: reward for pointing generally right (toward boss)
        angle_to_boss = math.atan2(
            self.boss_pos[1] - my, self.boss_pos[0] - mx
        )
        angle_diff = abs(angle_to_boss - self.missile.heading)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        if angle_diff < math.pi / 3:
            reward += REWARD_HEADING_BONUS

        # check if boss reached
        boss_dist = math.sqrt(
            (mx - self.boss_pos[0])**2 + (my - self.boss_pos[1])**2
        )
        if boss_dist < 40:
            reward = REWARD_HIT_TARGET
            terminated = True

        # check obstacle collision
        elif check_obstacle_collision(mx, my, MISSILE_RADIUS, self.all_obstacles):
            reward = REWARD_HIT_OBSTACLE
            terminated = True

        # check vertical bounds
        elif check_wall_collision(mx, my, ENV_HEIGHT):
            reward = REWARD_HIT_WALL
            terminated = True

        # check if went too far left (behind start)
        elif mx < -50:
            reward = REWARD_HIT_WALL
            terminated = True

        # truncate on time limit
        if self.steps >= MAX_STEPS:
            truncated = True

        self.prev_x = mx

        # get current zone name
        zone_name = "Unknown"
        for z in self.zones:
            if z.x_start <= mx < z.x_end:
                zone_name = z.name
                break

        obs = self._get_obs()
        info = {
            "zone": zone_name,
            "dist_to_boss": boss_dist,
            "progress": mx / WORLD_WIDTH,
            "steps": self.steps,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        mx, my = self.missile.x, self.missile.y

        # angle to boss
        angle_to_boss = math.atan2(
            self.boss_pos[1] - my, self.boss_pos[0] - mx
        )
        rel_angle = angle_to_boss - self.missile.heading
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi

        # distance to boss
        boss_dist = math.sqrt(
            (mx - self.boss_pos[0])**2 + (my - self.boss_pos[1])**2
        )

        # nearest obstacle distance (simple scan)
        min_obs_dist = 200.0
        for obs in self.all_obstacles:
            if hasattr(obs, 'x'):  # Rect
                # center of rect
                cx = obs.x + obs.w / 2
                cy = obs.y + obs.h / 2
                d = math.sqrt((mx - cx)**2 + (my - cy)**2)
                # approximate: subtract half the diagonal
                half_diag = math.sqrt(obs.w**2 + obs.h**2) / 2
                d = max(0, d - half_diag)
            else:  # Pipe
                cx = (obs.x1 + obs.x2) / 2
                cy = (obs.y1 + obs.y2) / 2
                d = math.sqrt((mx - cx)**2 + (my - cy)**2) - obs.radius
                d = max(0, d)
            if d < min_obs_dist:
                min_obs_dist = d

        return np.array([
            my / ENV_HEIGHT,
            self.missile.vx / max(MISSILE_MAX_SPEED, 0.01),
            self.missile.vy / max(MISSILE_MAX_SPEED, 0.01),
            math.sin(self.missile.heading),
            math.cos(self.missile.heading),
            mx / WORLD_WIDTH,
            boss_dist / self._diag,
            rel_angle / math.pi,
            self.missile.speed / max(MISSILE_MAX_SPEED, 0.01),
            min(min_obs_dist, 200.0) / 200.0,
        ], dtype=np.float32)

    def render(self):
        if self._renderer is None:
            from env.renderer import Renderer
            self._renderer = Renderer(ENV_WIDTH, ENV_HEIGHT, self.render_mode)

        return self._renderer.draw(
            missile=self.missile,
            boss_pos=self.boss_pos,
            zones=self.zones,
            step=self.steps,
            camera_x=self.missile.x - ENV_WIDTH * 0.3,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
