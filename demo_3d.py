"""
3D Demo of MissileSim using Ursina Engine.
Shows the rocket navigating through the level in true 3D.

Usage:
    python demo_3d.py              # random agent demo
    python demo_3d.py --model ppo  # trained PPO agent (after training)
"""
import sys
import math
import time as time_module

# must import ursina before pygame to avoid conflicts
from ursina import (
    Ursina, Entity, camera, color, Vec3,
    window, application, Text, time, held_keys,
    DirectionalLight, AmbientLight, Sky,
)

from env.missile_physics import Missile
from env.obstacles import build_level
from configs.default import *

import random


# ── scale factor: env coords → 3D world ──
S = 0.1  # scale


def env_to_3d(ex, ey):
    """Convert env (x, y) to 3D (x, y, z=0)."""
    return ex * S, (300 - ey) * S, 0


class MissileSim3D:
    """Full 3D visualization of the missile environment."""

    def __init__(self):
        self.app = Ursina(borderless=False, title="MissileSim 3D", size=(1100, 750))
        window.color = color.rgb(8, 8, 20)
        window.fps_counter.enabled = True

        # lighting
        sun = DirectionalLight(shadow_map_resolution=Vec3(1024, 1024, 1))
        sun.look_at(Vec3(1, -1, -0.5))

        # build the level
        self.zones, self.all_obstacles, self.boss_pos, self.checkpoints = \
            build_level(ENV_HEIGHT)

        self._build_static_level()
        self._build_rocket()
        self._build_boss()
        self._build_hud()

        # missile state
        self.missile = Missile(80, 300, heading=-0.2)
        self.steps = 0
        self.prev_x = 80
        self.game_over = False
        self.won = False
        self.passed_cps = set()

    def _build_static_level(self):
        """Build 3D geometry for the entire level."""
        D = 10  # corridor depth (z-axis)

        # ── Launch Pad: green ground ──
        Entity(model='cube', color=color.rgb(35, 140, 50),
               position=(200 * S, -30 * S, 0), scale=(500 * S, 5 * S, D * 1.5))
        # launch pad concrete
        Entity(model='cube', color=color.rgb(130, 130, 130),
               position=(80 * S, -28 * S, 0), scale=(4, 0.5, 4))
        # factory building
        Entity(model='cube', color=color.rgb(120, 120, 130),
               position=(-5, 5 * S, -8), scale=(10, 8, 5))
        # chimney tower
        Entity(model='cube', color=color.rgb(90, 90, 100),
               position=(-3, 20 * S, -8), scale=(2.5, 12, 2.5))

        # ── Chimney Entry: walls with gap ──
        gap_top = (200 - 60) * S   # top of gap
        gap_bot = (200 + 60) * S   # bottom of gap

        # top wall block
        Entity(model='cube', color=color.rgb(100, 100, 110),
               position=(525 * S, (300 - 70) * S, 0),
               scale=(250 * S, 140 * S, D))
        # bottom wall block
        Entity(model='cube', color=color.rgb(100, 100, 110),
               position=(525 * S, (300 - 530) * S, 0),
               scale=(250 * S, 340 * S, D))

        # ── Shaft: side walls + protruding blocks ──
        shaft_cx = (650 + 225) * S
        # back wall
        Entity(model='cube', color=color.rgb(60, 60, 80),
               position=(shaft_cx, 0, D / 2 + 0.5), scale=(450 * S, 600 * S, 1))
        # front wall (with opening)
        Entity(model='cube', color=color.rgb(60, 60, 80),
               position=(shaft_cx, 0, -D / 2 - 0.5), scale=(450 * S, 600 * S, 1))
        # left boundary
        Entity(model='cube', color=color.rgb(70, 70, 90),
               position=(650 * S - 1, 0, 0), scale=(2, 600 * S, D))
        # protruding blocks
        Entity(model='cube', color=color.rgb(140, 90, 80),
               position=((650 + 180) * S, (300 - 220) * S, 0),
               scale=(60 * S, 40 * S, D * 0.7))
        Entity(model='cube', color=color.rgb(140, 90, 80),
               position=((650 + 340) * S, (300 - 395) * S, 0),
               scale=(70 * S, 35 * S, D * 0.7))

        # ── Pipe Corridor: ceiling + floor + terracotta pipes ──
        pc_cx = (1100 + 350) * S
        Entity(model='cube', color=color.rgb(180, 100, 80),
               position=(pc_cx, 27 * S, 0), scale=(700 * S, 60 * S, D))
        Entity(model='cube', color=color.rgb(180, 100, 80),
               position=(pc_cx, -27 * S, 0), scale=(700 * S, 60 * S, D))

        # pipes
        pipes = [(1200, 150, 100), (1350, 350, 120), (1500, 200, 90),
                 (1600, 420, 110), (1700, 280, 100)]
        for px, py, pw in pipes:
            y3 = (300 - py) * S
            # main pipe
            Entity(model='cube', color=color.rgb(200, 110, 80),
                   position=(px * S, y3, 0), scale=(pw * S, 25 * S, D * 0.8))
            # pipe highlight
            Entity(model='cube', color=color.rgb(220, 130, 100),
                   position=(px * S, y3 + 12 * S, 0),
                   scale=(pw * S, 3 * S, D * 0.85))

        # vertical bars
        Entity(model='cube', color=color.rgb(180, 90, 70),
               position=(1380 * S, 12 * S, 0), scale=(20 * S, 120 * S, 20 * S))
        Entity(model='cube', color=color.rgb(180, 90, 70),
               position=(1520 * S, -18 * S, 0), scale=(20 * S, 120 * S, 20 * S))

        # ── Factory Hall: ceiling + floor + pillars + conveyor ──
        hc = (1800 + 400) * S
        Entity(model='cube', color=color.rgb(180, 110, 75),
               position=(hc, 28 * S, 0), scale=(800 * S, 50 * S, D))
        Entity(model='cube', color=color.rgb(180, 110, 75),
               position=(hc, -27 * S, 0), scale=(800 * S, 70 * S, D))

        # pillars
        for px in [1900, 2100, 2300]:
            Entity(model='cube', color=color.rgb(150, 85, 55),
                   position=(px * S, 0, D / 2 - 1.5),
                   scale=(30 * S, 200 * S, 30 * S))

        # conveyor belt
        for i in range(25):
            cx = (1850 + i * 20) * S
            col = color.rgb(220, 200, 40) if i % 2 == 0 else color.rgb(40, 40, 40)
            Entity(model='cube', color=col,
                   position=(cx, -21 * S, -1), scale=(18 * S, 12 * S, 2.5))

        # boxes
        for i in range(6):
            Entity(model='cube', color=color.rgb(220, 140, 40),
                   position=((1870 + i * 80) * S, -18 * S, -1),
                   scale=(1.2, 1.2, 1.2))

        # workers
        for i in range(6):
            wx = (1860 + i * 80) * S
            Entity(model='cube', color=color.rgb(200, 60, 60),
                   position=(wx, -15 * S, -3), scale=(0.6, 1.3, 0.6))
            Entity(model='sphere', color=color.rgb(220, 180, 140),
                   position=(wx, -12 * S, -3), scale=0.5)
            Entity(model='cube', color=color.rgb(220, 160, 30),
                   position=(wx, -11 * S, -3), scale=(0.65, 0.2, 0.65))

        # ── Boss Arena ──
        ac = (2500 + 250) * S
        Entity(model='cube', color=color.rgb(160, 80, 60),
               position=(ac, 28 * S, 0), scale=(500 * S, 50 * S, D))
        Entity(model='cube', color=color.rgb(160, 80, 60),
               position=(ac, -27 * S, 0), scale=(500 * S, 70 * S, D))
        # boss platform
        Entity(model='cube', color=color.rgb(160, 80, 60),
               position=(2880 * S, -17 * S, 0), scale=(120 * S, 100 * S, D * 0.8))

    def _build_rocket(self):
        """Build the 3D rocket model."""
        self.rocket = Entity(model='cube', color=color.rgb(200, 220, 240),
                             scale=(2.2, 0.55, 0.55))
        # nose cone
        Entity(model='cube', color=color.rgb(160, 160, 170),
               parent=self.rocket, position=(0.8, 0, 0),
               rotation=(0, 0, -90), scale=(0.5, 0.7, 0.5))
        # fins
        Entity(model='cube', color=color.rgb(70, 70, 90),
               parent=self.rocket, position=(-0.6, 0.35, 0),
               scale=(0.5, 0.35, 0.1))
        Entity(model='cube', color=color.rgb(70, 70, 90),
               parent=self.rocket, position=(-0.6, -0.35, 0),
               scale=(0.5, 0.35, 0.1))
        Entity(model='cube', color=color.rgb(70, 70, 90),
               parent=self.rocket, position=(-0.6, 0, 0.35),
               scale=(0.5, 0.1, 0.35))
        Entity(model='cube', color=color.rgb(70, 70, 90),
               parent=self.rocket, position=(-0.6, 0, -0.35),
               scale=(0.5, 0.1, 0.35))

        # flames
        self.flame_outer = Entity(model='cube', color=color.rgb(255, 130, 20),
                                  parent=self.rocket, position=(-1.2, 0, 0),
                                  scale=(1, 0.35, 0.35), enabled=False)
        self.flame_inner = Entity(model='cube', color=color.rgb(255, 220, 60),
                                  parent=self.rocket, position=(-0.9, 0, 0),
                                  scale=(0.6, 0.18, 0.18), enabled=False)

    def _build_boss(self):
        """Build the boss character at the end."""
        bx, by = 2900 * S, -8 * S
        self.boss = Entity(model='cube', color=color.rgb(50, 160, 80),
                           position=(bx, by, 0), scale=(1.2, 2.2, 0.9))
        Entity(model='sphere', color=color.rgb(220, 180, 140),
               parent=self.boss, position=(0, 1.4, 0), scale=0.8)
        Entity(model='cube', color=color.rgb(100, 70, 40),
               parent=self.boss, position=(0, 1.1, 0.3), scale=(0.5, 0.4, 0.3))
        Entity(model='cube', color=color.rgb(50, 50, 70),
               parent=self.boss, position=(-0.2, -1.3, 0), scale=(0.3, 0.9, 0.3))
        Entity(model='cube', color=color.rgb(50, 50, 70),
               parent=self.boss, position=(0.2, -1.3, 0), scale=(0.3, 0.9, 0.3))
        self.boss_arm = Entity(model='cube', color=color.rgb(50, 160, 80),
                               parent=self.boss, position=(0.6, 0.7, 0),
                               rotation=(0, 0, -50), scale=(0.2, 0.8, 0.2))

        self.boss_label = Text(text="BOSS", parent=self.boss,
                               position=(0, 2.5, 0), origin=(0, 0),
                               scale=8, color=color.white, billboard=True)

    def _build_hud(self):
        """Build HUD text."""
        self.hud = Text(text="", position=(-0.85, 0.46),
                        scale=1.3, color=color.rgb(200, 200, 200))
        self.zone_text = Text(text="", position=(0, 0.46), origin=(0, 0),
                              scale=1.8, color=color.rgb(100, 200, 255))
        self.progress_bar_bg = Entity(
            model='quad', color=color.rgb(40, 40, 60),
            position=(0, -0.47, 0), scale=(1.5, 0.02), parent=camera.ui)
        self.progress_bar = Entity(
            model='quad', color=color.rgb(50, 200, 100),
            position=(-0.75, -0.47, -0.1), scale=(0.01, 0.018),
            origin=(-0.5, 0), parent=camera.ui)

    def run(self):
        """Main loop — runs the simulation with random actions or loaded agent."""
        def update():
            if self.game_over:
                return

            # random action (replace with agent.select_action for trained model)
            action = random.randint(0, 6)

            self.missile.update(action)
            self.steps += 1
            mx, my = self.missile.x, self.missile.y

            # update 3D rocket position
            x3, y3, z3 = env_to_3d(mx, my)
            self.rocket.position = Vec3(x3, y3, z3)
            angle_deg = -math.degrees(self.missile.heading)
            self.rocket.rotation = Vec3(0, 0, angle_deg)

            # flame
            if self.missile.thrusting and self.missile.speed > 0.2:
                self.flame_outer.enabled = True
                self.flame_inner.enabled = True
                fs = 0.7 + self.missile.speed * 0.12 + random.uniform(-0.1, 0.1)
                self.flame_outer.scale_x = fs
                self.flame_inner.scale_x = fs * 0.55
            else:
                self.flame_outer.enabled = False
                self.flame_inner.enabled = False

            # boss arm wave
            wave = math.sin(time.time() * 3) * 25
            self.boss_arm.rotation_z = -50 + wave

            # camera: third-person chase
            cam_back = 12
            cam_up = 5
            cam_target = Vec3(x3 + 8, y3, z3)
            camera.position = Vec3(x3 - cam_back, y3 + cam_up, -14)
            camera.look_at(cam_target)

            # progress
            progress = min(100, mx / 3000 * 100)
            self.progress_bar.scale_x = max(0.01, 1.5 * progress / 100)

            # zone detection
            zone_name = ""
            for z in self.zones:
                if z.x_start <= mx < z.x_end:
                    zone_name = z.name
                    break
            self.zone_text.text = zone_name

            # boss distance
            boss_dist = math.sqrt((mx - self.boss_pos[0])**2 + (my - self.boss_pos[1])**2)

            # HUD
            self.hud.text = (
                f"Step: {self.steps}  |  Speed: {self.missile.speed:.1f}  |  "
                f"Boss: {boss_dist:.0f}  |  Progress: {progress:.0f}%"
            )

            # check terminal conditions
            from env.obstacles import check_obstacle_collision, check_wall_collision
            if boss_dist < 40:
                self.game_over = True
                self.won = True
                self.zone_text.text = "TARGET HIT!"
                self.zone_text.color = color.rgb(255, 200, 50)
            elif check_obstacle_collision(mx, my, MISSILE_RADIUS, self.all_obstacles):
                self.game_over = True
                self.zone_text.text = f"CRASHED in {zone_name}!"
                self.zone_text.color = color.rgb(255, 80, 80)
            elif check_wall_collision(mx, my, ENV_HEIGHT) or mx < -50:
                self.game_over = True
                self.zone_text.text = "OUT OF BOUNDS!"
                self.zone_text.color = color.rgb(255, 80, 80)
            elif self.steps >= MAX_STEPS:
                self.game_over = True
                self.zone_text.text = "TIME UP!"
                self.zone_text.color = color.rgb(255, 200, 50)

        # register update function with ursina
        from ursina import invoke
        self.app.update = update
        self.app.run()


if __name__ == "__main__":
    sim = MissileSim3D()
    sim.run()
