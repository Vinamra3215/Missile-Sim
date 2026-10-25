"""
3D Renderer for MissileSim using Ursina Engine.
Renders the stage-based level in true 3D with third-person camera.
"""
import math
import sys

try:
    from ursina import (
        Ursina, Entity, camera, color, Vec3, Vec2,
        held_keys, time, window, application, Text,
        destroy, scene, Sky, DirectionalLight, AmbientLight,
    )
    from ursina.prefabs.first_person_controller import FirstPersonController
    HAS_URSINA = True
except ImportError:
    HAS_URSINA = False


# scale factor: env coordinates → 3D world (env is 3000x600, 3D is 300x60)
SCALE = 0.1
CORRIDOR_DEPTH = 8  # z-axis depth of the corridor


class Renderer3D:
    """Ursina-based 3D renderer for the missile environment."""

    def __init__(self):
        if not HAS_URSINA:
            raise ImportError("ursina is required: pip install ursina")

        self.app = Ursina(borderless=False, title="MissileSim 3D", size=(1000, 700))
        window.color = color.rgb(10, 10, 25)
        window.fps_counter.enabled = True

        # lighting
        self.sun = DirectionalLight()
        self.sun.look_at(Vec3(1, -1, -1))

        # build static level geometry
        self.level_entities = []
        self.rocket_entity = None
        self.boss_entity = None
        self.boss_label = None
        self.flame_entity = None
        self.hud_text = None
        self.progress_text = None
        self.trail_entities = []

        self._build_level()
        self._build_rocket()
        self._build_boss()
        self._build_hud()

    def _build_level(self):
        """Create 3D geometry for all zones."""

        # ─── Zone 1: Launch Pad ───
        # green ground plane
        ground = Entity(
            model='cube',
            color=color.rgb(40, 160, 60),
            position=(20 * SCALE, -3, 0),
            scale=(400 * SCALE, 1, CORRIDOR_DEPTH * 2),
        )
        self.level_entities.append(ground)

        # launch pad platform
        pad = Entity(
            model='cube',
            color=color.rgb(100, 100, 100),
            position=(4, -2.5, 0),
            scale=(3, 0.5, 3),
        )
        self.level_entities.append(pad)

        # factory building behind
        factory = Entity(
            model='cube',
            color=color.rgb(120, 120, 130),
            position=(-2, 2, -5),
            scale=(8, 6, 4),
        )
        self.level_entities.append(factory)

        # chimney
        chimney = Entity(
            model='cube',
            color=color.rgb(100, 100, 110),
            position=(-2, 7, -5),
            scale=(2, 8, 2),
        )
        self.level_entities.append(chimney)

        # ─── Zone 2: Chimney Entry - narrow gap walls ───
        chimney_x = 400 * SCALE
        gap_center = 600 * SCALE - 30  # env y=200 → 3D y
        gap_half = 60 * SCALE

        # top wall
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(100, 100, 110),
            position=(chimney_x + 12.5 * SCALE, 20 * SCALE, 0),
            scale=(250 * SCALE, 140 * SCALE * 2, CORRIDOR_DEPTH),
        ))
        # bottom wall
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(100, 100, 110),
            position=(chimney_x + 12.5 * SCALE, -22 * SCALE, 0),
            scale=(250 * SCALE, 340 * SCALE * 2, CORRIDOR_DEPTH),
        ))

        # ─── Zone 3: Shaft - walls ───
        shaft_x = 650 * SCALE
        # left wall
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(80, 80, 100),
            position=(shaft_x + 9 * SCALE, 0, -CORRIDOR_DEPTH / 2 - 0.5),
            scale=(450 * SCALE, 600 * SCALE, 1),
        ))
        # right wall
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(80, 80, 100),
            position=(shaft_x + 9 * SCALE, 0, CORRIDOR_DEPTH / 2 + 0.5),
            scale=(450 * SCALE, 600 * SCALE, 1),
        ))
        # protruding blocks
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(120, 90, 90),
            position=((650 + 180) * SCALE, (600 - 200) * SCALE / 2, 0),
            scale=(60 * SCALE, 40 * SCALE, CORRIDOR_DEPTH * 0.6),
        ))
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(120, 90, 90),
            position=((650 + 350) * SCALE, -(380 - 300) * SCALE, 0),
            scale=(70 * SCALE, 35 * SCALE, CORRIDOR_DEPTH * 0.6),
        ))

        # ─── Zone 4: Pipe Corridor ───
        pipe_x_base = 1100 * SCALE
        # ceiling
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 100, 80),
            position=(pipe_x_base + 35 * SCALE, 27 * SCALE, 0),
            scale=(700 * SCALE, 60 * SCALE, CORRIDOR_DEPTH),
        ))
        # floor
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 100, 80),
            position=(pipe_x_base + 35 * SCALE, -27 * SCALE, 0),
            scale=(700 * SCALE, 60 * SCALE, CORRIDOR_DEPTH),
        ))

        # terracotta pipes (horizontal cylinders)
        pipe_positions = [
            (1200, 150, 100), (1350, 350, 120), (1500, 200, 90),
            (1600, 420, 110), (1700, 280, 100),
        ]
        for px, py, pw in pipe_positions:
            # convert env coords: y is inverted in 3D
            y_3d = (300 - py) * SCALE
            self.level_entities.append(Entity(
                model='cube',
                color=color.rgb(200, 110, 80),
                position=(px * SCALE, y_3d, 0),
                scale=(pw * SCALE, 25 * SCALE, CORRIDOR_DEPTH * 0.8),
            ))

        # vertical bars
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 90, 70),
            position=(1380 * SCALE, 12 * SCALE, 0),
            scale=(20 * SCALE, 120 * SCALE, 20 * SCALE),
        ))
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 90, 70),
            position=(1520 * SCALE, -18 * SCALE, 0),
            scale=(20 * SCALE, 120 * SCALE, 20 * SCALE),
        ))

        # ─── Zone 5: Factory Hall ───
        hall_x = 1800 * SCALE
        # ceiling
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 110, 75),
            position=(hall_x + 40 * SCALE, 28 * SCALE, 0),
            scale=(800 * SCALE, 50 * SCALE, CORRIDOR_DEPTH),
        ))
        # floor
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(180, 110, 75),
            position=(hall_x + 40 * SCALE, -27 * SCALE, 0),
            scale=(800 * SCALE, 70 * SCALE, CORRIDOR_DEPTH),
        ))
        # pillars
        for px in [1900, 2100, 2300]:
            self.level_entities.append(Entity(
                model='cube',
                color=color.rgb(160, 90, 60),
                position=(px * SCALE, 5 * SCALE, CORRIDOR_DEPTH / 2 - 1),
                scale=(30 * SCALE, 100 * SCALE, 30 * SCALE),
            ))

        # conveyor belt
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(220, 200, 40),
            position=((1850 + 250) * SCALE, -20 * SCALE, 0),
            scale=(500 * SCALE, 15 * SCALE, 2),
        ))
        # boxes on conveyor
        for i in range(5):
            self.level_entities.append(Entity(
                model='cube',
                color=color.rgb(220, 140, 40),
                position=((1880 + i * 100) * SCALE, -18 * SCALE, 0),
                scale=(1.5, 1.5, 1.5),
            ))

        # workers (simple block people)
        for i in range(5):
            wx = (1870 + i * 100) * SCALE
            # body
            self.level_entities.append(Entity(
                model='cube',
                color=color.rgb(200, 60, 60),
                position=(wx, -16 * SCALE, -2),
                scale=(0.6, 1.2, 0.6),
            ))
            # head
            self.level_entities.append(Entity(
                model='sphere',
                color=color.rgb(220, 180, 140),
                position=(wx, -13 * SCALE, -2),
                scale=(0.5, 0.5, 0.5),
            ))
            # hard hat
            self.level_entities.append(Entity(
                model='cube',
                color=color.rgb(220, 160, 30),
                position=(wx, -12 * SCALE, -2),
                scale=(0.6, 0.2, 0.6),
            ))

        # ─── Zone 6: Boss Arena ───
        arena_x = 2500 * SCALE
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(160, 80, 60),
            position=(arena_x + 25 * SCALE, 28 * SCALE, 0),
            scale=(500 * SCALE, 50 * SCALE, CORRIDOR_DEPTH),
        ))
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(160, 80, 60),
            position=(arena_x + 25 * SCALE, -27 * SCALE, 0),
            scale=(500 * SCALE, 70 * SCALE, CORRIDOR_DEPTH),
        ))
        # boss platform
        self.level_entities.append(Entity(
            model='cube',
            color=color.rgb(160, 80, 60),
            position=(2880 * SCALE, -17 * SCALE, 0),
            scale=(120 * SCALE, 100 * SCALE, CORRIDOR_DEPTH * 0.8),
        ))

    def _build_rocket(self):
        """Create the 3D rocket model."""
        # body
        self.rocket_entity = Entity(
            model='cube',
            color=color.rgb(200, 220, 240),
            scale=(2, 0.5, 0.5),
        )
        # nose cone
        self.nose = Entity(
            model='cone',
            color=color.rgb(160, 160, 170),
            parent=self.rocket_entity,
            position=(0.7, 0, 0),
            rotation=(0, 0, -90),
            scale=(0.5, 0.6, 0.5),
        )
        # fins
        self.fin_top = Entity(
            model='cube',
            color=color.rgb(80, 80, 100),
            parent=self.rocket_entity,
            position=(-0.5, 0.3, 0),
            scale=(0.4, 0.3, 0.1),
        )
        self.fin_bottom = Entity(
            model='cube',
            color=color.rgb(80, 80, 100),
            parent=self.rocket_entity,
            position=(-0.5, -0.3, 0),
            scale=(0.4, 0.3, 0.1),
        )
        self.fin_side = Entity(
            model='cube',
            color=color.rgb(80, 80, 100),
            parent=self.rocket_entity,
            position=(-0.5, 0, 0.3),
            scale=(0.4, 0.1, 0.3),
        )

        # exhaust flame
        self.flame_entity = Entity(
            model='cube',
            color=color.rgb(255, 150, 30),
            parent=self.rocket_entity,
            position=(-1, 0, 0),
            scale=(0.8, 0.3, 0.3),
            enabled=False,
        )
        self.flame_inner = Entity(
            model='cube',
            color=color.rgb(255, 220, 80),
            parent=self.rocket_entity,
            position=(-0.7, 0, 0),
            scale=(0.5, 0.15, 0.15),
            enabled=False,
        )

    def _build_boss(self):
        """Create the boss character."""
        bx = 2900 * SCALE
        by = -10 * SCALE

        # body
        self.boss_entity = Entity(
            model='cube',
            color=color.rgb(50, 160, 80),
            position=(bx, by, 0),
            scale=(1, 2, 0.8),
        )
        # head
        Entity(
            model='sphere',
            color=color.rgb(220, 180, 140),
            parent=self.boss_entity,
            position=(0, 1.3, 0),
            scale=(0.7, 0.7, 0.7),
        )
        # beard
        Entity(
            model='cube',
            color=color.rgb(100, 70, 40),
            parent=self.boss_entity,
            position=(0, 1.0, 0.3),
            scale=(0.5, 0.4, 0.3),
        )
        # legs
        Entity(
            model='cube',
            color=color.rgb(50, 50, 70),
            parent=self.boss_entity,
            position=(-0.2, -1.2, 0),
            scale=(0.25, 0.8, 0.25),
        )
        Entity(
            model='cube',
            color=color.rgb(50, 50, 70),
            parent=self.boss_entity,
            position=(0.2, -1.2, 0),
            scale=(0.25, 0.8, 0.25),
        )
        # arms
        self.boss_arm = Entity(
            model='cube',
            color=color.rgb(50, 160, 80),
            parent=self.boss_entity,
            position=(0.5, 0.6, 0),
            rotation=(0, 0, -45),
            scale=(0.2, 0.8, 0.2),
        )
        Entity(
            model='cube',
            color=color.rgb(50, 160, 80),
            parent=self.boss_entity,
            position=(-0.5, 0.2, 0),
            rotation=(0, 0, 30),
            scale=(0.2, 0.7, 0.2),
        )

        # "BOSS" label
        self.boss_label = Text(
            text="BOSS",
            position=(0, 0.4),
            origin=(0, 0),
            scale=2,
            color=color.white,
            parent=self.boss_entity,
            billboard=True,
        )

    def _build_hud(self):
        """HUD text overlay."""
        self.hud_text = Text(
            text="",
            position=(-0.85, 0.45),
            scale=1.2,
            color=color.rgb(220, 220, 220),
        )
        self.progress_text = Text(
            text="",
            position=(0, -0.45),
            origin=(0, 0),
            scale=1.5,
            color=color.rgb(100, 255, 150),
        )

    def update_rocket(self, missile):
        """Update rocket position and rotation from env state."""
        # convert env coords to 3D: env x → 3D x, env y → 3D y (inverted)
        x_3d = missile.x * SCALE
        y_3d = (300 - missile.y) * SCALE  # 300 is center of env height
        z_3d = 0

        self.rocket_entity.position = Vec3(x_3d, y_3d, z_3d)

        # rotation from heading (env heading: 0=right, positive=clockwise)
        angle_deg = -math.degrees(missile.heading)
        self.rocket_entity.rotation = Vec3(0, 0, angle_deg)

        # flame visibility
        if missile.thrusting and missile.speed > 0.1:
            self.flame_entity.enabled = True
            self.flame_inner.enabled = True
            # flickering flame size
            import random
            flame_scale = 0.6 + missile.speed * 0.15 + random.uniform(-0.1, 0.1)
            self.flame_entity.scale_x = flame_scale
            self.flame_inner.scale_x = flame_scale * 0.6
        else:
            self.flame_entity.enabled = False
            self.flame_inner.enabled = False

        # boss arm wave
        if self.boss_arm:
            wave = math.sin(self.app_time() * 3) * 20
            self.boss_arm.rotation_z = -45 + wave

        # camera follows rocket (third-person)
        cam_offset = Vec3(-8, 4, -12)
        camera.position = self.rocket_entity.position + cam_offset
        camera.look_at(self.rocket_entity.position + Vec3(5, 0, 0))

    def update_hud(self, step, missile, boss_pos):
        """Update HUD text."""
        dist = math.sqrt(
            (missile.x - boss_pos[0])**2 + (missile.y - boss_pos[1])**2
        )
        progress = min(100, missile.x / 3000 * 100)

        self.hud_text.text = (
            f"Step: {step}  |  Speed: {missile.speed:.1f}  |  "
            f"Boss Dist: {dist:.0f}"
        )
        self.progress_text.text = f"Progress: {progress:.0f}%"

    def step(self):
        """Advance one frame of the Ursina app."""
        self.app.step()

    def app_time(self):
        """Get app elapsed time."""
        try:
            return time.time()
        except Exception:
            return 0

    def close(self):
        """Cleanup."""
        try:
            application.quit()
        except Exception:
            pass
