"""
Renderer for MissileSim — dark, atmospheric, voxel-inspired style.
Side-scrolling camera that follows the rocket through the level zones.
"""
import math
import random

try:
    import pygame
    import pygame.gfxdraw
except ImportError:
    pygame = None


class Renderer:
    """Pygame renderer with scrolling camera and zone-aware visuals."""

    # colors
    MISSILE_BODY = (200, 220, 240)
    MISSILE_NOSE = (160, 160, 170)
    MISSILE_FIN = (80, 80, 100)
    FLAME_INNER = (255, 200, 50)
    FLAME_OUTER = (255, 120, 20)
    TRAIL_COLOR = (60, 120, 200)
    BOSS_BODY = (50, 160, 80)
    BOSS_LABEL = (255, 255, 255)
    HUD_TEXT = (220, 220, 220)
    GRID_ALPHA = 30
    CONVEYOR_YELLOW = (220, 200, 40)
    CONVEYOR_BLACK = (30, 30, 30)
    WORKER_BODY = (200, 60, 60)
    WORKER_PANTS = (50, 50, 70)
    BOX_COLOR = (220, 140, 40)

    def __init__(self, width, height, render_mode="human"):
        if pygame is None:
            raise ImportError("pygame is required")

        self.width = width
        self.height = height
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()

        if render_mode == "human":
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("MissileSim — RL Missile Navigation")
        else:
            self.screen = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font_hud = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_zone = pygame.font.SysFont("arial", 20, bold=True)
        self.font_boss = pygame.font.SysFont("arial", 14, bold=True)
        self.frame = 0
        self.particles = []

    def draw(self, missile, boss_pos, zones, step=0, camera_x=0):
        self.frame += 1

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        cam_x = max(0, camera_x)

        # ─── Background & Zones ───
        self.screen.fill((10, 10, 25))

        for zone in zones:
            zx = zone.x_start - cam_x
            zw = zone.x_end - zone.x_start

            if zx + zw < 0 or zx > self.width:
                continue

            # zone background
            zone_rect = pygame.Rect(int(zx), 0, int(zw), self.height)
            pygame.draw.rect(self.screen, zone.bg_color, zone_rect)

            # floor
            if zone.floor_y is not None:
                floor_rect = pygame.Rect(
                    int(zx), int(zone.floor_y), int(zw),
                    self.height - int(zone.floor_y)
                )
                pygame.draw.rect(self.screen, zone.wall_color, floor_rect)
                # floor line
                pygame.draw.line(
                    self.screen, self._lighten(zone.wall_color, 40),
                    (int(zx), int(zone.floor_y)),
                    (int(zx + zw), int(zone.floor_y)), 2
                )

            # ceiling
            if zone.ceil_y is not None:
                ceil_rect = pygame.Rect(int(zx), 0, int(zw), int(zone.ceil_y))
                pygame.draw.rect(self.screen, zone.wall_color, ceil_rect)
                pygame.draw.line(
                    self.screen, self._lighten(zone.wall_color, 40),
                    (int(zx), int(zone.ceil_y)),
                    (int(zx + zw), int(zone.ceil_y)), 2
                )

            # grid overlay
            self._draw_zone_grid(zx, zw, zone)

            # obstacles in this zone
            for obs in zone.obstacles:
                self._draw_obstacle(obs, cam_x, zone.wall_color)

            # zone label
            if 0 < zx + zw / 2 < self.width:
                label = self.font_zone.render(zone.name, True, (255, 255, 255, 100))
                label.set_alpha(60)
                self.screen.blit(label, (int(zx + zw / 2 - label.get_width() / 2), 8))

        # ─── Factory Hall decorations ───
        self._draw_factory_details(cam_x)

        # ─── Boss ───
        self._draw_boss(boss_pos, cam_x)

        # ─── Particles ───
        self._update_particles(cam_x)

        # ─── Missile Trail ───
        if missile.trail:
            n = len(missile.trail)
            for i in range(1, n):
                px, py = missile.trail[i]
                sx = int(px - cam_x)
                sy = int(py)
                if 0 <= sx < self.width:
                    alpha_f = i / n
                    r = max(1, int(3 * alpha_f))
                    color = (
                        int(30 + 40 * alpha_f),
                        int(80 + 80 * alpha_f),
                        int(150 + 60 * alpha_f)
                    )
                    pygame.draw.circle(self.screen, color, (sx, sy), r)

        # ─── Missile Body ───
        self._draw_missile(missile, cam_x)

        # ─── HUD ───
        self._draw_hud(step, missile, boss_pos)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
            return None
        else:
            import numpy as np
            return np.transpose(
                pygame.surfarray.array3d(self.screen), (1, 0, 2)
            )

    def _draw_missile(self, missile, cam_x):
        mx = missile.x - cam_x
        my = missile.y
        h = missile.heading
        if mx < -50 or mx > self.width + 50:
            return

        # exhaust flame
        if missile.thrusting and missile.speed > 0.1:
            flame_len = 12 + missile.speed * 3 + random.uniform(-3, 3)
            flame_w = 5 + random.uniform(-1, 1)
            back_x = mx - flame_len * math.cos(h)
            back_y = my - flame_len * math.sin(h)
            mid_x = mx - flame_len * 0.5 * math.cos(h)
            mid_y = my - flame_len * 0.5 * math.sin(h)

            # outer flame
            pts_outer = [
                (int(mx - 4 * math.cos(h + 1.5)), int(my - 4 * math.sin(h + 1.5))),
                (int(mx - 4 * math.cos(h - 1.5)), int(my - 4 * math.sin(h - 1.5))),
                (int(back_x), int(back_y)),
            ]
            pygame.draw.polygon(self.screen, self.FLAME_OUTER, pts_outer)

            # inner flame
            inner_len = flame_len * 0.6
            inner_x = mx - inner_len * math.cos(h)
            inner_y = my - inner_len * math.sin(h)
            pts_inner = [
                (int(mx - 2 * math.cos(h + 1.5)), int(my - 2 * math.sin(h + 1.5))),
                (int(mx - 2 * math.cos(h - 1.5)), int(my - 2 * math.sin(h - 1.5))),
                (int(inner_x), int(inner_y)),
            ]
            pygame.draw.polygon(self.screen, self.FLAME_INNER, pts_inner)

            # spawn particles
            for _ in range(2):
                self.particles.append({
                    'x': back_x + cam_x + random.uniform(-5, 5),
                    'y': back_y + random.uniform(-5, 5),
                    'vx': -missile.vx * 0.3 + random.uniform(-1, 1),
                    'vy': -missile.vy * 0.3 + random.uniform(-1, 1),
                    'life': random.randint(8, 20),
                    'size': random.randint(2, 4),
                })

        # body (triangle shape)
        size = 16
        tip = (mx + size * math.cos(h), my + size * math.sin(h))
        left = (mx + size * 0.55 * math.cos(h + 2.6), my + size * 0.55 * math.sin(h + 2.6))
        right = (mx + size * 0.55 * math.cos(h - 2.6), my + size * 0.55 * math.sin(h - 2.6))
        back_c = (mx - size * 0.4 * math.cos(h), my - size * 0.4 * math.sin(h))

        # fins
        fin_l = (mx + size * 0.7 * math.cos(h + 2.8), my + size * 0.7 * math.sin(h + 2.8))
        fin_r = (mx + size * 0.7 * math.cos(h - 2.8), my + size * 0.7 * math.sin(h - 2.8))
        pygame.draw.line(self.screen, self.MISSILE_FIN,
                         (int(left[0]), int(left[1])), (int(fin_l[0]), int(fin_l[1])), 3)
        pygame.draw.line(self.screen, self.MISSILE_FIN,
                         (int(right[0]), int(right[1])), (int(fin_r[0]), int(fin_r[1])), 3)

        # main body
        pts = [(int(p[0]), int(p[1])) for p in [tip, left, back_c, right]]
        pygame.draw.polygon(self.screen, self.MISSILE_BODY, pts)
        pygame.draw.polygon(self.screen, (255, 255, 255), pts, 1)

        # nose cone
        nose_pts = [(int(tip[0]), int(tip[1])),
                     (int(mx + 6 * math.cos(h + 0.5)), int(my + 6 * math.sin(h + 0.5))),
                     (int(mx + 6 * math.cos(h - 0.5)), int(my + 6 * math.sin(h - 0.5)))]
        pygame.draw.polygon(self.screen, self.MISSILE_NOSE, nose_pts)

    def _draw_obstacle(self, obs, cam_x, color):
        if hasattr(obs, 'w'):  # Rect
            rx = int(obs.x - cam_x)
            ry = int(obs.y)
            if rx + obs.w < 0 or rx > self.width:
                return
            rect = pygame.Rect(rx, ry, int(obs.w), int(obs.h))
            pygame.draw.rect(self.screen, color, rect)
            # highlight edge
            pygame.draw.rect(self.screen, self._lighten(color, 30), rect, 1)
            # inner shadow
            if obs.w > 20 and obs.h > 10:
                inner = pygame.Rect(rx + 3, ry + 3, int(obs.w) - 6, int(obs.h) - 6)
                pygame.draw.rect(self.screen, self._darken(color, 30), inner, 1)
        else:  # Pipe
            sx1 = int(obs.x1 - cam_x)
            sy1 = int(obs.y1)
            sx2 = int(obs.x2 - cam_x)
            sy2 = int(obs.y2)
            pygame.draw.line(self.screen, color, (sx1, sy1), (sx2, sy2),
                             int(obs.radius * 2))
            pygame.draw.line(self.screen, self._lighten(color, 40),
                             (sx1, sy1), (sx2, sy2), 2)

    def _draw_boss(self, boss_pos, cam_x):
        bx = int(boss_pos[0] - cam_x)
        by = int(boss_pos[1])
        if bx < -50 or bx > self.width + 50:
            return

        # body
        pygame.draw.rect(self.screen, self.BOSS_BODY,
                         (bx - 12, by - 10, 24, 35))
        # head
        pygame.draw.circle(self.screen, (220, 180, 140), (bx, by - 20), 12)
        # beard
        pygame.draw.rect(self.screen, (100, 70, 40), (bx - 8, by - 14, 16, 8))
        # arms (waving)
        arm_angle = math.sin(self.frame * 0.1) * 0.3
        arm_x = bx + int(15 * math.cos(-0.8 + arm_angle))
        arm_y = by - 5 + int(15 * math.sin(-0.8 + arm_angle))
        pygame.draw.line(self.screen, self.BOSS_BODY, (bx + 10, by), (arm_x, arm_y), 3)
        pygame.draw.line(self.screen, self.BOSS_BODY, (bx - 10, by), (bx - 18, by + 10), 3)
        # legs
        pygame.draw.line(self.screen, (50, 50, 70), (bx - 5, by + 25), (bx - 8, by + 40), 3)
        pygame.draw.line(self.screen, (50, 50, 70), (bx + 5, by + 25), (bx + 8, by + 40), 3)

        # "BOSS" label
        pulse = abs(math.sin(self.frame * 0.08)) * 3
        label = self.font_boss.render("BOSS", True, self.BOSS_LABEL)
        self.screen.blit(label, (bx - label.get_width() // 2, by - 45 - int(pulse)))

    def _draw_factory_details(self, cam_x):
        """Draw conveyor belt, boxes, and workers in Factory Hall zone."""
        hall_start = 1800

        # conveyor belt stripes
        for i in range(20):
            cx = hall_start + 1850 - hall_start + i * 25
            sx = int(cx - cam_x)
            if 0 <= sx < self.width:
                belt_y = self.height - 110
                color = self.CONVEYOR_YELLOW if i % 2 == 0 else self.CONVEYOR_BLACK
                pygame.draw.rect(self.screen, color, (sx, belt_y, 12, 15))

        # boxes on conveyor
        for i in range(4):
            bx = int(hall_start + 80 + i * 120 - cam_x)
            by = self.height - 130
            if 0 <= bx < self.width:
                pygame.draw.rect(self.screen, self.BOX_COLOR, (bx, by, 18, 18))
                pygame.draw.rect(self.screen, (180, 110, 20), (bx, by, 18, 18), 1)

        # workers
        for i in range(5):
            wx = int(hall_start + 60 + i * 100 - cam_x)
            wy = self.height - 145
            if 0 <= wx < self.width:
                # body
                pygame.draw.rect(self.screen, self.WORKER_BODY, (wx, wy, 10, 14))
                # head
                pygame.draw.circle(self.screen, (220, 180, 140), (wx + 5, wy - 5), 5)
                # hard hat
                pygame.draw.rect(self.screen, (220, 160, 30), (wx + 1, wy - 10, 8, 4))
                # legs
                pygame.draw.line(self.screen, self.WORKER_PANTS,
                                 (wx + 3, wy + 14), (wx + 1, wy + 22), 2)
                pygame.draw.line(self.screen, self.WORKER_PANTS,
                                 (wx + 7, wy + 14), (wx + 9, wy + 22), 2)

    def _draw_zone_grid(self, zx, zw, zone):
        """Subtle grid overlay per zone."""
        spacing = 40
        # vertical lines
        start_x = int(zx) - (int(zx) % spacing)
        for x in range(start_x, int(zx + zw), spacing):
            if 0 <= x < self.width:
                s = pygame.Surface((1, self.height), pygame.SRCALPHA)
                s.fill((*self._lighten(zone.bg_color, 15), 25))
                self.screen.blit(s, (x, 0))
        # horizontal lines
        for y in range(0, self.height, spacing):
            s = pygame.Surface((min(int(zw), self.width), 1), pygame.SRCALPHA)
            s.fill((*self._lighten(zone.bg_color, 15), 25))
            self.screen.blit(s, (max(0, int(zx)), y))

    def _update_particles(self, cam_x):
        """Update and draw exhaust particles."""
        alive = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] > 0:
                sx = int(p['x'] - cam_x)
                sy = int(p['y'])
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    alpha = min(255, p['life'] * 20)
                    color = (
                        min(255, 200 + p['life'] * 3),
                        min(255, 100 + p['life'] * 5),
                        30
                    )
                    pygame.draw.circle(self.screen, color, (sx, sy), p['size'])
                alive.append(p)
        self.particles = alive[:200]  # cap particles

    def _draw_hud(self, step, missile, boss_pos):
        """Heads-up display."""
        dist = math.sqrt(
            (missile.x - boss_pos[0])**2 + (missile.y - boss_pos[1])**2
        )
        progress = min(100, missile.x / 3000 * 100)

        texts = [
            f"Step: {step}",
            f"Speed: {missile.speed:.1f}",
            f"Boss Dist: {dist:.0f}",
            f"Progress: {progress:.0f}%",
        ]

        # semi-transparent HUD background
        hud_w = 160
        hud_h = len(texts) * 22 + 10
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 150))
        self.screen.blit(hud_surface, (6, 6))

        for i, text in enumerate(texts):
            surface = self.font_hud.render(text, True, self.HUD_TEXT)
            self.screen.blit(surface, (12, 12 + i * 22))

        # progress bar at bottom
        bar_w = self.width - 40
        bar_h = 6
        bar_x = 20
        bar_y = self.height - 15
        # background
        pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, bar_y, bar_w, bar_h))
        # fill
        fill_w = int(bar_w * progress / 100)
        color = (50, 200, 100) if progress < 90 else (255, 200, 50)
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))
        # checkpoints
        for cp in [400, 650, 1100, 1800, 2500]:
            cpx = bar_x + int(bar_w * cp / 3000)
            pygame.draw.line(self.screen, (200, 200, 200),
                             (cpx, bar_y - 2), (cpx, bar_y + bar_h + 2), 1)

    @staticmethod
    def _lighten(color, amount):
        return tuple(min(255, c + amount) for c in color[:3])

    @staticmethod
    def _darken(color, amount):
        return tuple(max(0, c - amount) for c in color[:3])

    def close(self):
        if pygame is not None:
            pygame.quit()
