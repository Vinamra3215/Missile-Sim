import math
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


class Renderer:
    """Pygame-based renderer for the missile environment."""

    # colors
    BG_COLOR = (15, 15, 30)
    GRID_COLOR = (30, 30, 50)
    MISSILE_COLOR = (0, 200, 255)
    TRAIL_COLOR = (0, 100, 180)
    TARGET_COLOR = (0, 255, 100)
    TARGET_PULSE_COLOR = (100, 255, 180)
    OBSTACLE_COLOR = (200, 50, 50)
    OBSTACLE_BORDER = (255, 100, 100)
    HUD_COLOR = (200, 200, 200)
    HUD_BG = (0, 0, 0, 120)

    def __init__(self, width, height, render_mode="human"):
        if pygame is None:
            raise ImportError("pygame is required for rendering: pip install pygame")

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
        self.font = pygame.font.SysFont("consolas", 16)
        self.frame_count = 0

    def draw(self, missile, target, obstacles, step=0, dist=0.0):
        """Draw one frame of the environment."""
        self.frame_count += 1

        # handle pygame events (so window doesn't freeze)
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # background
        self.screen.fill(self.BG_COLOR)
        self._draw_grid()

        # obstacles
        for obs in obstacles:
            pygame.draw.circle(
                self.screen, self.OBSTACLE_COLOR,
                (int(obs.cx), int(obs.cy)), int(obs.radius)
            )
            pygame.draw.circle(
                self.screen, self.OBSTACLE_BORDER,
                (int(obs.cx), int(obs.cy)), int(obs.radius), 2
            )

        # target with pulse effect
        pulse = abs(math.sin(self.frame_count * 0.05)) * 8
        tx, ty = int(target[0]), int(target[1])
        pygame.draw.circle(
            self.screen, self.TARGET_PULSE_COLOR,
            (tx, ty), int(20 + pulse), 2
        )
        pygame.draw.circle(self.screen, self.TARGET_COLOR, (tx, ty), 12)
        pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), 4)

        # missile trail (fading dots)
        if missile.trail:
            n = len(missile.trail)
            for i, (px, py) in enumerate(missile.trail):
                alpha = int(255 * (i / n) * 0.4)
                r = max(1, int(3 * (i / n)))
                color = (0, min(100, 50 + alpha), min(180, 80 + alpha))
                pygame.draw.circle(self.screen, color, (int(px), int(py)), r)

        # missile body (triangle pointing in heading direction)
        mx, my = missile.x, missile.y
        h = missile.heading
        size = 14

        # triangle vertices
        tip = (mx + size * math.cos(h), my + size * math.sin(h))
        left = (mx + size * 0.6 * math.cos(h + 2.5), my + size * 0.6 * math.sin(h + 2.5))
        right = (mx + size * 0.6 * math.cos(h - 2.5), my + size * 0.6 * math.sin(h - 2.5))

        points = [(int(p[0]), int(p[1])) for p in [tip, left, right]]
        pygame.draw.polygon(self.screen, self.MISSILE_COLOR, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 1)

        # thrust flame effect when moving
        if missile.speed > 0.5:
            flame_len = missile.speed * 3
            back = (mx - flame_len * math.cos(h), my - flame_len * math.sin(h))
            pygame.draw.line(
                self.screen, (255, 150, 0),
                (int(mx), int(my)), (int(back[0]), int(back[1])), 2
            )

        # HUD overlay
        self._draw_hud(step, dist, missile.speed)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
            return None
        else:
            # return pixel array for recording
            return np.transpose(
                pygame.surfarray.array3d(self.screen), (1, 0, 2)
            )

    def _draw_grid(self):
        """Subtle background grid."""
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width, y))

    def _draw_hud(self, step, dist, speed):
        """Draw heads-up display with stats."""
        texts = [
            f"Step: {step}",
            f"Dist: {dist:.0f}",
            f"Speed: {speed:.1f}",
        ]
        y_offset = 10
        for text in texts:
            surface = self.font.render(text, True, self.HUD_COLOR)
            bg_rect = pygame.Rect(8, y_offset - 2, surface.get_width() + 8, 22)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 140))
            self.screen.blit(bg_surface, bg_rect)
            self.screen.blit(surface, (12, y_offset))
            y_offset += 24

    def close(self):
        if pygame is not None:
            pygame.quit()
