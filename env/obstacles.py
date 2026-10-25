"""
Stage-based level layout inspired by the reference video.
Zones: Launch Pad → Chimney Entry → Vertical Shaft → Pipe Corridor → Factory Hall → Boss Arena
"""
import math


class Rect:
    """Axis-aligned rectangle obstacle."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def collides_circle(self, cx, cy, r):
        # nearest point on rect to circle center
        nx = max(self.x, min(cx, self.x + self.w))
        ny = max(self.y, min(cy, self.y + self.h))
        dx = cx - nx
        dy = cy - ny
        return (dx * dx + dy * dy) < (r * r)


class Pipe:
    """Horizontal or vertical cylindrical pipe obstacle."""
    def __init__(self, x1, y1, x2, y2, radius):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.radius = radius

    def collides_circle(self, cx, cy, r):
        # distance from point to line segment
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            dist = math.sqrt((cx - self.x1)**2 + (cy - self.y1)**2)
        else:
            t = max(0, min(1, ((cx - self.x1)*dx + (cy - self.y1)*dy) / length_sq))
            proj_x = self.x1 + t * dx
            proj_y = self.y1 + t * dy
            dist = math.sqrt((cx - proj_x)**2 + (cy - proj_y)**2)
        return dist < (self.radius + r)


class Zone:
    """A zone/stage in the level with its own obstacles and visual style."""
    def __init__(self, name, x_start, x_end, bg_color, wall_color,
                 floor_y=None, ceil_y=None, obstacles=None):
        self.name = name
        self.x_start = x_start
        self.x_end = x_end
        self.bg_color = bg_color
        self.wall_color = wall_color
        self.floor_y = floor_y       # y-coordinate of floor (None = no floor)
        self.ceil_y = ceil_y         # y-coordinate of ceiling (None = no ceiling)
        self.obstacles = obstacles or []


def build_level(env_height):
    """
    Build the complete level layout.
    Returns (zones, boss_pos, checkpoints).
    """
    H = env_height  # 600

    zones = []
    all_obstacles = []

    # ─── Zone 1: Launch Pad (outdoor) ───
    # Open green field, blue sky, no obstacles
    zones.append(Zone(
        name="Launch Pad",
        x_start=0, x_end=400,
        bg_color=(20, 20, 50),      # dark navy sky
        wall_color=(40, 160, 60),   # green ground
        floor_y=H - 80,
        ceil_y=None,
        obstacles=[]
    ))

    # ─── Zone 2: Chimney Entry ───
    # Narrow opening at top, walls on sides
    chimney_gap = 120  # width of the opening
    chimney_center_y = 200
    zones.append(Zone(
        name="Chimney",
        x_start=400, x_end=650,
        bg_color=(15, 15, 40),
        wall_color=(100, 100, 110),
        floor_y=None,
        ceil_y=None,
        obstacles=[
            # top wall (above chimney opening)
            Rect(400, 0, 250, chimney_center_y - chimney_gap // 2),
            # bottom wall (below chimney opening)
            Rect(400, chimney_center_y + chimney_gap // 2, 250,
                 H - chimney_center_y - chimney_gap // 2),
        ]
    ))
    all_obstacles.extend(zones[-1].obstacles)

    # ─── Zone 3: Shaft Descent ───
    # Walls narrowing, a few protruding blocks
    shaft_left = 180
    shaft_right = 420
    shaft_obs = [
        # left wall
        Rect(650, 0, shaft_left, H),
        # right wall
        Rect(650 + shaft_right, 0, 400 - shaft_right + 200, H),
        # protruding block from left
        Rect(650 + shaft_left, 200, 60, 40),
        # protruding block from right
        Rect(650 + shaft_right - 70, 380, 70, 35),
    ]
    zones.append(Zone(
        name="Shaft",
        x_start=650, x_end=1100,
        bg_color=(10, 10, 35),
        wall_color=(80, 80, 100),
        floor_y=None,
        ceil_y=None,
        obstacles=shaft_obs
    ))
    all_obstacles.extend(shaft_obs)

    # ─── Zone 4: Pipe Corridor ───
    # Horizontal terracotta pipes at different heights
    pipe_obs = [
        # top & bottom walls
        Rect(1100, 0, 700, 60),
        Rect(1100, H - 60, 700, 60),
        # pipes at various heights
        Pipe(1200, 150, 1200, 150, 1),  # anchor point
        Rect(1160, 140, 100, 25),       # pipe 1
        Rect(1300, 350, 120, 25),       # pipe 2
        Rect(1450, 200, 90, 25),        # pipe 3
        Rect(1550, 420, 110, 25),       # pipe 4
        Rect(1650, 280, 100, 25),       # pipe 5
        # some vertical bars
        Rect(1380, 60, 20, 120),
        Rect(1520, H - 180, 20, 120),
    ]
    zones.append(Zone(
        name="Pipe Corridor",
        x_start=1100, x_end=1800,
        bg_color=(12, 12, 30),
        wall_color=(180, 100, 80),  # terracotta
        floor_y=H - 60,
        ceil_y=60,
        obstacles=pipe_obs
    ))
    all_obstacles.extend(pipe_obs)

    # ─── Zone 5: Factory Hall ───
    # Conveyor area, wider corridor, some pillars
    hall_obs = [
        # floor and ceiling
        Rect(1800, 0, 800, 50),
        Rect(1800, H - 70, 800, 70),
        # pillars
        Rect(1900, 50, 30, 100),
        Rect(2100, H - 170, 30, 100),
        Rect(2300, 50, 30, 120),
        # conveyor belt (visual + slight obstacle)
        Rect(1850, H - 110, 500, 15),
    ]
    zones.append(Zone(
        name="Factory Hall",
        x_start=1800, x_end=2500,
        bg_color=(15, 12, 10),
        wall_color=(180, 110, 75),  # terracotta
        floor_y=H - 70,
        ceil_y=50,
        obstacles=hall_obs
    ))
    all_obstacles.extend(hall_obs)

    # ─── Zone 6: Boss Arena ───
    boss_obs = [
        Rect(2500, 0, 500, 50),
        Rect(2500, H - 70, 500, 70),
        # boss platform
        Rect(2850, H - 170, 120, 100),
    ]
    zones.append(Zone(
        name="Boss Arena",
        x_start=2500, x_end=3000,
        bg_color=(20, 10, 10),
        wall_color=(160, 80, 60),
        floor_y=H - 70,
        ceil_y=50,
        obstacles=boss_obs
    ))
    all_obstacles.extend(boss_obs)

    # boss position (on top of the platform)
    boss_pos = (2900, H - 200)

    # checkpoints (x positions that give bonus reward)
    checkpoints = [400, 650, 1100, 1800, 2500]

    return zones, all_obstacles, boss_pos, checkpoints


def check_obstacle_collision(mx, my, radius, obstacles):
    """Check if missile circle collides with any obstacle."""
    for obs in obstacles:
        if obs.collides_circle(mx, my, radius):
            return True
    return False


def check_wall_collision(mx, my, env_height):
    """Check if missile is off-screen vertically."""
    return my < 0 or my > env_height
