import math
import random


class Obstacle:
    """A circular obstacle at (cx, cy) with given radius."""

    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius


def generate_obstacles(n, width, height, missile_pos, target_pos,
                       min_gap=80, min_radius=15, max_radius=35, rng=None):
    """
    Generate n non-overlapping circular obstacles.
    Avoids placing them on the missile start or the target.

    Args:
        n: number of obstacles to generate
        width, height: environment dimensions
        missile_pos: (x, y) of missile start
        target_pos: (x, y) of target
        min_gap: minimum gap between obstacle edges and missile/target
        rng: numpy random generator (optional, for reproducibility)
    """
    obstacles = []
    margin = 50  # keep obstacles away from walls

    for _ in range(n * 50):  # max attempts
        if len(obstacles) >= n:
            break

        if rng is not None:
            cx = rng.uniform(margin, width - margin)
            cy = rng.uniform(margin, height - margin)
            r = rng.uniform(min_radius, max_radius)
        else:
            cx = random.uniform(margin, width - margin)
            cy = random.uniform(margin, height - margin)
            r = random.uniform(min_radius, max_radius)

        # check distance to missile start
        dist_missile = math.sqrt((cx - missile_pos[0])**2 + (cy - missile_pos[1])**2)
        if dist_missile < r + min_gap:
            continue

        # check distance to target
        dist_target = math.sqrt((cx - target_pos[0])**2 + (cy - target_pos[1])**2)
        if dist_target < r + min_gap:
            continue

        # check overlap with existing obstacles
        overlaps = False
        for obs in obstacles:
            dist = math.sqrt((cx - obs.cx)**2 + (cy - obs.cy)**2)
            if dist < r + obs.radius + 20:  # 20px gap between obstacles
                overlaps = True
                break

        if not overlaps:
            obstacles.append(Obstacle(cx, cy, r))

    return obstacles


def check_obstacle_collision(mx, my, missile_radius, obstacles):
    """Check if a circle at (mx, my) with missile_radius hits any obstacle."""
    for obs in obstacles:
        dist = math.sqrt((mx - obs.cx)**2 + (my - obs.cy)**2)
        if dist < missile_radius + obs.radius:
            return True
    return False


def check_wall_collision(mx, my, width, height):
    """Check if missile is outside the play area."""
    return mx < 0 or mx > width or my < 0 or my > height


def check_target_reached(mx, my, target_pos, target_radius):
    """Check if missile is close enough to the target."""
    dist = math.sqrt((mx - target_pos[0])**2 + (my - target_pos[1])**2)
    return dist < target_radius
