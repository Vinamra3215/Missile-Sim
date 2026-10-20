import math
from configs.default import MISSILE_MAX_SPEED, MISSILE_TURN_RATE, MISSILE_THRUST


class Missile:
    """
    2D missile with heading-based movement.
    The missile has a position, speed, and heading angle.
    Actions change the heading or speed each step.
    """

    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading   # radians, 0 = right, pi/2 = down
        self.speed = 0.0
        self.vx = 0.0
        self.vy = 0.0

        # keep a trail for rendering
        self.trail = []
        self.max_trail = 60

    def update(self, action):
        """
        Apply one of 5 actions and update physics.
        0 = coast (do nothing)
        1 = turn left (increase heading)
        2 = turn right (decrease heading)
        3 = accelerate (increase speed)
        4 = decelerate (decrease speed)
        """
        # adjust heading
        if action == 1:
            self.heading += MISSILE_TURN_RATE
        elif action == 2:
            self.heading -= MISSILE_TURN_RATE

        # keep heading in [-pi, pi]
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # adjust speed
        if action == 3:
            self.speed = min(self.speed + MISSILE_THRUST, MISSILE_MAX_SPEED)
        elif action == 4:
            self.speed = max(self.speed - MISSILE_THRUST, 0.0)

        # compute velocity from heading and speed
        self.vx = self.speed * math.cos(self.heading)
        self.vy = self.speed * math.sin(self.heading)

        # update position
        self.x += self.vx
        self.y += self.vy

        # save trail point
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def get_pos(self):
        return self.x, self.y
