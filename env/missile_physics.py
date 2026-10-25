import math
from configs.default import MISSILE_MAX_SPEED, MISSILE_TURN_RATE, MISSILE_THRUST, MISSILE_DRAG


class Missile:
    """
    2D rocket with heading-based thrust, drag, and rotation.
    Modeled after a voxel-style rocket: body, nose cone, fins, exhaust flame.
    """

    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.thrusting = False

        # trail for rendering
        self.trail = []
        self.max_trail = 80

    def update(self, action):
        """
        0 = coast, 1 = turn left, 2 = turn right,
        3 = accelerate, 4 = decelerate,
        5 = thrust + turn left, 6 = thrust + turn right
        """
        self.thrusting = False

        # parse combined actions
        turn = 0
        thrust = 0

        if action == 1:
            turn = 1
        elif action == 2:
            turn = -1
        elif action == 3:
            thrust = 1
        elif action == 4:
            thrust = -1
        elif action == 5:
            turn = 1
            thrust = 1
        elif action == 6:
            turn = -1
            thrust = 1

        # rotate
        if turn != 0:
            self.heading += turn * MISSILE_TURN_RATE
            self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi

        # thrust
        if thrust > 0:
            self.speed = min(self.speed + MISSILE_THRUST, MISSILE_MAX_SPEED)
            self.thrusting = True
        elif thrust < 0:
            self.speed = max(self.speed - MISSILE_THRUST * 1.5, 0.0)

        # drag
        self.speed *= MISSILE_DRAG

        # velocity from heading
        self.vx = self.speed * math.cos(self.heading)
        self.vy = self.speed * math.sin(self.heading)

        # update position
        self.x += self.vx
        self.y += self.vy

        # trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def get_pos(self):
        return self.x, self.y
