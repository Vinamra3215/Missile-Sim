"""
Configuration for the 3D tunnel environment.

All lengths are in arbitrary "world units" (think metres), time in seconds.
The tunnel runs along the +x axis; its square cross-section spans
y in [-tunnel_half, tunnel_half] and z in [-tunnel_half, tunnel_half].
"""
import math
from dataclasses import dataclass


@dataclass
class EnvConfig:
    # ── simulation ──
    num_envs: int = 128             # parallel environments, simulated together as one batch
    dt: float = 0.05                # seconds per step (20 Hz control)
    time_limit_factor: float = 1.8  # episode limit = factor * (length / min speed)

    # ── tunnel / map generation (procedural, new map every episode) ──
    tunnel_half: float = 8.0        # half of tunnel width/height
    len_min: float = 60.0           # tunnel length at curriculum level 0
    len_max: float = 220.0          # tunnel length at curriculum level 1
    max_level: float = 2.0          # levels 1..2 are the "expert" tier
    len_expert: float = 400.0       # tunnel length at level 2
    end_gap: float = 6.0            # space between target and back wall
    first_pipe_x: float = 18.0      # no pipes before this x (launch zone)
    pipe_min_spacing: float = 9.0   # min distance between consecutive pipes
    max_pipes: int = 32             # obstacle slots (inactive slots are masked)
    obstacles_l1: int = 16          # obstacle count at level 1 (keeps growing in expert tier)
    pipes_min: int = 2              # obstacles at level 0
    pipe_r_min: float = 0.8
    pipe_r_max: float = 2.6
    pipe_offset_frac: float = 0.65  # how far from the centre a pipe may pass
    pipe_tilt_max: float = 0.35     # max tilt of a pipe towards the x axis (rad)

    # moving pipes (appear from level `move_start`)
    move_start: float = 0.4
    move_frac_max: float = 0.6      # fraction of pipes that move at level 1
    move_amp_max: float = 3.0       # oscillation amplitude
    move_freq_min: float = 0.6      # rad/s
    move_freq_max: float = 1.6
    expert_move_frac: float = 0.9   # values reached at level 2
    expert_move_amp: float = 4.5
    expert_move_freq: float = 2.4

    # gates: walls with one circular hole (appear from level `gate_start`)
    gate_start: float = 0.3
    gate_prob_max: float = 0.35     # chance an obstacle slot is a gate at level 1
    gate_thickness: float = 0.6
    gate_hole_max: float = 4.5      # hole radius at gate_start
    gate_hole_min: float = 2.6      # hole radius at level 1
    expert_gate_prob: float = 0.5
    expert_gate_hole: float = 2.0

    # ── target ──
    target_radius: float = 1.8
    target_margin: float = 3.0      # keep target this far from walls

    # ── missile (arcade flight model) ──
    missile_radius: float = 0.4
    speed_min: float = 8.0          # a missile can never stop
    speed_max: float = 22.0
    speed_init: float = 12.0
    accel: float = 12.0             # units/s^2 when boosting / braking
    turn_rate: float = 1.8          # rad/s for pitch and yaw
    pitch_limit: float = 1.2        # rad

    # ── sensors ──
    ray_grid: int = 5               # ray_grid x ray_grid "depth camera"
    ray_fov: float = math.radians(70)
    ray_range: float = 40.0

    # ── rewards ──
    r_progress: float = 0.1         # per unit of distance closed to target
    r_step: float = -0.01           # small time penalty
    r_hit: float = 20.0
    r_crash: float = -10.0
    r_timeout: float = -5.0


# Discrete actions shared by DQN and PPO
ACTIONS = ["coast", "pitch_up", "pitch_down", "yaw_left", "yaw_right", "boost", "brake"]
N_ACTIONS = len(ACTIONS)
