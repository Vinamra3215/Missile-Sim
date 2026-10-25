"""
TunnelEnv: a batched (vectorised) 3D environment written in pure PyTorch.

Thousands of independent environments are simulated at once as tensors.
Every episode procedurally generates a new map: a square tunnel along +x with
randomly placed, randomly angled cylindrical pipes, and a target core near
the far end. The missile must fly through the gaps and hit the target.

API (vectorised, Gymnasium-like):
    env = TunnelEnv(cfg, device="cpu", seed=0)
    obs = env.reset()                                  # [N, OBS_DIM]
    obs, reward, terminated, truncated, info = env.step(actions)   # actions: [N] long
Finished environments are reset automatically. info["final_obs"] holds the
observation *before* the automatic reset (needed for correct bootstrapping),
and info["episodes"] holds stats of episodes that just ended.
"""
import math
import torch

from sim3d.config import EnvConfig, N_ACTIONS

INF = 1e9


class TunnelEnv:
    def __init__(self, cfg: EnvConfig = None, device="cpu", seed=0):
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        self.device = torch.device(device)
        self.N = c.num_envs
        self.P = c.max_pipes
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        self.n_rays = c.ray_grid * c.ray_grid
        self.obs_dim = 2 * self.n_rays + 11      # rays now + rays last step (motion) + state
        self.n_actions = N_ACTIONS

        # curriculum: global level in [0, 1]; per-episode level is sampled below it
        self.level = 0.0
        self.level_spread = 0.3
        self.fixed_level = None          # if set, every map uses exactly this level
        self.auto_reset = True           # viewer sets False to show crashes/hits first

        # ray angle offsets (tangent space), shape [R]
        g = torch.linspace(-c.ray_fov / 2, c.ray_fov / 2, c.ray_grid, device=self.device)
        yy, pp = torch.meshgrid(g, g, indexing="xy")
        self.ray_tan_yaw = torch.tan(yy.reshape(-1))
        self.ray_tan_pitch = torch.tan(pp.reshape(-1))

        N, P, dev = self.N, self.P, self.device
        z = lambda *s: torch.zeros(*s, device=dev)
        self.pos = z(N, 3)
        self.yaw = z(N)
        self.pitch = z(N)
        self.speed = z(N)
        self.steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.max_steps = torch.ones(N, dtype=torch.long, device=dev)
        self.length = z(N)
        self.ep_level = z(N)
        self.target = z(N, 3)
        self.pipe_p = z(N, P, 3)
        self.pipe_u = z(N, P, 3)
        self.pipe_r = z(N, P)
        self.pipe_active = torch.zeros(N, P, dtype=torch.bool, device=dev)
        # moving pipes: p(t) = p0 + move_dir * amp * sin(freq * t + phase)
        self.pipe_p0 = z(N, P, 3)
        self.pipe_mdir = z(N, P, 3)
        self.pipe_amp = z(N, P)
        self.pipe_freq = z(N, P)
        self.pipe_phase = z(N, P)
        # gates: wall at x = gate_x with a circular hole centred at gate_c (y, z)
        self.gate_active = torch.zeros(N, P, dtype=torch.bool, device=dev)
        self.gate_x = z(N, P)
        self.gate_c = z(N, P, 2)
        self.gate_r = z(N, P)
        self.last_rays = -torch.ones(N, self.n_rays, device=dev)
        self._cur_rays = None
        self.prev_dist = z(N)
        self.ep_return = z(N)

    # ------------------------------------------------------------------ utils
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _uniform(self, lo, hi, *shape):
        return lo + (hi - lo) * self._rand(*shape)

    def set_level(self, level):
        self.level = float(max(0.0, min(self.cfg.max_level, level)))

    @staticmethod
    def heading(yaw, pitch):
        cp = torch.cos(pitch)
        return torch.stack([cp * torch.cos(yaw), cp * torch.sin(yaw), torch.sin(pitch)], -1)

    def body_frame(self):
        """Forward, right and up unit vectors of every missile: each [N, 3]."""
        f = self.heading(self.yaw, self.pitch)
        world_up = torch.zeros_like(f)
        world_up[:, 2] = 1.0
        r = torch.nn.functional.normalize(torch.cross(f, world_up, dim=-1), dim=-1)
        u = torch.cross(r, f, dim=-1)
        return f, r, u

    # ------------------------------------------------------------- map making
    def _generate(self, idx):
        """Build fresh maps and missile start states for the env indices `idx`."""
        c, M, P = self.cfg, idx.numel(), self.P
        if M == 0:
            return
        S = c.tunnel_half

        if self.fixed_level is not None:
            lvl = torch.full((M,), float(self.fixed_level), device=self.device)
        else:
            lo = max(0.0, self.level - self.level_spread)
            lvl = self._uniform(lo, self.level, M) if self.level > lo else torch.full(
                (M,), self.level, device=self.device)
        base = lvl.clamp(max=1.0)                      # normal tier progress   (0..1)
        xp = (lvl - 1.0).clamp(0.0, 1.0)              # expert tier progress   (0..1)
        L = c.len_min + base * (c.len_max - c.len_min) + xp * (c.len_expert - c.len_max)

        # number of pipes grows with level, limited by available space
        span = L - c.first_pipe_x - 12.0
        n_fit = torch.floor(span / c.pipe_min_spacing) + 1
        n = torch.round(c.pipes_min + lvl * (c.obstacles_l1 - c.pipes_min))
        n = torch.minimum(torch.minimum(n, n_fit), torch.tensor(float(P), device=self.device))

        k = torch.arange(P, device=self.device).float().unsqueeze(0)          # [1, P]
        spacing = (span / n).unsqueeze(1)                                     # [M, 1]
        jitter = self._uniform(-0.3, 0.3, M, P) * spacing
        x = c.first_pipe_x + (k + 0.5) * spacing + jitter
        active = k < n.unsqueeze(1)

        # ── gates: some slots become walls with a hole; the slot before a gate is
        #    left empty so the agent has room to line up
        ramp_g = ((lvl - c.gate_start) / (1 - c.gate_start)).clamp(0, 1).unsqueeze(1)
        x1 = xp.unsqueeze(1)
        gate_prob = ramp_g * c.gate_prob_max + x1 * (c.expert_gate_prob - c.gate_prob_max)
        is_gate = active & (self._rand(M, P) < gate_prob)
        prev_gate = torch.zeros_like(is_gate)
        prev_gate[:, 1:] = is_gate[:, :-1]
        is_gate = is_gate & ~prev_gate                    # no two gates in a row
        blocked = torch.zeros_like(is_gate)
        blocked[:, :-1] = is_gate[:, 1:]                  # empty slot before each gate
        pipe_on = active & ~is_gate & ~blocked
        hole_r = (c.gate_hole_max - (c.gate_hole_max - c.gate_hole_min) * ramp_g
                  - (c.gate_hole_min - c.expert_gate_hole) * x1).expand(M, P)
        lim = (S - hole_r - 0.8).clamp_min(0)
        gate_c = torch.stack([self._uniform(-1, 1, M, P) * lim, self._uniform(-1, 1, M, P) * lim], -1)

        theta = self._uniform(0, math.pi, M, P)                               # angle in y-z plane
        off = self._uniform(-c.pipe_offset_frac, c.pipe_offset_frac, M, P) * S
        tilt_max = (c.pipe_tilt_max * lvl.clamp(max=1.5)).unsqueeze(1)
        tilt = self._uniform(-1, 1, M, P) * tilt_max
        r_hi = c.pipe_r_min + (0.4 + 0.6 * base).unsqueeze(1) * (c.pipe_r_max - c.pipe_r_min)
        radius = c.pipe_r_min + self._rand(M, P) * (r_hi - c.pipe_r_min)

        p = torch.stack([x, -off * torch.sin(theta), off * torch.cos(theta)], -1)
        u = torch.stack([torch.sin(tilt),
                         torch.cos(tilt) * torch.cos(theta),
                         torch.cos(tilt) * torch.sin(theta)], -1)

        # ── moving pipes: slide back and forth across the tunnel
        ramp_m = ((lvl - c.move_start) / (1 - c.move_start)).clamp(0, 1).unsqueeze(1)
        move_frac = ramp_m * c.move_frac_max + x1 * (c.expert_move_frac - c.move_frac_max)
        moving = pipe_on & (self._rand(M, P) < move_frac)
        amp = 1.0 + self._rand(M, P) * (c.move_amp_max - 1.0 + x1 * (c.expert_move_amp - c.move_amp_max))
        amp = torch.minimum(amp, (0.85 * S - off.abs()).clamp_min(0)) * moving
        mdir = torch.stack([torch.zeros_like(theta), -torch.sin(theta), torch.cos(theta)], -1)
        freq = c.move_freq_min + self._rand(M, P) * (
            c.move_freq_max - c.move_freq_min + x1 * (c.expert_move_freq - c.move_freq_max))
        phase = self._uniform(0, 2 * math.pi, M, P)

        # park unused pipe slots far behind the start, with zero radius
        p[~pipe_on] = torch.tensor([-1000.0, 0.0, 0.0], device=self.device)
        u[~pipe_on] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        radius = torch.where(pipe_on, radius, torch.zeros_like(radius))
        active = pipe_on

        m = c.tunnel_half - c.target_margin
        tgt = torch.stack([L, self._uniform(-m, m, M), self._uniform(-m, m, M)], -1)

        start = torch.stack([torch.zeros(M, device=self.device),
                             self._uniform(-3, 3, M), self._uniform(-3, 3, M)], -1)

        self.pos[idx] = start
        self.yaw[idx] = self._uniform(-0.2, 0.2, M)
        self.pitch[idx] = self._uniform(-0.15, 0.15, M)
        self.speed[idx] = c.speed_init
        self.steps[idx] = 0
        self.length[idx] = L
        self.ep_level[idx] = lvl
        self.max_steps[idx] = torch.ceil(
            c.time_limit_factor * (L + c.end_gap) / c.speed_min / c.dt).long()
        self.target[idx] = tgt
        self.pipe_p[idx] = p
        self.pipe_u[idx] = u
        self.pipe_r[idx] = radius
        self.pipe_active[idx] = active
        self.pipe_p0[idx] = p
        self.pipe_mdir[idx] = mdir
        self.pipe_amp[idx] = amp
        self.pipe_freq[idx] = freq
        self.pipe_phase[idx] = phase
        self.gate_active[idx] = is_gate
        self.gate_x[idx] = torch.where(is_gate, x, torch.full_like(x, -1000.0))
        self.gate_c[idx] = gate_c
        self.gate_r[idx] = hole_r
        self.last_rays[idx] = -1.0
        self._move_pipes(idx)
        self.prev_dist[idx] = torch.linalg.norm(tgt - start, dim=-1)
        self.ep_return[idx] = 0.0

    def reset(self):
        self._generate(torch.arange(self.N, device=self.device))
        obs = self._obs()
        self.last_rays = self._cur_rays
        return obs

    def reset_envs(self, idx):
        """Regenerate maps for specific envs (used when auto_reset is off)."""
        self._generate(torch.as_tensor(idx, device=self.device).long().view(-1))
        obs = self._obs()
        self.last_rays = self._cur_rays
        return obs

    # ---------------------------------------------------------------- physics
    def _move_pipes(self, idx=None):
        t = (self.steps.float() * self.cfg.dt)
        if idx is None:
            off = self.pipe_amp * torch.sin(self.pipe_freq * t.unsqueeze(1) + self.pipe_phase)
            self.pipe_p = self.pipe_p0 + self.pipe_mdir * off.unsqueeze(-1)
        else:
            off = self.pipe_amp[idx] * torch.sin(self.pipe_freq[idx] * t[idx].unsqueeze(1)
                                                  + self.pipe_phase[idx])
            self.pipe_p[idx] = self.pipe_p0[idx] + self.pipe_mdir[idx] * off.unsqueeze(-1)

    def _gate_hit(self, pos):
        c = self.cfg
        near = (pos[:, 0:1] - self.gate_x).abs() < c.gate_thickness / 2 + c.missile_radius
        d = torch.linalg.norm(pos[:, None, 1:] - self.gate_c, dim=-1)
        return (near & (d > self.gate_r - c.missile_radius) & self.gate_active).any(-1)

    def _pipe_distance(self, pos):
        """Distance from each missile centre to each pipe axis: [N, P]."""
        w = pos.unsqueeze(1) - self.pipe_p                       # [N, P, 3]
        along = (w * self.pipe_u).sum(-1, keepdim=True)
        perp = w - along * self.pipe_u
        return torch.linalg.norm(perp, dim=-1)

    def step(self, actions):
        c = self.cfg
        a = actions.to(self.device).long()
        turn = c.turn_rate * c.dt

        self.pitch = torch.clamp(
            self.pitch + turn * ((a == 1).float() - (a == 2).float()),
            -c.pitch_limit, c.pitch_limit)
        self.yaw = self.yaw + turn * ((a == 3).float() - (a == 4).float())
        self.yaw = torch.remainder(self.yaw + math.pi, 2 * math.pi) - math.pi
        self.speed = torch.clamp(
            self.speed + c.accel * c.dt * ((a == 5).float() - (a == 6).float()),
            c.speed_min, c.speed_max)

        f = self.heading(self.yaw, self.pitch)
        self.pos = self.pos + f * (self.speed * c.dt).unsqueeze(-1)
        self.steps += 1
        self._move_pipes()

        # ── events ──
        dist = torch.linalg.norm(self.target - self.pos, dim=-1)
        hit = dist < (c.target_radius + c.missile_radius)

        S = c.tunnel_half - c.missile_radius
        x, y, zc = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
        wall = (y.abs() > S) | (zc.abs() > S) | (x < -2.0) | \
               (x > self.length + c.end_gap - c.missile_radius)
        pipe = ((self._pipe_distance(self.pos) < self.pipe_r + c.missile_radius)
                & self.pipe_active).any(-1)
        crash = (wall | pipe | self._gate_hit(self.pos)) & ~hit
        terminated = hit | crash
        timeout = (self.steps >= self.max_steps) & ~terminated

        # ── reward ──
        reward = c.r_progress * (self.prev_dist - dist) + c.r_step
        reward = reward + c.r_hit * hit.float() + c.r_crash * crash.float() \
                 + c.r_timeout * timeout.float()
        self.prev_dist = dist
        self.ep_return += reward

        final_obs = self._obs()
        done = terminated | timeout
        info = {"final_obs": final_obs}
        if done.any():
            d = done.nonzero(as_tuple=True)[0]
            info["episodes"] = {
                "return": self.ep_return[d].clone(),
                "length": self.steps[d].clone(),
                "success": hit[d].float(),
                "crash": crash[d].float(),
                "timeout": timeout[d].float(),
                "level": self.ep_level[d].clone(),
            }
            if self.auto_reset:
                self._generate(d)
                obs = self._obs()
            else:
                obs = final_obs
        else:
            obs = final_obs
        self.last_rays = self._cur_rays
        return obs, reward, terminated, timeout, info

    # ------------------------------------------------------------ observation
    def cast_rays(self, origin, dirs):
        """
        Ray distances from `origin` [N,3] along unit `dirs` [N,R,3] to the
        nearest tunnel wall or pipe. Returns [N, R] clipped to ray_range.
        """
        c = self.cfg
        S = c.tunnel_half
        o = origin.unsqueeze(1)                                   # [N,1,3]
        t = torch.full(dirs.shape[:2], INF, device=self.device)

        # side / floor / ceiling walls (we are inside the box)
        for ax in (1, 2):
            d = dirs[..., ax]
            safe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
            tw = torch.where(d > 0, (S - o[..., ax]) / safe, (-S - o[..., ax]) / safe)
            t = torch.minimum(t, torch.where(d.abs() < 1e-6, torch.full_like(t, INF), tw))
        # back wall and entrance
        d = dirs[..., 0]
        safe = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        x_end = (self.length + self.cfg.end_gap).view(-1, 1)
        tw = torch.where(d > 0, (x_end - o[..., 0]) / safe, (-2.0 - o[..., 0]) / safe)
        t = torch.minimum(t, torch.where(d.abs() < 1e-6, torch.full_like(t, INF), tw))

        # pipes: ray vs infinite cylinder, [N, R, P]
        u = self.pipe_u.unsqueeze(1)                              # [N,1,P,3]
        w = o.unsqueeze(2) - self.pipe_p.unsqueeze(1)             # [N,1,P,3]
        dr = dirs.unsqueeze(2)                                    # [N,R,1,3]
        d_perp = dr - (dr * u).sum(-1, keepdim=True) * u          # [N,R,P,3]
        w_perp = w - (w * u).sum(-1, keepdim=True) * u            # [N,1,P,3]
        A = (d_perp * d_perp).sum(-1).clamp_min(1e-9)
        B = 2.0 * (d_perp * w_perp).sum(-1)
        C = (w_perp * w_perp).sum(-1) - self.pipe_r.unsqueeze(1) ** 2
        disc = B * B - 4 * A * C
        tc = (-B - torch.sqrt(disc.clamp_min(0))) / (2 * A)
        ok = (disc >= 0) & (tc > 0) & self.pipe_active.unsqueeze(1)
        tc = torch.where(ok, tc, torch.full_like(tc, INF))
        t = torch.minimum(t, tc.min(-1).values)

        # gates: thin wall at gate_x, the ray passes only through the hole
        dx = dirs[..., 0:1]                                       # [N,R,1]
        safe = torch.where(dx.abs() < 1e-6, torch.full_like(dx, 1e-6), dx)
        face = self.gate_x.unsqueeze(1) - torch.sign(safe) * c.gate_thickness / 2   # [N,1|R,P]
        tg = (face - o[..., 0:1]) / safe                          # [N,R,P]
        hit_yz = o[..., None, 1:] + tg.unsqueeze(-1) * dirs[..., None, 1:]          # [N,R,P,2]
        in_hole = torch.linalg.norm(hit_yz - self.gate_c.unsqueeze(1), dim=-1) < self.gate_r.unsqueeze(1)
        okg = (tg > 0) & ~in_hole & self.gate_active.unsqueeze(1) & (dx.abs() > 1e-6)
        tg = torch.where(okg, tg, torch.full_like(tg, INF))
        t = torch.minimum(t, tg.min(-1).values)

        return t.clamp(0, c.ray_range)

    def ray_dirs(self, f, r, u):
        d = f.unsqueeze(1) + self.ray_tan_yaw.view(1, -1, 1) * r.unsqueeze(1) \
            + self.ray_tan_pitch.view(1, -1, 1) * u.unsqueeze(1)
        return torch.nn.functional.normalize(d, dim=-1)

    def _obs(self):
        c = self.cfg
        f, r, u = self.body_frame()
        rays = self.cast_rays(self.pos, self.ray_dirs(f, r, u)) / c.ray_range   # [N,R] in [0,1]
        prev = torch.where(self.last_rays < 0, rays, self.last_rays)
        self._cur_rays = rays

        v = self.target - self.pos
        dist = torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
        vb = torch.stack([(v * f).sum(-1), (v * r).sum(-1), (v * u).sum(-1)], -1) / dist.unsqueeze(-1)

        speed_n = 2 * (self.speed - c.speed_min) / (c.speed_max - c.speed_min) - 1
        S = c.tunnel_half
        extra = torch.stack([
            torch.clamp(dist / 100.0, 0, 1),
            speed_n,
            self.pitch / c.pitch_limit,
            torch.sin(self.yaw),
            torch.cos(self.yaw),
            self.pos[:, 1] / S,
            self.pos[:, 2] / S,
            torch.clamp((self.length - self.pos[:, 0]) / c.len_expert, -1, 1),
        ], -1)
        return torch.cat([rays, prev, vb, extra], -1)

    # --------------------------------------------------------------- export
    def export_map(self, i=0):
        """Plain-python description of env i's map (for the 3D viewer)."""
        act = self.pipe_active[i].cpu()
        return {
            "tunnel_half": self.cfg.tunnel_half,
            "length": float(self.length[i]),
            "end_gap": self.cfg.end_gap,
            "target": self.target[i].tolist(),
            "target_radius": self.cfg.target_radius,
            "pipes": [
                {"index": j, "p": self.pipe_p[i, j].tolist(), "u": self.pipe_u[i, j].tolist(),
                 "r": float(self.pipe_r[i, j]), "moving": bool(self.pipe_amp[i, j] > 0)}
                for j in range(self.P) if act[j]
            ],
            "gates": [
                {"x": float(self.gate_x[i, j]), "c": self.gate_c[i, j].tolist(),
                 "r": float(self.gate_r[i, j]), "thickness": self.cfg.gate_thickness}
                for j in range(self.P) if self.gate_active[i, j]
            ],
        }

    def export_state(self, i=0):
        return {"pos": self.pos[i].tolist(), "yaw": float(self.yaw[i]),
                "pitch": float(self.pitch[i]), "speed": float(self.speed[i])}
