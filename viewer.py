"""
MissileSim 3D viewer (Ursina).

Watch a trained agent (or fly yourself) through freshly generated tunnels.
The viewer runs the exact same simulator used for training, one env on CPU.

Examples:
    python viewer.py --policy models/ppo_s1_best.pth
    python viewer.py --policy models/dqn_s1_best.pth --level 0.6
    python viewer.py --policy steer                         # hand-coded baseline
    python viewer.py --policy models/ppo_s1_best.pth --record 30 --out demo/ppo.mp4

Controls:
    TAB      switch AI <-> manual flight      C      cycle camera (chase / side / top)
    arrows   pitch / yaw (manual mode)         SHIFT  boost, CTRL brake (manual)
    R        new map                            SPACE  pause
    + / -    difficulty level up / down         [ / ]  slower / faster playback
    ESC      quit
"""
import argparse
import math
import os
import random
import shutil
import subprocess
import sys

import torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="steer", help="model .pth path, 'steer' or 'random'")
    p.add_argument("--level", type=float, default=1.0, help="map difficulty 0..2 (1-2 = expert tier)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    p.add_argument("--record", type=float, default=0, help="seconds of video to record, then quit")
    p.add_argument("--out", default="demo/run.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--screenshot", default=None, help="save one screenshot after N sim steps: path:N")
    p.add_argument("--camera", default="chase", choices=["chase", "side", "top"])
    return p.parse_args()


ARGS = parse()

from ursina import (Ursina, Entity, Mesh, Text, Vec3, Vec2, Shader, camera, color, window,
                    scene, time, held_keys, destroy, application, Grid, curve)
from panda3d.core import Filename, Point3, Vec3 as PVec3

from agents.policy import load_policy
from sim3d.config import EnvConfig, ACTIONS
from sim3d.tunnel_env import TunnelEnv

W, H = (int(v) for v in ARGS.size.lower().split("x"))
app = Ursina(title="MissileSim - RL Missile Navigation", size=(W, H), borderless=False,
             development_mode=False, vsync=not ARGS.record)
window.color = color.rgb(0.02, 0.025, 0.05)
window.fps_counter.enabled = False
window.exit_button.visible = False

# ─────────────────────────────────────────────────────────────── shader
VERT = """
#version 140
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    v_pos = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
"""
FRAG = """
#version 140
uniform vec4 p3d_ColorScale;
uniform vec3 cam_pos;
uniform vec3 glow_pos;
uniform vec3 glow_col;
uniform vec3 flame_pos;
uniform float flame_pow;
uniform vec3 fog_col;
uniform float fog_den;
uniform float emissive;
in vec3 v_normal;
in vec3 v_pos;
out vec4 frag;
void main() {
    vec4 base = p3d_ColorScale;
    vec3 n = normalize(v_normal);
    vec3 view = normalize(cam_pos - v_pos);
    if (dot(n, view) < 0.0) n = -n;                       // two-sided
    vec3 key = normalize(vec3(0.3, 0.8, -0.4));
    float diff = max(dot(n, key), 0.0);
    float head = max(dot(n, view), 0.0);                  // camera "headlight"
    vec3 lit = base.rgb * (0.22 + 0.38 * diff + 0.40 * head);
    vec3 Lg = glow_pos - v_pos;  float dg = length(Lg);
    lit += base.rgb * glow_col * max(dot(n, Lg / dg), 0.0) * 6.0 / (1.0 + 0.04 * dg * dg);
    vec3 Lf = flame_pos - v_pos; float df = length(Lf);
    lit += base.rgb * vec3(1.0, 0.55, 0.2) * flame_pow * max(dot(n, Lf / df), 0.0) * 3.0 / (1.0 + 0.3 * df * df);
    float rim = pow(1.0 - head, 3.0);
    lit += base.rgb * rim * 0.25;
    lit = mix(lit, base.rgb, emissive);
    float d = length(v_pos - cam_pos);
    float f = 1.0 - exp(-fog_den * d);
    frag = vec4(mix(lit, fog_col, f * (1.0 - 0.6 * emissive)), base.a);
}
"""
shader = Shader(language=Shader.GLSL, vertex=VERT, fragment=FRAG,
                default_input={"emissive": 0.0})


def E(*a, **kw):
    """3D entity using the custom lighting shader (UI keeps Ursina's default)."""
    kw.setdefault("shader", shader)
    e = Entity(*a, **kw)
    e.double_sided = True
    return e


FOG = color.rgb(0.02, 0.025, 0.05)
for k, v in {"cam_pos": Vec3(0, 0, 0), "glow_pos": Vec3(0, 0, 0), "glow_col": Vec3(0, 0.9, 1),
             "flame_pos": Vec3(0, 0, 0), "flame_pow": 1.0,
             "fog_col": Vec3(FOG.r, FOG.g, FOG.b), "fog_den": 0.012}.items():
    scene.set_shader_input(k, v)


# ─────────────────────────────────────────────────────────────── geometry helpers
def U(v):
    """simulator (x fwd, y left, z up)  ->  ursina (x right, y up, z fwd)"""
    return Vec3(-v[1], v[2], v[0])


def cone_mesh(radius, height, res=20, back=False):
    """Cone along +Z (tip at z=height) or -Z if back=True, with normals."""
    s = -1 if back else 1
    tip = Vec3(0, 0, s * height)
    verts, norms = [], []
    for i in range(res):
        a0, a1 = 2 * math.pi * i / res, 2 * math.pi * (i + 1) / res
        p0 = Vec3(math.cos(a0) * radius, math.sin(a0) * radius, 0)
        p1 = Vec3(math.cos(a1) * radius, math.sin(a1) * radius, 0)
        for p in (p0, p1, tip):
            verts.append(p)
        for a in (a0, a1, (a0 + a1) / 2):
            norms.append(Vec3(math.cos(a), math.sin(a), s * radius / height).normalized())
        verts += [Vec3(0, 0, 0), p1, p0]
        norms += [Vec3(0, 0, -s)] * 3
    return Mesh(vertices=verts, normals=norms, mode="triangle")


def cyl_mesh(radius, length, direction=(0, 0, 1), res=24, caps=True):
    """Cylinder centred on the origin along `direction`, with normals."""
    d = Vec3(*direction).normalized()
    a = Vec3(1, 0, 0) if abs(d.x) < 0.9 else Vec3(0, 1, 0)
    e1 = d.cross(a).normalized()
    e2 = d.cross(e1).normalized()
    h = length / 2
    verts, norms = [], []
    for i in range(res):
        t0, t1 = 2 * math.pi * i / res, 2 * math.pi * (i + 1) / res
        n0 = e1 * math.cos(t0) + e2 * math.sin(t0)
        n1 = e1 * math.cos(t1) + e2 * math.sin(t1)
        b0, b1 = n0 * radius - d * h, n1 * radius - d * h
        f0, f1 = n0 * radius + d * h, n1 * radius + d * h
        verts += [b0, b1, f1, b0, f1, f0]
        norms += [n0, n1, n1, n0, n1, n0]
        if caps:
            verts += [d * h, f0, f1, -d * h, b1, b0]
            norms += [d] * 3 + [-d] * 3
    return Mesh(vertices=verts, normals=norms, mode="triangle")


def gate_mesh(S, cx, cy, r, th, res=72):
    """Square wall (2S x 2S, thickness th, centred on z=0) with a round hole at (cx, cy)."""
    verts, norms = [], []
    hz = th / 2

    def edge(a):
        dx, dy = math.cos(a), math.sin(a)
        ts = []
        if abs(dx) > 1e-9: ts.append(((S if dx > 0 else -S) - cx) / dx)
        if abs(dy) > 1e-9: ts.append(((S if dy > 0 else -S) - cy) / dy)
        t = min(ts)
        return Vec3(cx + dx * t, cy + dy * t, 0)

    for i in range(res):
        a0, a1 = 2 * math.pi * i / res, 2 * math.pi * (i + 1) / res
        c0 = Vec3(cx + r * math.cos(a0), cy + r * math.sin(a0), 0)
        c1 = Vec3(cx + r * math.cos(a1), cy + r * math.sin(a1), 0)
        b0, b1 = edge(a0), edge(a1)
        quads = [(c0, c1, b1, b0)]
        if abs(b0.x - b1.x) > 1e-6 and abs(b0.y - b1.y) > 1e-6:      # segment wraps a corner
            corner = Vec3(S if (b0.x + b1.x) > 0 else -S, S if (b0.y + b1.y) > 0 else -S, 0)
            quads = [(c0, c1, b1, corner), (c0, corner, b0, b0)]
        for z, n in ((-hz, Vec3(0, 0, -1)), (hz, Vec3(0, 0, 1))):
            for q in quads:
                q = [Vec3(v.x, v.y, z) for v in q]
                verts += [q[0], q[1], q[2], q[0], q[2], q[3]]
                norms += [n] * 6
        # inside of the hole
        n0 = Vec3(-math.cos(a0), -math.sin(a0), 0)
        n1 = Vec3(-math.cos(a1), -math.sin(a1), 0)
        f0, f1 = Vec3(c0.x, c0.y, -hz), Vec3(c1.x, c1.y, -hz)
        k0, k1 = Vec3(c0.x, c0.y, hz), Vec3(c1.x, c1.y, hz)
        verts += [f0, f1, k1, f0, k1, k0]
        norms += [n0, n1, n1, n0, n1, n0]
    return Mesh(vertices=verts, normals=norms, mode="triangle")


def box(parent=None, **kw):
    return E(parent=parent or scene, model="cube", **kw)


# ─────────────────────────────────────────────────────────────── palette
WALL = color.rgb(0.10, 0.12, 0.18)
GRID = color.rgb(0.22, 0.28, 0.40)
RIB = color.rgb(0.30, 0.38, 0.55)
PIPE = color.rgb(0.78, 0.36, 0.26)
PIPE_JOINT = color.rgb(0.55, 0.24, 0.18)
CORE = color.rgb(0.2, 1.0, 0.95)
HALO = color.rgba(0.2, 0.9, 1.0, 0.18)
MOVER = color.rgb(1.0, 0.78, 0.15)
GATE = color.rgb(0.28, 0.31, 0.38)
GATE_RIM = color.rgb(1.0, 0.6, 0.15)


# ─────────────────────────────────────────────────────────────── world
class World:
    def __init__(self):
        self.root = None
        self.pipes = {}
        self.core = self.halo = None
        self.rings = []

    def build(self, m):
        if self.root:
            destroy(self.root)
        self.root = E()
        S, L, gap = m["tunnel_half"], m["length"], m["end_gap"]
        x0, x1 = -6.0, L + gap
        Ln, zc = x1 - x0, (x0 + x1) / 2
        r = self.root

        # walls (cubes) + grid lines
        t = 0.2
        box(r, scale=(2 * S + 2 * t, t, Ln), position=(0, -S - t / 2, zc), color=WALL)
        box(r, scale=(2 * S + 2 * t, t, Ln), position=(0, S + t / 2, zc), color=WALL)
        box(r, scale=(t, 2 * S, Ln), position=(-S - t / 2, 0, zc), color=WALL)
        box(r, scale=(t, 2 * S, Ln), position=(S + t / 2, 0, zc), color=WALL)
        box(r, scale=(2 * S, 2 * S, t), position=(0, 0, x1 + t / 2), color=color.rgb(0.12, 0.16, 0.22))
        nx, nz = int(2 * S / 2), max(1, int(Ln / 2))
        for pos, rot, sc in [((0, -S + 0.02, zc), (90, 0, 0), (2 * S, Ln)),
                             ((0, S - 0.02, zc), (90, 0, 0), (2 * S, Ln)),
                             ((-S + 0.02, 0, zc), (0, 90, 0), (Ln, 2 * S)),
                             ((S - 0.02, 0, zc), (0, 90, 0), (Ln, 2 * S))]:
            g = E(parent=r, model=Grid(nx if sc[0] == 2 * S else nz, nz if sc[1] == Ln else nx,
                                             thickness=1.5),
                       position=pos, rotation=rot, scale=sc, color=GRID,
                       shader=Entity.default_shader)       # lines have no normals: unlit
        # glowing ribs every 12 units
        z = 0.0
        while z < x1:
            for pos, sc in [((0, -S + 0.06, z), (2 * S, 0.12, 0.35)), ((0, S - 0.06, z), (2 * S, 0.12, 0.35)),
                            ((-S + 0.06, 0, z), (0.12, 2 * S, 0.35)), ((S - 0.06, 0, z), (0.12, 2 * S, 0.35))]:
                b = box(r, position=pos, scale=sc, color=RIB)
                b.set_shader_input("emissive", 0.6)
            z += 12.0

        # pipes with joints (moving pipes get a hazard stripe and are updated every frame)
        self.pipes = {}
        for pp in m["pipes"]:
            c, d, rad = U(pp["p"]), U(pp["u"]), pp["r"]
            holder = E(parent=r, position=c)
            E(parent=holder, model=cyl_mesh(rad, 4.4 * S, tuple(d)), color=PIPE)
            for s in (-0.55, 0.55):
                E(parent=holder, model=cyl_mesh(rad * 1.12, 0.6, tuple(d)),
                  position=d * (s * S), color=PIPE_JOINT)
            if pp["moving"]:
                for s in (-0.25, 0.0, 0.25):
                    E(parent=holder, model=cyl_mesh(rad * 1.06, 0.35, tuple(d)),
                      position=d * (s * S), color=MOVER).set_shader_input("emissive", 0.7)
            self.pipes[pp["index"]] = holder

        # gates: dark wall with a glowing ring around the hole
        for g in m.get("gates", []):
            cx, cy = -g["c"][0], g["c"][1]
            E(parent=r, model=gate_mesh(S, cx, cy, g["r"], g["thickness"]), z=g["x"], color=GATE)
            for dz in (-g["thickness"] / 2 - 0.02, g["thickness"] / 2 + 0.02):
                E(parent=r, model=cyl_mesh(g["r"] + 0.05, 0.3, (0, 0, 1), res=64, caps=False),
                  position=(cx, cy, g["x"] + dz), color=GATE_RIM).set_shader_input("emissive", 1.0)

        # target: glowing core + halo + rotating rings
        tp, tr = U(m["target"]), m["target_radius"]
        self.target_pos = tp
        self.core = E(parent=r, model="sphere", position=tp, scale=tr * 1.1, color=CORE)
        self.core.set_shader_input("emissive", 0.85)
        self.halo = E(parent=r, model="sphere", position=tp, scale=tr * 2.2, color=HALO)
        self.halo.set_shader_input("emissive", 1.0)
        self.rings = []
        for i in range(2):
            ring = E(parent=r, position=tp, rotation=(random.uniform(0, 90), i * 90, 0))
            E(parent=ring, model=cyl_mesh(tr * 1.5, 0.08, (0, 0, 1), res=48, caps=False),
                   color=color.rgb(0.6, 1.0, 1.0)).set_shader_input("emissive", 1.0)
            self.rings.append(ring)
        scene.set_shader_input("glow_pos", tp)

    def animate(self, t):
        if self.core and self.core.enabled:
            k = 1 + 0.08 * math.sin(t * 4)
            self.halo.scale = self.core.scale_x * 2.0 * k
            for i, ring in enumerate(self.rings):
                ring.rotation_y += (60 + 40 * i) * time.dt
                ring.rotation_x += (25 - 15 * i) * time.dt

    def hide_target(self):
        for e in [self.core, self.halo] + self.rings:
            e.enabled = False
        scene.set_shader_input("glow_col", Vec3(0, 0, 0))


# ─────────────────────────────────────────────────────────────── missile
class Missile(Entity):
    def __init__(self):
        super().__init__()
        self.roll_pivot = E(parent=self)
        rp = self.roll_pivot
        E(parent=rp, model=cyl_mesh(0.24, 1.8, (0, 0, 1)), color=color.rgb(0.92, 0.92, 0.95))
        E(parent=rp, model=cyl_mesh(0.245, 0.18, (0, 0, 1)), z=0.55, color=color.rgb(0.85, 0.2, 0.2))
        E(parent=rp, model=cone_mesh(0.24, 0.6), z=0.9, color=color.rgb(0.75, 0.75, 0.8))
        for k in range(4):
            piv = E(parent=rp, rotation_z=45 + 90 * k)
            box(piv, position=(0.38, 0, -0.65), scale=(0.32, 0.03, 0.45), color=color.rgb(0.35, 0.37, 0.42))
        self.flame = E(parent=rp, model=cone_mesh(0.16, 0.8, back=True), z=-0.9, color=color.rgb(1, 0.55, 0.1))
        self.flame.set_shader_input("emissive", 1.0)
        self.core = E(parent=rp, model=cone_mesh(0.1, 0.5, back=True), z=-0.9, color=color.rgb(1, 0.95, 0.6))
        self.core.set_shader_input("emissive", 1.0)
        self.scale = 1.3

    def pose(self, pos, fwd, roll):
        self.position = pos
        self.lookAt(Point3(*(pos + fwd)), PVec3(0, 1, 0))     # Panda native: +Z -> fwd, keep +Y up
        self.roll_pivot.rotation_z = roll


# ─────────────────────────────────────────────────────────────── particles
class Particles:
    def __init__(self):
        self.items = []

    def spawn(self, pos, vel, life, scale, col, model="cube", emissive=1.0, grow=0.0, gravity=0.0):
        e = E(model=model, position=pos, scale=scale, color=col)
        e.set_shader_input("emissive", emissive)
        self.items.append([e, Vec3(vel), life, life, scale, grow, gravity])

    def explosion(self, pos, big=True):
        n = 90 if big else 45
        for _ in range(n):
            d = Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)).normalized()
            sp = random.uniform(4, 18) if big else random.uniform(3, 10)
            col = random.choice([color.white, color.rgb(1, 0.85, 0.4), color.rgb(1, 0.5, 0.1),
                                 color.rgb(0.6, 1, 1) if big else color.rgb(0.9, 0.2, 0.1)])
            self.spawn(pos, d * sp, random.uniform(0.6, 1.4), random.uniform(0.15, 0.6), col,
                       gravity=-6 if not big else -2)
        for _ in range(12 if big else 6):       # smoke
            d = Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)).normalized()
            self.spawn(pos, d * random.uniform(2, 5), random.uniform(0.8, 1.4), random.uniform(0.4, 0.8),
                       color.rgba(0.6, 0.6, 0.65, 0.3), model="sphere", emissive=0.3, grow=1.0)
        for _ in range(30 if big else 0):       # glowing sparks
            d = Vec3(random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1)).normalized()
            self.spawn(pos, d * random.uniform(15, 28), random.uniform(0.3, 0.7), 0.12,
                       color.rgb(0.7, 1, 1), model="sphere")

    def update(self, dt):
        keep = []
        for it in self.items:
            e, v, life, life0, s0, grow, g = it
            life -= dt
            if life <= 0:
                destroy(e)
                continue
            v.y += g * dt
            v *= (1 - 1.5 * dt)
            e.position += v * dt
            k = life / life0
            e.scale = s0 * (k if grow == 0 else 1 + grow * (1 - k))
            c = e.color
            e.color = color.rgba(c.r, c.g, c.b, min(c.a, k + 0.05) if grow else c.a)
            it[1], it[2] = v, life
            keep.append(it)
        self.items = keep

    def clear(self):
        for it in self.items:
            destroy(it[0])
        self.items = []


# ─────────────────────────────────────────────────────────────── controller
class Viewer(Entity):
    def __init__(self):
        super().__init__()
        self.env = TunnelEnv(EnvConfig(num_envs=1), device="cpu", seed=ARGS.seed)
        self.env.auto_reset = False
        self.env.fixed_level = ARGS.level
        self.policy = load_policy(ARGS.policy, self.env.obs_dim, self.env.n_actions, "cpu")
        self.policy_name = os.path.basename(ARGS.policy)
        self.manual = False
        self.paused = False
        self.speed = ARGS.speed
        self.cam_mode = ARGS.camera
        self.world = World()
        self.missile = Missile()
        self.fx = Particles()
        self.stats = {"hit": 0, "crash": 0, "timeout": 0}
        self.t = 0.0
        self.trail_t = 0.0
        self.cam_pos = None
        self.flash = Entity(parent=camera.ui, model="quad", scale=(3, 2), color=color.rgba(1, 1, 1, 0), z=-1)
        self.hud = Text("", parent=camera.ui, position=window.top_left + Vec2(0.03, -0.03),
                        scale=1.0, color=color.rgb(0.85, 0.9, 1.0))
        self.banner = Text("", parent=camera.ui, origin=(0, 0), y=0.12, scale=3, color=CORE)
        self.help = Text("TAB manual/AI   C camera   R new map   +/- level   SPACE pause",
                         parent=camera.ui, position=window.bottom_left + Vec2(0.03, 0.04),
                         scale=0.8, color=color.rgba(0.7, 0.75, 0.9, 0.7))
        self.frames_dir = None
        self.frame_idx = 0
        if ARGS.record:
            self.frames_dir = os.path.join(os.path.dirname(ARGS.out) or ".", "_frames")
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            os.makedirs(self.frames_dir)
            self.help.enabled = False
        self.new_episode()

    # ── episode handling ──
    def new_episode(self):
        self.obs = self.env.reset()
        self.world.build(self.env.export_map(0))
        scene.set_shader_input("glow_col", Vec3(0.0, 0.9, 1.0))
        self.fx.clear()
        self.missile.enabled = True
        self.prev = self.cur = self.env.export_state(0)
        self.pipes_prev = self.pipes_cur = self.env.pipe_p[0].clone()
        self.acc = 0.0
        self.roll = 0.0
        self.ending = 0.0
        self.last_action = 0
        self.steps = 0
        self.ret = 0.0
        self.banner.text = ""
        self.cam_pos = None

    def end_episode(self, kind):
        self.stats[kind] += 1
        pos = U(self.cur["pos"])
        if kind == "hit":
            self.fx.explosion(self.world.target_pos, big=True)
            self.world.hide_target()
            self.banner.text, self.banner.color = "TARGET HIT", CORE
            self.flash.color = color.rgba(1, 1, 1, 0.5)
        elif kind == "crash":
            self.fx.explosion(pos, big=False)
            self.banner.text, self.banner.color = "CRASH", color.rgb(1, 0.35, 0.25)
            self.flash.color = color.rgba(1, 0.5, 0.2, 0.35)
        else:
            self.banner.text, self.banner.color = "TIMEOUT", color.rgb(1, 0.85, 0.3)
        self.missile.enabled = kind == "timeout"
        self.ending = 1.8

    def manual_action(self):
        if held_keys["up arrow"]: return 1
        if held_keys["down arrow"]: return 2
        if held_keys["left arrow"]: return 3
        if held_keys["right arrow"]: return 4
        if held_keys["shift"]: return 5
        if held_keys["control"]: return 6
        return 0

    def sim_step(self):
        a = self.manual_action() if self.manual else int(self.policy(self.obs)[0])
        self.last_action = a
        self.prev = self.cur
        self.obs, r, term, trunc, info = self.env.step(torch.tensor([a]))
        self.cur = self.env.export_state(0)
        self.pipes_prev, self.pipes_cur = self.pipes_cur, self.env.pipe_p[0].clone()
        self.steps += 1
        self.ret += float(r[0])
        if "episodes" in info:
            ep = info["episodes"]
            kind = "hit" if ep["success"][0] > 0 else "crash" if ep["crash"][0] > 0 else "timeout"
            self.end_episode(kind)

    # ── per frame ──
    def update(self):
        dt = 1.0 / ARGS.fps if ARGS.record else time.dt
        self.t += dt
        self.world.animate(self.t)
        self.fx.update(dt)
        self.flash.color = color.rgba(self.flash.color.r, self.flash.color.g, self.flash.color.b,
                                      max(0.0, self.flash.color.a - 2.5 * dt))

        if self.ending > 0:
            self.ending -= dt
            if self.ending <= 0:
                self.new_episode()
        elif not self.paused:
            self.acc += dt * self.speed
            sim_dt = self.env.cfg.dt
            while self.acc >= sim_dt and self.ending <= 0:
                self.acc -= sim_dt
                self.sim_step()

        # interpolated pose
        k = min(self.acc / self.env.cfg.dt, 1.0) if self.ending <= 0 else 1.0
        p0, p1 = U(self.prev["pos"]), U(self.cur["pos"])
        pos = p0 + (p1 - p0) * k
        yaw = self.prev["yaw"] + math.remainder(self.cur["yaw"] - self.prev["yaw"], 2 * math.pi) * k
        pitch = self.prev["pitch"] + (self.cur["pitch"] - self.prev["pitch"]) * k
        f_sim = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        fwd = U(f_sim)
        target_roll = {3: -35, 4: 35}.get(self.last_action, 0) if self.ending <= 0 else self.roll
        self.roll += (target_roll - self.roll) * min(1, 6 * dt)
        self.missile.pose(pos, fwd, self.roll)
        pp = self.pipes_prev + (self.pipes_cur - self.pipes_prev) * k
        for j, holder in self.world.pipes.items():
            holder.position = U(pp[j].tolist())

        # flame flicker + smoke trail
        boost = 1.4 if self.last_action == 5 else 0.6 if self.last_action == 6 else 1.0
        fl = boost * (0.85 + 0.3 * random.random())
        self.missile.flame.scale_z = fl
        self.missile.core.scale_z = fl
        tail = pos - fwd * 1.6
        scene.set_shader_input("flame_pos", tail)
        scene.set_shader_input("flame_pow", 1.0 if self.missile.enabled else 0.0)
        if self.missile.enabled and self.ending <= 0:
            self.trail_t += dt
            while self.trail_t > 0.03:
                self.trail_t -= 0.03
                self.fx.spawn(tail - fwd * 0.6 + Vec3(random.gauss(0, .05), random.gauss(0, .05), random.gauss(0, .05)),
                              -fwd * 1.5 + Vec3(0, 0.25, 0), 1.0, random.uniform(0.12, 0.2),
                              color.rgba(0.8, 0.82, 0.9, 0.25), model="sphere", emissive=0.6, grow=3.0)

        self.update_camera(pos, fwd, dt)
        st = self.cur
        tgt = Vec3(*self.env.target[0].tolist())
        dist = (tgt - Vec3(*st["pos"])).length()
        mode = "MANUAL" if self.manual else f"AI: {self.policy_name}"
        self.hud.text = (f"{mode}\nLevel {ARGS.level:.1f}   Step {self.steps}\n"
                         f"Speed {st['speed']:4.1f}   Dist {dist:5.1f}\n"
                         f"Action {ACTIONS[self.last_action]}\n"
                         f"Hits {self.stats['hit']}  Crashes {self.stats['crash']}  "
                         f"Timeouts {self.stats['timeout']}")
        if self.frames_dir is not None:
            self.capture()

    def update_camera(self, pos, fwd, dt):
        if self.cam_mode == "chase":
            flat = Vec3(fwd.x, 0, fwd.z).normalized() if abs(fwd.y) < 0.99 else fwd
            want = pos - fwd * 7.5 - flat * 1.5 + Vec3(0, 2.6, 0)
            look = pos + fwd * 8
        elif self.cam_mode == "side":
            want = Vec3(-7.6, 1.5, pos.z - 6)
            look = pos + Vec3(0, 0, 4)
        else:
            want = Vec3(0, 7.6, pos.z - 10)
            look = pos + Vec3(0, -2, 8)
        S = self.env.cfg.tunnel_half - 0.4
        want = Vec3(max(-S, min(S, want.x)), max(-S, min(S, want.y)), want.z)
        if self.cam_pos is None:
            self.cam_pos = want
        self.cam_pos += (want - self.cam_pos) * min(1, 5 * dt)
        camera.position = self.cam_pos
        camera.look_at(look)
        scene.set_shader_input("cam_pos", self.cam_pos)

    def capture(self):
        path = os.path.join(self.frames_dir, f"f_{self.frame_idx:05d}.png")
        base_win().saveScreenshot(Filename(path))
        self.frame_idx += 1
        if self.frame_idx >= ARGS.record * ARGS.fps:
            finish_recording(self.frames_dir)

    def input(self, key):
        if key == "tab":
            self.manual = not self.manual
        elif key == "c":
            modes = ["chase", "side", "top"]
            self.cam_mode = modes[(modes.index(self.cam_mode) + 1) % 3]
            self.cam_pos = None
        elif key == "r":
            self.new_episode()
        elif key == "space":
            self.paused = not self.paused
        elif key in ("+", "=", "+ up"):
            ARGS.level = min(2.0, round(ARGS.level + 0.1, 2))
            self.env.fixed_level = ARGS.level
        elif key == "-":
            ARGS.level = max(0.0, round(ARGS.level - 0.1, 2))
            self.env.fixed_level = ARGS.level
        elif key == "]":
            self.speed = min(4.0, self.speed * 1.5)
        elif key == "[":
            self.speed = max(0.25, self.speed / 1.5)


def base_win():
    return application.base.win


def finish_recording(frames_dir):
    out = ARGS.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(ARGS.fps),
                        "-i", os.path.join(frames_dir, "f_%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", out])
        gif = os.path.splitext(out)[0] + ".gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                        "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer",
                        gif])
        shutil.rmtree(frames_dir, ignore_errors=True)
        print(f"saved {out} and {gif}")
    else:
        print(f"ffmpeg not found; frames left in {frames_dir}")
    application.quit()
    sys.exit(0)


viewer = Viewer()

if ARGS.screenshot:
    path, n = ARGS.screenshot.rsplit(":", 1)
    ARGS.record = 0

    class Shot(Entity):
        k = 0

        def update(self):
            Shot.k += 1
            if Shot.k == int(n):
                base_win().saveScreenshot(Filename(path))
                print("screenshot", path)
                application.quit()
                sys.exit(0)
    Shot()

app.run()
