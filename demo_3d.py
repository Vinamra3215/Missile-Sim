"""
MissileSim 3D Demo — Ursina Engine
Run: python demo_3d.py
Controls: Right-click + drag to rotate, scroll to zoom, middle-click to pan
"""
import math
import random
import sys

from ursina import *

sys.path.insert(0, '.')
from env.missile_physics import Missile
from env.obstacles import build_level, check_obstacle_collision, check_wall_collision
from configs.default import *

# ── scale ──
S = 0.1


def e2w(ex, ey):
    """env coords → 3D world"""
    return ex * S, (300 - ey) * S, 0


app = Ursina(borderless=False, title='MissileSim 3D', size=(1100, 750))
window.color = color.black

# editor camera (mouse-controlled: right-click+drag rotate, scroll zoom)
ec = EditorCamera()
ec.position = Vec3(8, 5, -20)
ec.rotation = Vec3(15, -10, 0)

# build level
zones, all_obstacles, boss_pos, checkpoints = build_level(ENV_HEIGHT)
D = 10

# ═══════════════════════════════════════
# LEVEL GEOMETRY
# ═══════════════════════════════════════

# ── Launch Pad ──
Entity(model='cube', color=color.rgb(35, 140, 50),
       position=(200*S, -30*S, 0), scale=(500*S, 5*S, D*1.5))
Entity(model='cube', color=color.rgb(130, 130, 130),
       position=(80*S, -28*S, 0), scale=(4, 0.5, 4))
# factory
Entity(model='cube', color=color.rgb(120, 120, 130),
       position=(-5, 5*S, -6), scale=(10, 8, 5))
Entity(model='cube', color=color.rgb(90, 90, 100),
       position=(-3, 20*S, -6), scale=(2.5, 12, 2.5))

# ── Chimney walls ──
Entity(model='cube', color=color.rgb(100, 100, 110),
       position=(525*S, (300-70)*S, 0), scale=(250*S, 140*S, D))
Entity(model='cube', color=color.rgb(100, 100, 110),
       position=(525*S, (300-530)*S, 0), scale=(250*S, 340*S, D))

# ── Shaft walls ──
shaft_cx = (650+225)*S
Entity(model='cube', color=color.rgb(60, 60, 80),
       position=(shaft_cx, 0, D/2+0.5), scale=(450*S, 60*S, 1))
Entity(model='cube', color=color.rgb(60, 60, 80),
       position=(shaft_cx, 0, -D/2-0.5), scale=(450*S, 60*S, 1))
Entity(model='cube', color=color.rgb(140, 90, 80),
       position=((650+180)*S, (300-220)*S, 0), scale=(60*S, 40*S, D*0.7))
Entity(model='cube', color=color.rgb(140, 90, 80),
       position=((650+340)*S, (300-395)*S, 0), scale=(70*S, 35*S, D*0.7))

# ── Pipe Corridor ──
pc_cx = (1100+350)*S
Entity(model='cube', color=color.rgb(180, 100, 80),
       position=(pc_cx, 27*S, 0), scale=(700*S, 6*S, D))
Entity(model='cube', color=color.rgb(180, 100, 80),
       position=(pc_cx, -27*S, 0), scale=(700*S, 6*S, D))
for px, py, pw in [(1200,150,100),(1350,350,120),(1500,200,90),(1600,420,110),(1700,280,100)]:
    y3 = (300-py)*S
    Entity(model='cube', color=color.rgb(200, 110, 80),
           position=(px*S, y3, 0), scale=(pw*S, 2.5*S, D*0.8))

# ── Factory Hall ──
hc = (1800+400)*S
Entity(model='cube', color=color.rgb(180, 110, 75),
       position=(hc, 28*S, 0), scale=(800*S, 5*S, D))
Entity(model='cube', color=color.rgb(180, 110, 75),
       position=(hc, -27*S, 0), scale=(800*S, 7*S, D))
# pillars
for px in [1900, 2100, 2300]:
    Entity(model='cube', color=color.rgb(150, 85, 55),
           position=(px*S, 0, D/2-1.5), scale=(3*S, 20*S, 3*S))
# conveyor
for i in range(25):
    cx = (1850+i*20)*S
    col = color.rgb(220, 200, 40) if i%2==0 else color.rgb(40, 40, 40)
    Entity(model='cube', color=col, position=(cx, -21*S, -1), scale=(1.8*S, 1.2*S, 2.5))
# boxes
for i in range(6):
    Entity(model='cube', color=color.rgb(220, 140, 40),
           position=((1870+i*80)*S, -18*S, -1), scale=(1.2, 1.2, 1.2))
# workers
for i in range(6):
    wx = (1860+i*80)*S
    Entity(model='cube', color=color.rgb(200, 60, 60),
           position=(wx, -15*S, -3), scale=(0.6, 1.3, 0.6))
    Entity(model='cube', color=color.rgb(220, 180, 140),
           position=(wx, -12*S, -3), scale=(0.45, 0.45, 0.45))
    Entity(model='cube', color=color.rgb(220, 160, 30),
           position=(wx, -11*S, -3), scale=(0.65, 0.2, 0.65))

# ── Boss Arena ──
ac = (2500+250)*S
Entity(model='cube', color=color.rgb(160, 80, 60),
       position=(ac, 28*S, 0), scale=(500*S, 5*S, D))
Entity(model='cube', color=color.rgb(160, 80, 60),
       position=(ac, -27*S, 0), scale=(500*S, 7*S, D))
Entity(model='cube', color=color.rgb(160, 80, 60),
       position=(2880*S, -17*S, 0), scale=(12*S, 10*S, D*0.8))

# ── Boss ──
boss = Entity(model='cube', color=color.rgb(50, 160, 80),
              position=(2900*S, -8*S, 0), scale=(1.2, 2.2, 0.9))
Entity(model='cube', color=color.rgb(220, 180, 140),
       parent=boss, position=(0, 1.4, 0), scale=(0.6, 0.6, 0.6))
Entity(model='cube', color=color.rgb(100, 70, 40),
       parent=boss, position=(0, 1.1, 0.3), scale=(0.5, 0.4, 0.3))
Entity(model='cube', color=color.rgb(50, 50, 70),
       parent=boss, position=(-0.2, -1.3, 0), scale=(0.3, 0.9, 0.3))
Entity(model='cube', color=color.rgb(50, 50, 70),
       parent=boss, position=(0.2, -1.3, 0), scale=(0.3, 0.9, 0.3))
boss_arm = Entity(model='cube', color=color.rgb(50, 160, 80),
                  parent=boss, position=(0.6, 0.7, 0), rotation=(0,0,-50),
                  scale=(0.2, 0.8, 0.2))

# ═══════════════════════════════════════
# ROCKET
# ═══════════════════════════════════════
rocket = Entity(model='cube', color=color.rgb(200, 220, 240), scale=(2.2, 0.55, 0.55))
# nose
Entity(model='cube', color=color.rgb(160, 160, 170),
       parent=rocket, position=(0.8, 0, 0), scale=(0.5, 0.45, 0.45))
# fins
for fy, fz in [(0.35, 0), (-0.35, 0), (0, 0.35), (0, -0.35)]:
    Entity(model='cube', color=color.rgb(70, 70, 90),
           parent=rocket, position=(-0.6, fy, fz),
           scale=(0.5, 0.35 if fz==0 else 0.1, 0.1 if fz==0 else 0.35))
# flames
flame_out = Entity(model='cube', color=color.rgb(255, 130, 20),
                   parent=rocket, position=(-1.2, 0, 0),
                   scale=(1, 0.35, 0.35), enabled=False)
flame_in = Entity(model='cube', color=color.rgb(255, 220, 60),
                  parent=rocket, position=(-0.9, 0, 0),
                  scale=(0.6, 0.18, 0.18), enabled=False)

# ═══════════════════════════════════════
# HUD
# ═══════════════════════════════════════
hud = Text(text='', position=(-0.85, 0.46), scale=1.3, color=color.white)
zone_txt = Text(text='Launch Pad', position=(0, 0.46), origin=(0,0),
                scale=2, color=color.cyan)
status_txt = Text(text='', position=(0, 0), origin=(0,0),
                  scale=3, color=color.yellow, enabled=False)
help_txt = Text(text='[C] Toggle chase cam  |  Right-click+drag: rotate  |  Scroll: zoom',
                position=(0, -0.46), origin=(0,0), scale=1, color=color.gray)

# ═══════════════════════════════════════
# STATE
# ═══════════════════════════════════════
missile = Missile(80, 300, heading=-0.2)
game_over = False
steps = 0
restart_timer = 0
chase_mode = False


def input(key):
    """Toggle between editor camera and chase camera."""
    global chase_mode
    if key == 'c':
        chase_mode = not chase_mode
        ec.enabled = not chase_mode
        if chase_mode:
            help_txt.text = '[C] Toggle editor cam  |  Chase camera active'
        else:
            help_txt.text = '[C] Toggle chase cam  |  Right-click+drag: rotate  |  Scroll: zoom'
        print(f"Camera mode: {'Chase' if chase_mode else 'Editor'}")


def update():
    global game_over, steps, missile, restart_timer

    try:
        # auto-restart after crash
        if game_over:
            restart_timer += time.dt
            if restart_timer > 3:
                missile = Missile(80, 300, heading=-0.2)
                game_over = False
                steps = 0
                restart_timer = 0
                status_txt.enabled = False
            return

        # random action
        action = random.randint(0, 6)
        missile.update(action)
        steps += 1
        mx, my = missile.x, missile.y

        # update rocket 3D
        x3, y3, _ = e2w(mx, my)
        rocket.position = Vec3(x3, y3, 0)
        rocket.rotation_z = -math.degrees(missile.heading)

        # flame
        if missile.thrusting and missile.speed > 0.2:
            flame_out.enabled = True
            flame_in.enabled = True
            fs = 0.7 + missile.speed * 0.12 + random.uniform(-0.1, 0.1)
            flame_out.scale_x = fs
            flame_in.scale_x = fs * 0.55
        else:
            flame_out.enabled = False
            flame_in.enabled = False

        # boss arm wave
        boss_arm.rotation_z = -50 + math.sin(time.time() * 3) * 25

        # chase camera
        if chase_mode:
            camera.position = lerp(camera.position,
                                   Vec3(x3 - 12, y3 + 5, -14), time.dt * 5)
            camera.look_at(Vec3(x3 + 5, y3, 0))

        # zone
        zone_name = ""
        for z in zones:
            if z.x_start <= mx < z.x_end:
                zone_name = z.name
                break

        bd = math.sqrt((mx - boss_pos[0])**2 + (my - boss_pos[1])**2)
        prog = min(100, mx / 3000 * 100)

        hud.text = f'Step:{steps}  Speed:{missile.speed:.1f}  Boss:{bd:.0f}'
        zone_txt.text = f'{zone_name}  ({prog:.0f}%)'

        # collisions
        if bd < 40:
            game_over = True
            status_txt.text = 'TARGET HIT!'
            status_txt.color = color.lime
            status_txt.enabled = True
        elif check_obstacle_collision(mx, my, MISSILE_RADIUS, all_obstacles):
            game_over = True
            status_txt.text = f'CRASHED in {zone_name}!'
            status_txt.color = color.red
            status_txt.enabled = True
        elif check_wall_collision(mx, my, ENV_HEIGHT) or mx < -50:
            game_over = True
            status_txt.text = 'OUT OF BOUNDS!'
            status_txt.color = color.red
            status_txt.enabled = True
        elif steps >= MAX_STEPS:
            game_over = True
            status_txt.text = 'TIME UP!'
            status_txt.color = color.yellow
            status_txt.enabled = True

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()


print("\n=== MissileSim 3D ===")
print("Controls:")
print("  Right-click + drag  → Rotate camera")
print("  Scroll wheel        → Zoom in/out")
print("  Middle-click + drag → Pan camera")
print("  [C] key             → Toggle chase camera")
print("  Close window to exit\n")

app.run()
