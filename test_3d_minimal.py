"""
Minimal 3D test to verify Ursina rendering works.
"""
from ursina import *

app = Ursina(borderless=False, title='3D Test', size=(800, 600))
window.color = color.black

# editor camera so you can look around with mouse
EditorCamera()

# big colored cubes at origin
Entity(model='cube', color=color.red, position=(0, 0, 0), scale=3)
Entity(model='cube', color=color.green, position=(5, 0, 0), scale=2)
Entity(model='cube', color=color.blue, position=(0, 3, 0), scale=2)
Entity(model='cube', color=color.yellow, position=(0, 0, 5), scale=2)

# ground
Entity(model='cube', color=color.olive, position=(0, -2, 0), scale=(20, 0.5, 20))

# text
Text(text='IF YOU SEE COLORED CUBES - URSINA WORKS!', position=(0, 0.45), origin=(0,0), scale=2, color=color.white)

print("Window should show colored cubes. Use mouse to look around.")
print("Right-click + drag to rotate, scroll to zoom.")

app.run()
