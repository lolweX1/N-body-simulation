import pyvista as pv
import numpy as np
import time
import math

# Create a plotter
pl = pv.Plotter()

bodies = []
objects = []

info = input("input body, format: 'radius,mass,x,y,z,f_x,f_y,f_z' - ")
while info != "clear":
    big_plot = info.split("*")
    for i in big_plot:
        data = i.split(",")
        bodies.append(
            {
                "radius": float(data[0]), 
                "mass": float(data[1]), 
                "position": [float(data[2]), float(data[3]), float(data[4])],
                "velocity": [float(data[5]), float(data[6]), float(data[7])],
                "force": [0.0, 0.0, 0.0]
            }
        )
    info = input("input body, format: 'radius,mass,x,y,z,f_x,f_y,f_z' - ")


# Add each sphere to the scene
for planet in bodies:
    s = pv.Sphere(radius=planet["radius"], center=(0, 0, 0))
    objects.append(pl.add_mesh(s, color="red"))

pl.iren.initialize()

delta_t = 20 # milliseconds
G = 1.0 # 0.00000000006674

def normalize_3_vector(arr):
    magnitude = math.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
    return [arr[0]/magnitude, arr[1]/magnitude, arr[2]/magnitude]

def update(step):
    for body in range(len(objects)):
        for force in range(body+1, len(objects)):
            distance = (bodies[body]["position"][0] - bodies[force]["position"][0])**2 + (bodies[body]["position"][1] - bodies[force]["position"][1])**2 + (bodies[body]["position"][2] - bodies[force]["position"][2])**2
            force_applied = G * bodies[body]["mass"] * bodies[force]["mass"] / distance
            direction_vector = [-bodies[body]["position"][0] + bodies[force]["position"][0], -bodies[body]["position"][1] + bodies[force]["position"][1], -bodies[body]["position"][2] + bodies[force]["position"][2]]
            normalized_direction_vector = normalize_3_vector(direction_vector)
            bodies[body]["force"][0] += force_applied * normalized_direction_vector[0]
            bodies[body]["force"][1] += force_applied * normalized_direction_vector[1]
            bodies[body]["force"][2] += force_applied * normalized_direction_vector[2]
            bodies[force]["force"][0] -= force_applied * normalized_direction_vector[0] # cuts process down by about half 
            bodies[force]["force"][1] -= force_applied * normalized_direction_vector[1]
            bodies[force]["force"][2] -= force_applied * normalized_direction_vector[2]

        acceleration = [bodies[body]["force"][0]/bodies[body]["mass"], bodies[body]["force"][1]/bodies[body]["mass"], bodies[body]["force"][2]/bodies[body]["mass"]]
        
        bodies[body]["velocity"][0] += acceleration[0] * (delta_t/1000)
        bodies[body]["velocity"][1] += acceleration[1] * (delta_t/1000)
        bodies[body]["velocity"][2] += acceleration[2] * (delta_t/1000)

        bodies[body]["position"][0] += bodies[body]["velocity"][0] * (delta_t/1000) 
        bodies[body]["position"][1] += bodies[body]["velocity"][1] * (delta_t/1000)
        bodies[body]["position"][2] += bodies[body]["velocity"][2] * (delta_t/1000)
        objects[body].position = bodies[body]["position"]

        bodies[body]["force"][0] = 0.0
        bodies[body]["force"][1] = 0.0
        bodies[body]["force"][2] = 0.0

pl.add_timer_event(max_steps = 8000, duration=16, callback=update)

pl.reset_camera()
pl.camera.zoom(0.8)

pl.show()

