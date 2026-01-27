"""
Visualizer to help debug the agent-obstacle score function. Allows you to click on a position on the map and modify the $\\sigma$ value to view the impact each obstacle has on the agent's overall score function.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

from ael.optimize import Problem

with open("instances_data/instances_dense.json", "r") as f:
    data = json.load(f)

problem = Problem.from_json(data[0])
problem.visualize()


t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)
fig, ax = plt.subplots()
ax.plot(t, s)


def on_move(event):
    if event.inaxes:
        print(
            f"data coords {event.xdata} {event.ydata},",
            f"pixel coords {event.x} {event.y}",
        )


def on_click(event):
    if event.button is MouseButton.LEFT:
        print("disconnecting callback")
        plt.disconnect(binding_id)


binding_id = plt.connect("motion_notify_event", on_move)
plt.connect("button_press_event", on_click)

plt.show()
