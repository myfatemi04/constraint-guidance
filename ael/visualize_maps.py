import json

import matplotlib.pyplot as plt

from ael.problem import Problem
from ael.visualize import visualize

plt.figure(figsize=(12, 4))

for i, set_name in enumerate(["connected_room", "shelf", "dense", "simple"]):
    with open(f"instances_data/instances_{set_name}.json", "r") as f:
        data = json.load(f)
    problem = Problem.from_json(data[0])
    plt.subplot(1, 4, i + 1)
    plt.title(set_name.replace("_", " ").title())
    visualize(problem, plt.gca(), start_markersize=0, end_markersize=0)

plt.tight_layout()
plt.savefig("figures/problem_sets.png")
