"""
Creates a prior distribution based on visibility graphs.
"""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pyvisgraph

from ael.problem import Problem


def main():
    with open("instances_data/instances_dense.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[0])

    polygons = []
    circle_approximation_num_sides = 6

    for i in range(problem.num_obstacles):
        x, y = problem.obstacle_positions[i]
        r = problem.obstacle_radii[i]
        polygon = [
            pyvisgraph.Point(
                x + r * np.cos(j * 2 * np.pi / circle_approximation_num_sides),
                y + r * np.sin(j * 2 * np.pi / circle_approximation_num_sides),
            )
            for j in range(circle_approximation_num_sides)
        ]
        polygons.append(polygon)

    vg = pyvisgraph.VisGraph()
    vg.build(polygons)

    # Render the graph.
    graph = cast(pyvisgraph.Graph, vg.visgraph)
    for node, edges in graph.graph.items():
        for edge in edges:
            neighbor = edge.get_adjacent(node)
            plt.plot(
                [node.x, neighbor.x],
                [node.y, neighbor.y],
                "k-",
                alpha=0.5,
            )

    plt.show()


if __name__ == "__main__":
    main()
