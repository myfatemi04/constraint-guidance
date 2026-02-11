"""
Creates a prior distribution based on visibility graphs.
"""

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pyvisgraph

from ael.problem import Problem


def main_pyvisgraph():
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


def get_distance_at_theta(start_location, theta, vertex, next_vertex):
    mat = np.array(
        [
            [np.cos(theta), (vertex - next_vertex)[0]],
            [np.sin(theta), (vertex - next_vertex)[1]],
        ]
    )
    distance, _beta = np.linalg.inv(mat) @ np.array(
        [vertex[0] - start_location[0], vertex[1] - start_location[1]]
    )
    return distance


def main():
    with open("instances_data/instances_dense.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[0])

    polygons = np.zeros((problem.num_obstacles, 6, 2))
    circle_approximation_num_sides = 6

    for obstacle_i in range(problem.num_obstacles):
        x, y = problem.obstacle_positions[obstacle_i]
        r = problem.obstacle_radii[obstacle_i] + problem.agent_radii[0]
        polygons[obstacle_i] = [
            [
                x + r * np.cos(j * 2 * np.pi / circle_approximation_num_sides),
                y + r * np.sin(j * 2 * np.pi / circle_approximation_num_sides),
            ]
            for j in range(circle_approximation_num_sides)
        ]

    start_location = problem.agent_start_positions[0]

    # Iterate over all points outside of start position.
    # Edges shall be identified by (polygon_index, vertex_index).
    # Vertices stored will be in order of (angle, edge identifier).
    # Then, consecutive vertices can be checked for visibility.
    # We assume that all edges are non-intersecting line segments.
    events = []
    for obstacle_i in range(problem.num_obstacles):
        thetas = np.arctan2(
            polygons[obstacle_i, :, 1] - start_location[1],
            polygons[obstacle_i, :, 0] - start_location[0],
        )
        for vertex_i in range(circle_approximation_num_sides):
            edge_start_theta = thetas[vertex_i]
            edge_end_theta = thetas[(vertex_i + 1) % circle_approximation_num_sides]
            # check that this edge has the correct orientation.
            cross_prod = np.cross(
                polygons[obstacle_i, vertex_i] - start_location,
                polygons[obstacle_i, (vertex_i + 1) % circle_approximation_num_sides]
                - start_location,
            )
            if cross_prod > 0:
                # the vector that points to the direction of the edge must move clockwise for this edge to be visible
                continue

            # counterintuitively, we should have that the theta for the second point in the edge is smaller than the theta for the first point in the edge
            if edge_end_theta > edge_start_theta:
                edge_end_theta = edge_end_theta - 2 * np.pi
            edge_identifier = (obstacle_i, vertex_i)
            events.append((edge_end_theta, edge_identifier, "add"))
            events.append((edge_start_theta, edge_identifier, "remove"))

    events = sorted(events)

    # Now, identify the points that are visible from the start location.
    # In the `active_edges` list, edges are kept in the order of their distance from the start location.
    active_edges = []
    # In the `pieces` list, we log whenever there's a change in the closest edge (which is active_edges[0]).
    # The theta in `pieces` indicates the *start theta* for that piece.
    pieces = []
    leading_edge = None
    for theta, edge_identifier, action in events:
        if action == "remove":
            active_edges.remove(edge_identifier)
            if leading_edge == edge_identifier:
                new_leading_edge = active_edges[0] if len(active_edges) > 0 else None
                pieces.append((theta, new_leading_edge))
                leading_edge = new_leading_edge

        elif action == "add":
            distances_to_active_edges = []
            for obstacle_i, vertex_i in active_edges:
                vertex = polygons[obstacle_i, vertex_i]
                next_vertex = polygons[
                    obstacle_i, (vertex_i + 1) % circle_approximation_num_sides
                ]
                distances_to_active_edges.append(
                    get_distance_at_theta(start_location, theta, vertex, next_vertex)
                )

            obstacle_i, vertex_i = edge_identifier
            my_dist = np.linalg.norm(polygons[obstacle_i, vertex_i] - start_location)

            insertion_index = 0
            while insertion_index < len(distances_to_active_edges):
                dist = distances_to_active_edges[insertion_index]

                if dist > my_dist:
                    break

                insertion_index += 1

            active_edges.insert(insertion_index, edge_identifier)

            if insertion_index == 0:
                pieces.append((theta, edge_identifier))
                leading_edge = edge_identifier

    # remove earlier pieces if there is the same theta.
    final_pieces = []
    for theta, edge_identifier in pieces:
        if len(final_pieces) > 0 and final_pieces[-1][0] == theta:
            final_pieces[-1] = (theta, edge_identifier)
        else:
            final_pieces.append((theta, edge_identifier))

    pieces = final_pieces

    from ael.visualize import visualize

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    for piece_i, (theta, edge_identifier) in enumerate(pieces):
        if edge_identifier is None:
            continue

        obstacle_i, vertex_i = edge_identifier
        vertex = polygons[obstacle_i, vertex_i]
        next_vertex = polygons[
            obstacle_i, (vertex_i + 1) % circle_approximation_num_sides
        ]
        distance = get_distance_at_theta(start_location, theta, vertex, next_vertex)
        seen_point = start_location + distance * np.array(
            [np.cos(theta), np.sin(theta)]
        )

        # Plot the edge itself.
        # plt.plot(
        #     [
        #         polygons[obstacle_i, vertex_i, 0],
        #         polygons[
        #             obstacle_i, (vertex_i + 1) % circle_approximation_num_sides, 0
        #         ],
        #     ],
        #     [
        #         polygons[obstacle_i, vertex_i, 1],
        #         polygons[
        #             obstacle_i, (vertex_i + 1) % circle_approximation_num_sides, 1
        #         ],
        #     ],
        #     "k-",
        #     alpha=0.5,
        # )

        # This section of the edge stops being seen at the next theta.
        next_theta = pieces[(piece_i + 1) % len(pieces)][0]
        next_distance = get_distance_at_theta(
            start_location, next_theta, vertex, next_vertex
        )
        next_seen_point = start_location + next_distance * np.array(
            [np.cos(next_theta), np.sin(next_theta)]
        )

        # Plot line of sight to first seen point.
        plt.plot(
            [start_location[0], seen_point[0]],
            [start_location[1], seen_point[1]],
            c="gray",
            alpha=0.2,
        )
        # Plot line of sight to next seen point.
        plt.plot(
            [start_location[0], next_seen_point[0]],
            [start_location[1], next_seen_point[1]],
            c="gray",
            alpha=0.2,
        )
        # Plot line of sight between the seen points.
        plt.plot(
            [seen_point[0], next_seen_point[0]],
            [seen_point[1], next_seen_point[1]],
            "r-",
        )

    plt.show()


if __name__ == "__main__":
    main()
