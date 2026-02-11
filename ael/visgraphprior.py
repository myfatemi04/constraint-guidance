"""
Creates a prior distribution based on visibility graphs.
"""

import json
from collections import defaultdict
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


def identify_visible_pieces(start_location, polygons):
    # Iterate over all points outside of start position.
    # Edges shall be identified by (polygon_index, vertex_index).
    # Vertices stored will be in order of (angle, edge identifier).
    # Then, consecutive vertices can be checked for visibility.
    # We assume that all edges are non-intersecting line segments.
    events = []
    active_at_branch_cut = []
    for obstacle_i in range(len(polygons)):
        polygon = polygons[obstacle_i]
        thetas = np.arctan2(
            polygon[:, 1] - start_location[1], polygon[:, 0] - start_location[0]
        )
        npts = len(polygon)
        for vertex_i in range(npts):
            remove_edge_theta = thetas[vertex_i]
            add_edge_theta = thetas[(vertex_i + 1) % npts]
            # check that this edge has the correct orientation.
            cross_prod = np.cross(
                polygon[vertex_i] - start_location,
                polygon[(vertex_i + 1) % npts] - start_location,
            )
            if cross_prod > 0:
                # the vector that points to the direction of the edge must move clockwise for this edge to be visible
                continue

            edge_identifier = (obstacle_i, vertex_i)
            events.append((add_edge_theta, edge_identifier, "add"))
            events.append((remove_edge_theta, edge_identifier, "remove"))
            if add_edge_theta > remove_edge_theta:
                active_at_branch_cut.append(edge_identifier)

    events = sorted(events)

    # Now, identify the points that are visible from the start location.
    # In the `active_edges` list, edges are kept in the order of their distance from the start location.
    active_edges = active_at_branch_cut
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
                polygon = polygons[obstacle_i]
                vertex = polygon[vertex_i]
                next_vertex = polygon[(vertex_i + 1) % len(polygon)]
                distances_to_active_edges.append(
                    get_distance_at_theta(start_location, theta, vertex, next_vertex)
                )

            obstacle_i, vertex_i = edge_identifier
            polygon = polygons[obstacle_i]
            my_dist = np.linalg.norm(polygon[vertex_i] - start_location)

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
    prev_theta = None
    for theta, edge_identifier in pieces:
        if theta == prev_theta:
            final_pieces[-1] = (theta, edge_identifier)
        else:
            final_pieces.append((theta, edge_identifier))
        prev_theta = theta

    # import pprint
    # pprint.pprint(final_pieces)

    return final_pieces


def main():
    with open("instances_data/instances_dense.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[10])

    polygons = []
    circle_approximation_num_sides = 6

    for obstacle_i in range(problem.num_obstacles):
        x, y = problem.obstacle_positions[obstacle_i]
        r = problem.obstacle_radii[obstacle_i] + problem.agent_radii[0]
        polygons.append(
            np.array(
                [
                    [
                        x + r * np.cos(j * 2 * np.pi / circle_approximation_num_sides),
                        y + r * np.sin(j * 2 * np.pi / circle_approximation_num_sides),
                    ]
                    for j in range(circle_approximation_num_sides)
                ]
            )
        )

    # Note: order here must be counter clockwise to represent the flipped orientation.
    polygons.append(
        np.array(
            [
                [-1, -1],
                [-1, 1],
                [1, 1],
                [1, -1],
            ]
        )
    )

    start_location = problem.agent_start_positions[0]

    pieces = identify_visible_pieces(start_location, list(polygons))

    from ael.visualize import visualize

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    # Representation strategy:
    observed_polygons = []
    observed_polygon_graph_neighbors = defaultdict(set)

    for piece_i, (theta, edge_identifier) in enumerate(pieces):
        obstacle_i, vertex_i = edge_identifier

        color = "r" if edge_identifier == (9, 5) else "b"

        polygon = polygons[obstacle_i]
        vertex = polygon[vertex_i]
        next_vertex = polygon[(vertex_i + 1) % len(polygon)]
        curr_theta_distance = get_distance_at_theta(
            start_location, theta, vertex, next_vertex
        )
        seen_point = start_location + curr_theta_distance * np.array(
            [np.cos(theta), np.sin(theta)]
        )

        # This section of the edge stops being seen at the next theta.
        next_theta = pieces[(piece_i + 1) % len(pieces)][0]
        next_theta_distance = get_distance_at_theta(
            start_location, next_theta, vertex, next_vertex
        )
        next_seen_point = start_location + next_theta_distance * np.array(
            [np.cos(next_theta), np.sin(next_theta)]
        )

        # make line from current depth to depth of previous polygon
        (_prev_theta_start, (prev_obstacle_i, prev_first_vertex_i)) = pieces[
            piece_i - 1
        ]
        prev_second_vertex_i = (prev_first_vertex_i + 1) % len(
            polygons[prev_obstacle_i]
        )
        prev_first_vertex = polygons[prev_obstacle_i][prev_first_vertex_i]
        prev_second_vertex = polygons[prev_obstacle_i][prev_second_vertex_i]
        prev_theta_distance = get_distance_at_theta(
            start_location,
            theta,
            prev_first_vertex,
            prev_second_vertex,
        )

        plt.plot(
            [
                start_location[0] + curr_theta_distance * np.cos(theta),
                start_location[0] + prev_theta_distance * np.cos(theta),
            ],
            [
                start_location[1] + curr_theta_distance * np.sin(theta),
                start_location[1] + prev_theta_distance * np.sin(theta),
            ],
            c="r",
            alpha=0.2,
        )

        # Plot line of sight between the seen points.
        plt.plot(
            [seen_point[0], next_seen_point[0]],
            [seen_point[1], next_seen_point[1]],
            color + "-",
        )

        # Create polygon for the observed piece.
        observed_polygon = np.array([start_location, seen_point, next_seen_point])
        observed_polygons.append(observed_polygon)

        # Plot centroid of observed polygon.
        centroid = np.mean(observed_polygon, axis=0)
        plt.plot(centroid[0], centroid[1], color + "x")

    # create edges
    for i in range(len(observed_polygons)):
        observed_polygon_graph_neighbors[i].add((i + 1) % len(observed_polygons))
        observed_polygon_graph_neighbors[i].add((i - 1) % len(observed_polygons))

    plt.show()

    # OK, now to make this into a complete tree search, we can create a frontier of edges to explore from.
    # Then whenever one of these edges has a "visible piece" on the observed region, we can sort of do a
    # loop closure type of thing.

    start_location = problem.agent_start_positions[0]

    pieces = identify_visible_pieces(start_location, list(polygons))


if __name__ == "__main__":
    main()
