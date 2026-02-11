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


def create_observed_polygon_graph(pieces, polygons, start_location):
    # Representation strategy:
    observed_polygon_points = []
    observed_polygon_source_obstacle_indices = []
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

        # Create polygon for the observed piece.
        observed_polygon = np.array([start_location, seen_point, next_seen_point])
        observed_polygon_points.append(observed_polygon)
        observed_polygon_source_obstacle_indices.append(obstacle_i)

        # Plot centroid of observed polygon.
        centroid = np.mean(observed_polygon, axis=0)
        plt.plot(centroid[0], centroid[1], color + "x")

    # create edges
    for i in range(len(observed_polygon_points)):
        observed_polygon_graph_neighbors[i].add((i + 1) % len(observed_polygon_points))
        observed_polygon_graph_neighbors[i].add((i - 1) % len(observed_polygon_points))

    return (
        observed_polygon_points,
        observed_polygon_source_obstacle_indices,
        observed_polygon_graph_neighbors,
    )


def identify_frontier_points(observed_polygon_points):
    # for numerical stability
    eps = 1e-6
    sweep_point = observed_polygon_points[0][1]
    frontier_points = []

    # add the first polygon twice, to give a chance in case point1 of the first polygon is a sudden jump deeper from point2 of the last polygon.
    # points can be added if they are at the start of the new polygon and deeper or the end of the old polygon and deeper.
    for observed_polygon in [*observed_polygon_points, observed_polygon_points[0]]:
        viewer_pos, point1, point2 = observed_polygon

        # if point1's depth != sweep_depth, then whichever point is deeper is the frontier point.
        point1_depth = np.linalg.norm(point1 - viewer_pos)
        sweep_depth = np.linalg.norm(sweep_point - viewer_pos)
        if abs(point1_depth - sweep_depth) > eps:
            frontier_points.append(sweep_point)
            frontier_points.append(point1)

        sweep_point = point2

    return frontier_points


def visualize_visibility_border(polygons):
    for polygon in polygons:
        plt.plot(polygon[1:, 0], polygon[1:, 1], "k-")


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

    visibility_polygons = []

    (
        observed_polygons,
        observed_polygon_source_obstacle_indices,
        observed_polygon_graph_neighbors,
    ) = create_observed_polygon_graph(pieces, polygons, start_location)

    polygons.extend(observed_polygons)
    visibility_polygons.extend(observed_polygons)

    # pick any point at the frontier. these are points on visibility polygons where there were sudden jumps in the depth.
    frontier = identify_frontier_points(observed_polygons)
    furthest_frontier_point = max(
        frontier, key=lambda point: np.linalg.norm(point - start_location)
    )
    visualize_visibility_border(observed_polygons)
    for point in frontier:
        plt.scatter(point[0], point[1], c="r", marker="x")
    plt.scatter(
        furthest_frontier_point[0], furthest_frontier_point[1], c="g", marker="x"
    )
    plt.show()

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    start_location = furthest_frontier_point

    pieces = identify_visible_pieces(start_location, list(polygons))
    (
        observed_polygons,
        observed_polygon_source_obstacle_indices,
        observed_polygon_graph_neighbors,
    ) = create_observed_polygon_graph(pieces, polygons, start_location)

    for polygon in polygons:
        for i in range(len(polygon)):
            plt.plot(
                polygon[[i, (i + 1) % len(polygon)], 0],
                polygon[[i, (i + 1) % len(polygon)], 1],
                "k-",
                c="blue",
            )

    for polygon in observed_polygons:
        for i in range(len(polygon)):
            plt.plot(
                polygon[[i, (i + 1) % len(polygon)], 0],
                polygon[[i, (i + 1) % len(polygon)], 1],
                "k-",
                c="red",
            )

    plt.scatter(start_location[0], start_location[1], c="g", marker="o")
    plt.show()


if __name__ == "__main__":
    main()
