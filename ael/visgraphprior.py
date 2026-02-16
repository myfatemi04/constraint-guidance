"""
Creates a prior distribution based on visibility graphs.
"""

import heapq
import json
from collections import defaultdict
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from ael.problem import Problem
from ael.visualize import visualize


def main_pyvisgraph():
    import pyvisgraph

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


def get_graph_without_vertices_in_obstacles(
    vor: Voronoi,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    agent_radius: float,
):
    """creates a graph of points on the Voronoi diagram that are not in the obstacles, as well as trimming tree branches"""
    graph = defaultdict(set)
    ok_vertices = (
        np.linalg.norm(
            # (v, :, 2) - (:, o, 2) -> (v, o, 2)
            vor.vertices[:, None, :] - obstacle_positions[None, :, :],
            axis=-1,
        )  # (v, o)
        >= (obstacle_radii + agent_radius)[None, :]
    ).all(axis=-1)  # (v,)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        assert len(simplex) == 2
        if np.all(simplex >= 0) and ok_vertices[simplex].all():
            graph[simplex[0]].add(simplex[1])
            graph[simplex[1]].add(simplex[0])

    return graph


def remove_tree_vertices(graph):
    """if a node is reached which has no *other* children, it is removed from the graph. assumes a connected graph."""
    rm = [0]
    while len(rm) > 0:
        rm = [node for node in graph if len(graph[node]) == 1]
        for node in rm:
            neighbor = next(iter(graph[node]))
            graph[neighbor].remove(node)
            del graph[node]


def voronoi_plot_2d(vor, ax, obstacle_positions, obstacle_radii, **kw):
    """
    From scipy
    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get("show_points", True):
        point_size = kw.get("point_size", None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], ".", markersize=point_size)
    if kw.get("show_vertices", True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], "o")

    line_colors = kw.get("line_colors", "k")
    line_width = kw.get("line_width", 1.0)
    line_alpha = kw.get("line_alpha", 1.0)

    # clip all vertices
    vertices = np.clip(vor.vertices, -1.5, 1.5)

    finite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        # check if either vertex is in an obstacle
        in_obs = False
        for vertex in vertices[simplex[simplex >= 0]]:
            for obstacle_pos, obstacle_radius in zip(
                obstacle_positions, obstacle_radii
            ):
                if np.linalg.norm(vertex - obstacle_pos) < (obstacle_radius + 0.05):
                    in_obs = True
        if in_obs:
            continue
        if np.all(simplex >= 0):
            finite_segments.append(vertices[simplex])

    ax.add_collection(
        LineCollection(
            finite_segments,
            colors=line_colors,
            lw=line_width,
            alpha=line_alpha,
            linestyle="solid",
        )
    )

    return ax.figure


def main_voronoi():
    with open("instances_data/instances_simple.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[10])

    polygons = []
    circle_approximation_num_sides = 32

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
    all_points = []
    for polygon in polygons:
        all_points.extend(polygon)

    min_x = -1.5
    max_x = 1.5
    min_y = -1.5
    max_y = 1.5
    all_points.extend(np.array([max_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([min_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), max_y * np.ones(32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), min_y * np.ones(32)]).T)

    all_points = np.array(all_points)

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)
    plt.plot(
        [min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], "r-"
    )
    vor = Voronoi(all_points)
    # remove points inside obstacles (just using the radius check)
    voronoi_plot_2d(
        vor,
        plt.gca(),
        problem.obstacle_positions,
        problem.obstacle_radii,
        show_points=False,
        show_vertices=False,
    )
    # adjust ax limits
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


def read_from_parents(parents, goal_vertex_id):
    path = []
    current = goal_vertex_id
    while current in parents:
        path.append(current)
        current = parents[current]
    path.append(current)
    return list(reversed(path))


def astar(graph, vertices, start_vertex_id, goal_vertex_id):
    # (heuristic, vertex_id)
    parents = {}
    queue = [
        (
            np.linalg.norm(vertices[start_vertex_id] - vertices[goal_vertex_id]),
            start_vertex_id,
        )
    ]
    visited = {start_vertex_id}
    while len(queue) > 0:
        _, vertex_id = heapq.heappop(queue)
        if vertex_id == goal_vertex_id:
            return read_from_parents(parents, goal_vertex_id)
        for neighbor in graph[vertex_id]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            trav_cost = np.linalg.norm(vertices[neighbor] - vertices[vertex_id])
            heur_remain = np.linalg.norm(vertices[neighbor] - vertices[goal_vertex_id])
            heapq.heappush(
                queue,
                (heur_remain + trav_cost, neighbor),
            )
            if neighbor not in parents:
                parents[neighbor] = vertex_id
    return None


def topk_shortest_paths(graph, vertices, start_vertex_id, goal_vertex_id, k):
    # identifies the top k paths from start to goal. maybe this is actually done more effectively with Dijkstra.
    import networkx as nx

    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(
                node,
                neighbor,
                weight=np.linalg.norm(vertices[neighbor] - vertices[node]),
            )
    results = []
    for i, path in enumerate(
        nx.shortest_simple_paths(G, start_vertex_id, goal_vertex_id, weight="weight")
    ):
        if i >= k:
            break
        length = (
            sum(G[path[j]][path[j + 1]]["weight"] for j in range(len(path) - 1))
            if len(path) > 1
            else 0
        )
        results.append((path, length))
    return results


def create_interpolated_path(vertices, path, dt, speed):
    """Assumes caller has ensured total_length <= dt * speed"""
    points = []
    total_time = 0
    for i in range(len(path) - 1):
        start_vertex = vertices[path[i]]
        end_vertex = vertices[path[i + 1]]
        segment_length = np.linalg.norm(end_vertex - start_vertex)
        segment_time = segment_length / speed
        total_time += segment_time
        num_points_in_segment = int((total_time - len(points) * dt) / dt)
        if num_points_in_segment > 0:
            segment = np.linspace(
                start_vertex, end_vertex, num_points_in_segment, endpoint=False
            )
            points.extend(segment)
    points.append(vertices[path[-1]])
    return np.array(points)


def generate_sample_trajectories():
    with open("instances_data/instances_dense.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[10], "numpy")

    polygons = []
    circle_approximation_num_sides = 32

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
    all_points = []
    for polygon in polygons:
        all_points.extend(polygon)

    min_x = -1.5
    max_x = 1.5
    min_y = -1.5
    max_y = 1.5
    all_points.extend(np.array([max_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([min_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), max_y * np.ones(32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), min_y * np.ones(32)]).T)

    all_points = np.array(all_points)

    vor = Voronoi(all_points)
    g = get_graph_without_vertices_in_obstacles(
        vor, problem.obstacle_positions, problem.obstacle_radii, problem.agent_radii[0]
    )
    remove_tree_vertices(g)

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    # Relabel the vertices
    indices = sorted(g.keys())
    vertices = vor.vertices[indices]
    vertex_mapping = {indices[i]: i for i in range(len(indices))}
    g = {
        vertex_mapping[node]: set(vertex_mapping[neighbor] for neighbor in neighbors)
        for node, neighbors in g.items()
    }

    # Add start and goal positions.
    agent_i = 1
    v0 = problem.agent_start_positions[agent_i]
    v1 = problem.agent_end_positions[agent_i]
    startgoal = np.vstack([v0, v1])
    closest_to_startgoal = np.argmin(
        np.linalg.norm(vertices[:, None, :] - startgoal[None, :, :], axis=-1), axis=0
    )
    vertices = np.concatenate([vertices, startgoal], axis=0)
    g[len(vertices) - 2] = {closest_to_startgoal[0]}
    g[len(vertices) - 1] = {closest_to_startgoal[1]}
    g[closest_to_startgoal[0]].add(len(vertices) - 2)
    g[closest_to_startgoal[1]].add(len(vertices) - 1)

    # plot graph
    for node in g:
        for neighbor in g[node]:
            plt.plot(
                [vertices[node, 0], vertices[neighbor, 0]],
                [vertices[node, 1], vertices[neighbor, 1]],
                "k-",
            )

    topk_paths = topk_shortest_paths(
        g, vertices, len(vertices) - 2, len(vertices) - 1, k=5
    )

    for path, length in topk_paths:
        plt.plot(
            vertices[path, 0],
            vertices[path, 1],
            "-",
            linewidth=2,
            label=f"length {length:.2f}",
        )

    plt.show()

    speed = 0.05
    dt = 1.0
    num_points = 64

    for path, length in topk_paths:
        traj = create_interpolated_path(vertices, path, dt, speed)
        traj_points = np.zeros((num_points, 2))
        traj_points[: len(traj)] = traj
        traj_points[len(traj) :] = traj[-1]
        plt.plot(traj_points[:, 0], traj_points[:, 1], "-o", markersize=2)

    plt.show()


if __name__ == "__main__":
    generate_sample_trajectories()
