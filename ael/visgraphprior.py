"""
Creates a prior distribution based on visibility graphs.
"""

import heapq
import json
from collections import defaultdict
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
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

    for i in range(problem.num_circular_obstacles):
        x, y = problem.circular_obstacle_positions[i]
        r = problem.circular_obstacle_radii[i]
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

    for obstacle_i in range(problem.num_circular_obstacles):
        x, y = problem.circular_obstacle_positions[obstacle_i]
        r = problem.circular_obstacle_radii[obstacle_i] + problem.agent_radii[0]
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


def _get_graph_without_vertices_in_obstacles(
    vor: Voronoi,
    circular_obstacle_positions: np.ndarray,
    circular_obstacle_radii: np.ndarray,
    axis_aligned_box_obstacle_bounds: np.ndarray,
    agent_radius: float,
):
    """creates a graph of points on the Voronoi diagram that are not in the obstacles, as well as trimming tree branches"""
    graph = defaultdict(set)

    ok_vertices = np.ones(vor.vertices.shape[0], dtype=bool)

    if circular_obstacle_positions.size > 0:
        ok_vertices &= (
            np.linalg.norm(
                # (v, :, 2) - (:, o, 2) -> (v, o, 2)
                vor.vertices[:, None, :] - circular_obstacle_positions[None, :, :],
                axis=-1,
            )  # (v, o)
            >= (circular_obstacle_radii + agent_radius)[None, :]
        ).all(axis=-1)  # (v,)

    if axis_aligned_box_obstacle_bounds.size > 0:
        lower = axis_aligned_box_obstacle_bounds[:, 0, :] - agent_radius
        upper = axis_aligned_box_obstacle_bounds[:, 1, :] + agent_radius
        ok_vertices &= ~(
            (vor.vertices[:, None, :] >= lower[None, :, :])
            & (vor.vertices[:, None, :] <= upper[None, :, :])
        ).all(axis=-1).any(axis=-1)

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        assert len(simplex) == 2
        if np.all(simplex >= 0) and ok_vertices[simplex].all():
            graph[simplex[0]].add(simplex[1])
            graph[simplex[1]].add(simplex[0])

    return graph


def _remove_tree_vertices(graph):
    """if a node is reached which has no *other* children, it is removed from the graph. assumes a connected graph."""
    rm = [0]
    while len(rm) > 0:
        rm = [node for node in graph if len(graph[node]) == 1]
        for node in rm:
            neighbor = next(iter(graph[node]))
            graph[neighbor].remove(node)
            del graph[node]


def _voronoi_plot_2d(vor, ax, obstacle_positions, obstacle_radii, **kw):
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


def _create_voronoi_polygon(
    problem: Problem,
    circle_approximation_num_sides: int,
    min_x=-1.5,
    max_x=1.5,
    min_y=-1.5,
    max_y=1.5,
) -> np.ndarray:
    polygons = []
    circle_approximation_num_sides = 32

    for obstacle_i in range(problem.num_circular_obstacles):
        x, y = problem.circular_obstacle_positions[obstacle_i]
        r = problem.circular_obstacle_radii[obstacle_i] + problem.agent_radii[0]
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

    for obstacle_i in range(problem.num_axis_aligned_box_obstacles):
        lower = problem.axis_aligned_box_obstacle_bounds[obstacle_i, 0]
        upper = problem.axis_aligned_box_obstacle_bounds[obstacle_i, 1]
        # add the agent radius to the box size
        lower -= problem.agent_radii[0]
        upper += problem.agent_radii[0]
        base_points = np.array(
            [
                [lower[0], lower[1]],
                [lower[0], upper[1]],
                [upper[0], upper[1]],
                [upper[0], lower[1]],
            ]
        )
        polygons.append(
            np.array(
                [
                    *np.linspace(
                        base_points[0],
                        base_points[1],
                        circle_approximation_num_sides // 4,
                        endpoint=False,
                    ),
                    *np.linspace(
                        base_points[1],
                        base_points[2],
                        circle_approximation_num_sides // 4,
                        endpoint=False,
                    ),
                    *np.linspace(
                        base_points[2],
                        base_points[3],
                        circle_approximation_num_sides // 4,
                        endpoint=False,
                    ),
                    *np.linspace(
                        base_points[3],
                        base_points[0],
                        circle_approximation_num_sides // 4,
                        endpoint=False,
                    ),
                ]
            )
        )

    # Note: order here must be counter clockwise to represent the flipped orientation.
    all_points = []
    for polygon in polygons:
        all_points.extend(polygon)

    all_points.extend(np.array([max_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([min_x * np.ones(32), np.linspace(min_y, max_y, 32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), max_y * np.ones(32)]).T)
    all_points.extend(np.array([np.linspace(min_x, max_x, 32), min_y * np.ones(32)]).T)

    all_points = np.array(all_points)
    return all_points


def main_voronoi():
    with open("instances_data/instances_simple.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[10])

    min_x = -1.5
    max_x = 1.5
    min_y = -1.5
    max_y = 1.5
    all_points = _create_voronoi_polygon(problem, 32, min_x, max_x, min_y, max_y)

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)
    plt.plot(
        [min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], "r-"
    )
    vor = Voronoi(all_points)
    # remove points inside obstacles (just using the radius check)
    _voronoi_plot_2d(
        vor,
        plt.gca(),
        problem.circular_obstacle_positions,
        problem.circular_obstacle_radii,
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
        for neighbor in graph.neighbors(vertex_id):
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


def topk_shortest_paths(
    graph: nx.Graph, start_vertex_id, goal_vertex_id, k
) -> list[np.ndarray]:
    # Graph is already a NetworkX graph with weights
    results = []
    for i, path in enumerate(
        nx.shortest_simple_paths(
            graph, start_vertex_id, goal_vertex_id, weight="weight"
        )
    ):
        if i >= k:
            break

        results.append(
            np.array([graph.nodes[path[index]]["pos"] for index in range(len(path))])
        )

    return results


def interpolate(path: np.ndarray, dt, speed):
    """Assumes caller has ensured total_length <= dt * speed"""
    points = []
    total_time = 0
    for i in range(len(path) - 1):
        start_vertex = path[i]
        end_vertex = path[i + 1]
        segment_length = np.linalg.norm(end_vertex - start_vertex)
        segment_time = segment_length / speed
        total_time += segment_time
        num_points_in_segment = int((total_time - len(points) * dt) / dt)
        if num_points_in_segment > 0:
            segment = np.linspace(
                start_vertex, end_vertex, num_points_in_segment, endpoint=False
            )
            points.extend(segment)
    points.append(path[-1])
    return np.array(points)


def _get_voronoi_graph(problem: Problem, voronoi: Voronoi) -> nx.Graph:
    dict_graph = _get_graph_without_vertices_in_obstacles(
        voronoi,
        problem.circular_obstacle_positions,
        problem.circular_obstacle_radii,
        problem.axis_aligned_box_obstacle_bounds,
        problem.agent_radii[0],
    )
    _remove_tree_vertices(dict_graph)

    # Relabel the vertices after filtration.
    indices = sorted(dict_graph.keys())
    vertices = voronoi.vertices[indices]
    vertex_mapping = {indices[i]: i for i in range(len(indices))}

    # Create NetworkX graph
    graph = nx.Graph()
    for node_id, neighbors in dict_graph.items():
        mapped_node_id = vertex_mapping[node_id]
        graph.add_node(mapped_node_id, pos=vertices[mapped_node_id])
        for neighbor in neighbors:
            mapped_neighbor = vertex_mapping[neighbor]
            # Add edge with weight as euclidean distance
            weight = np.linalg.norm(
                vertices[mapped_node_id] - vertices[mapped_neighbor]
            )
            graph.add_edge(mapped_node_id, mapped_neighbor, weight=weight)

    return graph


def make_roadmap(
    problem: Problem, circle_approximation_num_sides: int = 32
) -> nx.Graph:
    """Generates sample trajectories, using num_trajectories for each agent."""
    all_points = _create_voronoi_polygon(problem, circle_approximation_num_sides)
    vor = Voronoi(all_points)
    return _get_voronoi_graph(problem, vor)


def generate_paths(
    graph: nx.Graph, start_position, end_position, num_paths
) -> list[np.ndarray]:
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()
    vertices = np.array(
        [graph_copy.nodes[node]["pos"] for node in sorted(graph_copy.nodes())]
    )

    start_and_goal = np.vstack([start_position, end_position])
    closest_to_start_id, closest_to_goal_id = np.argmin(
        np.linalg.norm(vertices[:, None, :] - start_and_goal[None, :, :], axis=-1),
        axis=0,
    )

    # Create extended vertices array with start and goal positions
    extended_vertices = np.concatenate([vertices, start_and_goal], axis=0)
    start_id = len(vertices)
    goal_id = len(vertices) + 1

    # Add start and goal nodes to the graph copy
    graph_copy.add_node(start_id, pos=start_position)
    graph_copy.add_node(goal_id, pos=end_position)

    # Add temporary edges to connect start and goal to the roadmap
    start_weight = np.linalg.norm(
        extended_vertices[start_id] - extended_vertices[closest_to_start_id]
    )
    goal_weight = np.linalg.norm(
        extended_vertices[goal_id] - extended_vertices[closest_to_goal_id]
    )

    graph_copy.add_edge(start_id, closest_to_start_id, weight=start_weight)
    graph_copy.add_edge(goal_id, closest_to_goal_id, weight=goal_weight)

    # Find paths on the extended graph
    topk_paths = topk_shortest_paths(graph_copy, start_id, goal_id, k=5)

    return topk_paths


def generate_sample_trajectories_demo():
    with open("instances_data/instances_shelf.json") as f:
        data = json.load(f)

    problem = Problem.from_json(data[14], "numpy")

    graph = make_roadmap(problem)
    vertices = np.array([graph.nodes[node]["pos"] for node in sorted(graph.nodes())])

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    # Add start and goal positions.
    agent_i = 1
    v0 = problem.agent_start_positions[agent_i]
    v1 = problem.agent_end_positions[agent_i]

    # plot graph
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            plt.plot(
                [vertices[node, 0], vertices[neighbor, 0]],
                [vertices[node, 1], vertices[neighbor, 1]],
                "k-",
            )

    # Use generate_paths to properly handle start and goal positions
    topk_paths = generate_paths(graph, v0, v1, num_paths=5)

    for path in topk_paths:
        plt.plot(path[:, 0], path[:, 1], "-", linewidth=2)

    plt.show()

    speed = 0.05
    dt = 1.0
    num_points = 64

    visualize(problem, plt.gca(), start_markersize=2, end_markersize=2)

    for path in topk_paths:
        traj = interpolate(path, dt, speed)
        if len(traj) > num_points:
            continue
        traj_points = np.zeros((num_points, 2))
        traj_points[: len(traj)] = traj
        traj_points[len(traj) :] = traj[-1]
        plt.plot(traj_points[:, 0], traj_points[:, 1], "-o", markersize=2)

    plt.show()


if __name__ == "__main__":
    generate_sample_trajectories_demo()
