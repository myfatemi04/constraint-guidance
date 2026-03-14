import networkx as nx
import numpy as np
from loguru import logger

from ael.problem import Problem
from ael.visgraphprior import generate_paths, interpolate, make_roadmap


def get_initial_paths_by_agent(
    problem: Problem, dt: float, graph: nx.Graph | None = None
):
    """
    This function returns an initial path for an agent, allowing the use of a custom graph.
    - `problem`: specifies obstacle locations, start and end positions, etc.
    - `dt`: the time step for interpolation of the paths.
    - `graph`: an optional pre-constructed graph to use for pathfinding. If None, a new graph will be constructed using `make_roadmap`.

    We allow supplying a custom `graph` so that conflict-based constraints can be applied.
    """
    if graph is None:
        graph = make_roadmap(problem)

    start_positions = problem.agent_start_positions
    end_positions = problem.agent_end_positions
    target_paths_by_agent = []
    for agent_index in range(problem.num_agents):
        try:
            paths = generate_paths(
                graph,
                start_positions[agent_index],
                end_positions[agent_index],
                num_paths=5,
            )
        except Exception as e:
            logger.warning(
                f"Failed to generate paths for agent {agent_index}, using straight line path: {e}"
            )
            paths = [
                np.linspace(
                    start_positions[agent_index],
                    end_positions[agent_index],
                    num=problem.num_timesteps,
                )
            ]
        paths_interpolated = []
        for path in paths:
            for speed in [
                0.2 * problem.agent_max_speeds[agent_index],
                0.5 * problem.agent_max_speeds[agent_index],
                0.8 * problem.agent_max_speeds[agent_index],
                problem.agent_max_speeds[agent_index],
            ]:
                path_length = np.linalg.norm(np.diff(path, axis=0), axis=-1).sum()
                min_speed = path_length / (problem.num_timesteps * dt) + 0.01
                speed = max(speed, min_speed)
                interpolated = interpolate(path, dt, speed)
                interpolated_path_timesteps = interpolated.shape[0]

                # Check if this fits within the time horizon, if not, skip it.
                if interpolated_path_timesteps > problem.num_timesteps:
                    print(interpolated_path_timesteps)
                    continue

                path = np.zeros((problem.num_timesteps, 2))
                path[:interpolated_path_timesteps] = interpolated
                path[interpolated_path_timesteps:] = interpolated[-1]
                paths_interpolated.append(path)

            if len(paths_interpolated) == 0:
                logger.warning(
                    "No path found for agent %d that fits within the time horizon, using straight line path.",
                    agent_index,
                )
                path = np.linspace(
                    start_positions[agent_index],
                    end_positions[agent_index],
                    num=problem.num_timesteps,
                )
                paths_interpolated.append(path)
        target_paths_by_agent.append(paths_interpolated[0])

    return np.array(target_paths_by_agent)
